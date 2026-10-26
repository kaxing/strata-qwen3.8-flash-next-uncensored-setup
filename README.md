# Qwen Flash Next IQ4 on Strata

Local setup for `AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF`, quant `AD-3.86bpw-IQ4_XS`.

- Verified hardware: Linux x86-64, RTX 3080 Ti Laptop (16 GiB VRAM), 64 GiB RAM, NVMe SSD.
- Model download: 32 shards, 79.4 GiB. Allow at least 100 GiB free for weights, pack, and build.
- Prerequisites: Git, curl, tar, CMake, GCC/G++ 12, CUDA toolkit (`nvcc`), NVIDIA driver, Python 3, and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- This setup uses a local CUDA build. The packaged engine rejects this model's Q4_1 n-gram table.
- Context: 65,536 tokens, Q4 KV cache. Short responses and Pi tool calls worked. Full-length 64K inference remains unverified.

Run these steps in one Bash terminal, from this guide's directory.

## 1. Clone Strata

```bash
git clone https://github.com/Niko1221/Strata.git Strata
cd Strata
git checkout 7dd5cf18e68c639a8773e096c5c79f971ea2a9f5
export ROOT="$PWD"
```

The pinned revision matches the verified setup. Do not apply this patch blindly to newer revisions.

## 2. Apply the local patch and build

The patch adds Q4_1 n-gram decoding. It does not convert model weights.

```bash
git apply --check ../q4_1.patch
git apply ../q4_1.patch

mkdir -p third_party/llama.cpp
curl -fL https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz \
  -o /tmp/strata-llama-source.tar.gz
tar -xzf /tmp/strata-llama-source.tar.gz --strip-components=1 -C third_party/llama.cpp

cmake -S . -B build-local \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
  -DSTRATA_GGML_DIR="$ROOT/third_party/llama.cpp"
cmake --build build-local --target strata ple_reader_test -j 6
./build-local/ple_reader_test --selftest
```

`86` is the CUDA architecture for this RTX 3080 Ti. Adjust the architecture and compiler paths for other hardware.

## 3. Download all shards

```bash
export MODEL=Qwen3.8-Flash-Next-Uncensored-AD-3.86bpw-IQ4_XS
uvx --from huggingface_hub hf download \
  AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF \
  --include "$MODEL/*" --local-dir "$ROOT/models"
export GGUF="$ROOT/models/$MODEL/$MODEL-00001-of-00032.gguf"
```

Repeat the download command after an interruption. Keep all 32 shards together.

## 4. Create the compatibility pack

```bash
uv run --no-project --with-requirements requirements.txt \
  python tools/iq_pack.py --gguf "$GGUF" \
  --out "$ROOT/packs/qfn" --compat-bf16
```

The pack references the GGUF shards. Do not delete or move them afterward.

## 5. Create the server configuration

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
r = Path(os.environ['ROOT'])
config = {
    'exe': str(r / 'build-local/strata'),
    'args': [
        '--pack', str(r / 'packs/qfn'),
        '--native', os.environ['GGUF'],
        '--expert-profile', str(r / 'data/expert-profile.bin'),
        '--max-context', '65536', '--kv', 'q4_0',
        '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '2',
        '--mmap-experts', '--pool-workers', '9'
    ],
    'cwd': str(r),
    'tokenizer': str(r / 'packs/qfn/tokenizer'),
    'model_name': os.environ['MODEL'],
    'log': str(r / 'qfn-engine.log'),
    'host': '127.0.0.1', 'port': 8080, 'gpu': 0,
    'engine_ready_s': 3600
}
(r / 'qfn.json').write_text(json.dumps(config, indent=2) + '\n')
PY
```

- `--expert-profile`: required here for speculative verification's VRAM tier.
- `--spec 2`: required for this native IQ pack.
- `--mmap-experts`: lets the engine use experts outside VRAM.
- `--pool-workers 9`: matches this machine. Adjust for your CPU.
- `--max-context`: reduce to `32768` if memory is insufficient. Larger contexts can reduce speed.

## 6. Start the server

```bash
uv run --no-project --with-requirements requirements.txt \
  python serve/server.py --engine strata --config qfn.json
```

Keep this terminal open. Use Ctrl-C to stop. The API listens only on localhost.

## 7. Check from another terminal

```bash
curl -fsS http://127.0.0.1:8080/v1/status
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen3.8-Flash-Next-Uncensored-AD-3.86bpw-IQ4_XS",
    "messages":[{"role":"user","content":"What is 2+2? Reply with the number only."}],
    "max_tokens":64,
    "chat_template_kwargs":{"enable_thinking":false}
  }'
```

Wait for `"loaded": true`. The expected response is `4`.

For Pi or another client, use `http://127.0.0.1:8080/v1`, the exact model ID above, and a dummy API key. Set the context limit to `65536`, output limit to `4096`, and `chat_template_kwargs.enable_thinking` to `false`. Pi uses API type `openai-completions`.

## Known limits

- Vision is disabled. This guide covers text and tool calls only.
- The compatibility patch and engine stay local. No upstream submission is required.
- This guide records the verified build, pack, and server settings. A clean installation of every command together remains unverified.
