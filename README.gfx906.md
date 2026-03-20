# vLLM for AMD gfx906 (Vega II / Radeon VII / MI50 / MI60)

> **Maintainer:** [@kubeages](https://github.com/kubeages)  
> **Base:** [nalanzeyu/vllm-gfx906](https://hub.docker.com/r/nalanzeyu/vllm-gfx906) (archived) — this fork continues the work.

This fork patches vLLM 0.12.0 to support AMD gfx906 GPUs (Vega II architecture, 32GB HBM2) including support for modern model architectures like **Qwen3.5** (GatedDeltaNet hybrid attention).

---

## What's different from upstream

| Patch | Description |
|---|---|
| `transformers >= 5.0` support | `ALLOWED_LAYER_TYPES` → `ALLOWED_MLP_LAYER_TYPES` rename compatibility |
| **Qwen3.5 support** | Full `Qwen3_5ForConditionalGeneration` implementation with GatedDeltaNet (linear attention) layers |
| gfx906 ROCm patches | Flash attention tiling, WNA16 fallback, Triton attention tile sizing |

---

## Quick start (Docker)

```bash
# Build the image (from repo root)
sudo podman build \
  -t localhost/vllm-gfx906:latest \
  -f docker/dockerfiles/vllm-gfx906-qwen35/Dockerfile \
  .

# Run Qwen3.5-9B
sudo podman run -d \
  --name=vllm \
  --security-opt label=disable \
  --group-add keep-groups \
  --device /dev/kfd:/dev/kfd \
  --device /dev/dri:/dev/dri \
  -e HSA_OVERRIDE_GFX_VERSION=9.0.6 \
  -e HF_HOME=/models \
  -v /path/to/model/cache:/models \
  -p 8000:8000 \
  --shm-size 16g \
  localhost/vllm-gfx906:latest \
  vllm serve Qwen/Qwen3.5-9B \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen35-9b \
  --enable-auto-tool-choice \
  --tool-call-parser=hermes
```

---

## Tested models (Vega II / 32GB HBM2)

| Model | Size | VRAM | TTFT | Throughput | Status |
|---|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 9B BF16 | ~18GB | ~0.25s | ~21 tok/s | ✅ |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | 32B AWQ | ~20GB | ~0.8s | ~30 tok/s | ✅ |
| `Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` | 32B AWQ | ~20GB | ~0.8s | ~30 tok/s | ✅ |
| `casperhansen/deepseek-r1-distill-qwen-32b-awq` | 32B AWQ | ~20GB | ~1.0s | ~28 tok/s | ✅ |
| `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` | 30B MoE AWQ | ~8GB | ~0.3s | ~45 tok/s | ✅ |

### Notes on Qwen3.5-9B

- Uses **GatedDeltaNet** (linear attention) for 24/32 layers — requires this fork
- **Thinking mode** enabled by default. Disable with `chat_template_kwargs: {"enable_thinking": false}`
- `cyankiwi/Qwen3.5-9B-AWQ-4bit` is **not compatible** (uses `compressed-tensors` format, not AWQ)

---

## Build from source

The Docker image is based on `nalanzeyu/vllm-gfx906:latest` (vLLM 0.12.0 + ROCm 6.3, precompiled for gfx906) with Python patches applied on top. This avoids the 6-12h ROCm/PyTorch compilation.

```dockerfile
FROM docker.io/nalanzeyu/vllm-gfx906:latest
RUN pip install --upgrade transformers
COPY vllm/ /opt/torchenv/lib/python3.12/site-packages/vllm/
```

See `docker/dockerfiles/vllm-gfx906-qwen35/Dockerfile`.

---

## Hardware requirements

- AMD GPU with gfx906 architecture (Radeon VII, Vega II, MI50, MI60)
- ROCm 6.3+
- 32GB VRAM recommended for 9B BF16 models

---

## Roadmap

- [ ] AWQ support for Qwen3.5 (blocked on `compressed-tensors` format compatibility)
- [ ] Qwen3.5 vision model support  
- [ ] Automatic build CI when upstream nalanzeyu releases new base image
- [ ] Support for newer vLLM versions (0.13+)
