FROM docker.io/nalanzeyu/vllm-gfx906:latest

# Upgrade transformers to 5.x for qwen3_5 architecture support.
# transformers 5.0 renamed ALLOWED_LAYER_TYPES; vllm files are patched in the fork.
RUN pip install "transformers>=5.0.0,<5.4.0"

# Install the fork's vllm source over the base image's installed package.
# Build context must be the fork root: podman build -f <this-file> /path/to/vllm-gfx906/
# This picks up all gfx906 patches including Qwen3.5 (GatedDeltaNet) support.
COPY vllm/ /opt/torchenv/lib/python3.12/site-packages/vllm/

# Patch compressed-tensors QuantizationArgs for forward-compat with >= 0.14.x:
# adds zp_dtype + scale_dtype fields and relaxes extra="forbid" → extra="ignore"
# so models quantized with newer compressed-tensors (e.g. cyankiwi/Qwen3.5-9B-AWQ-4bit)
# can be loaded. This patches the external compressed-tensors package, not vllm.
COPY docker/dockerfiles/vllm-gfx906-qwen35/quant_args.py \
     /opt/torchenv/lib/python3.12/site-packages/compressed_tensors/quantization/quant_args.py
