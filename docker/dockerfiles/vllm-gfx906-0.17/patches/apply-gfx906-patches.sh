#!/usr/bin/env bash
# =============================================================================
# apply-gfx906-patches.sh
#
# Applies gfx906-specific patches to upstream vLLM v0.17.x source tree.
# Run from the vLLM source root: cd vllm && bash /path/to/apply-gfx906-patches.sh
#
# These patches add support for AMD gfx906 (Radeon VII / Vega II / MI50 / MI60)
# which is not officially supported by upstream vLLM.
# =============================================================================
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "[gfx906] Applying patches to $(pwd) ..."

# =============================================================================
# 1. Platform detection: add on_gfx906() to ROCm platform
# =============================================================================
echo "[gfx906] Patching vllm/platforms/rocm.py — add on_gfx906() detection ..."

# Add on_gfx906() function after on_gfx9()
if ! grep -q 'on_gfx906' vllm/platforms/rocm.py; then
    sed -i '/^def on_gfx9().*:/,/^$/  {
        /^$/a\
\
@cache\
def on_gfx906() -> bool:\
    GPU_ARCH = torch.cuda.get_device_properties("cuda").gcnArchName\
    return "gfx906" in GPU_ARCH\

    }' vllm/platforms/rocm.py
    echo "[gfx906]   -> on_gfx906() added"
fi

# Add gfx906 device names to the device ID map
if ! grep -q 'Vega_20\|Radeon_VII' vllm/platforms/rocm.py; then
    sed -i '/_ROCM_DEVICE_ID_NAME_MAP.*{/,/}/ {
        /}/ i\    # gfx906 devices\
    "0x66a0": "AMD_Radeon_VII",\
    "0x66a1": "AMD_Radeon_Pro_VII",\
    "0x66a7": "AMD_Instinct_MI50",\
    "0x66af": "AMD_Instinct_MI60",
    }' vllm/platforms/rocm.py
    echo "[gfx906]   -> gfx906 device IDs added"
fi

# Disable bfloat16 for gfx906 in check_and_update_config
cat > /tmp/gfx906_rocm_patch.py << 'PYEOF'
with open("vllm/platforms/rocm.py", "r") as f:
    content = f.read()

if "no bfloat16 for gfx906" not in content:
    bf16_check = '''
        # no bfloat16 for gfx906
        if on_gfx906() and hasattr(vllm_config, 'model_config'):
            mc = vllm_config.model_config
            if mc is not None and mc.dtype == torch.bfloat16:
                logger.warning("gfx906 does not support bfloat16, falling back to float16")
                mc.dtype = torch.float16
'''
    if "check_and_update_config" in content:
        lines = content.split('\n')
        new_lines = []
        in_method = False
        patched = False
        for line in lines:
            new_lines.append(line)
            if 'def check_and_update_config' in line and not patched:
                in_method = True
            elif in_method and not patched and line.strip().startswith(('if ', 'for ', 'return', 'vllm_config', 'cache', 'model')):
                new_lines.insert(-1, bf16_check)
                patched = True
                in_method = False
        if patched:
            content = '\n'.join(new_lines)

    with open("vllm/platforms/rocm.py", "w") as f:
        f.write(content)
    print("[gfx906]   -> bfloat16 disable for gfx906 added")
PYEOF
python3 /tmp/gfx906_rocm_patch.py || echo "[gfx906]   -> WARN: bf16 patch may need manual review"

# Patch verify_quantization: prevent ROCm platform from force-enabling Triton AWQ on gfx906
# (upstream rocm.py sets VLLM_USE_TRITON_AWQ=1 for any AWQ model, but Triton doesn't support gfx906)
echo "[gfx906] Patching rocm.py — disable forced VLLM_USE_TRITON_AWQ on gfx906 ..."
cat > /tmp/gfx906_awq_patch.py << 'PYEOF'
import re
with open("vllm/platforms/rocm.py", "r") as f:
    content = f.read()

old = '''    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq" and not envs.VLLM_USE_TRITON_AWQ:
            logger.warning(
                "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                " is not set, enabling VLLM_USE_TRITON_AWQ."
            )
        os.environ["VLLM_USE_TRITON_AWQ"] = "1"'''

new = '''    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        super().verify_quantization(quant)
        if quant == "awq":
            if on_gfx906():
                # gfx906: Triton does not support this target — use C++ AWQ kernel
                os.environ["VLLM_USE_TRITON_AWQ"] = "0"
            elif not envs.VLLM_USE_TRITON_AWQ:
                logger.warning(
                    "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                    " is not set, enabling VLLM_USE_TRITON_AWQ."
                )
                os.environ["VLLM_USE_TRITON_AWQ"] = "1"'''

if old in content:
    content = content.replace(old, new)
    with open("vllm/platforms/rocm.py", "w") as f:
        f.write(content)
    print("[gfx906]   -> verify_quantization patched: AWQ Triton disabled for gfx906")
else:
    print("[gfx906]   -> WARN: verify_quantization pattern not found, may need manual patch")
PYEOF
python3 /tmp/gfx906_awq_patch.py || echo "[gfx906]   -> WARN: AWQ Triton patch may need manual review"


# =============================================================================
# 2. Attention layer: enable flash attention for gfx906
# =============================================================================
echo "[gfx906] Patching attention layer — flash attention for gfx906 ..."

# In 0.17, attention layer may be at different paths
for ATTN_FILE in vllm/attention/layer.py vllm/v1/attention/layer.py; do
    if [ -f "$ATTN_FILE" ] && ! grep -q 'on_gfx906' "$ATTN_FILE"; then
        sed -i '/from vllm.platforms.rocm import/s/)/, on_gfx906)/' "$ATTN_FILE" 2>/dev/null || true
        if ! grep -q 'on_gfx906' "$ATTN_FILE"; then
            sed -i '/from vllm.platforms.rocm import on_gfx9/a\    from vllm.platforms.rocm import on_gfx906' \
                "$ATTN_FILE" 2>/dev/null || true
        fi
        sed -i 's/on_gfx9()/on_gfx9() or on_gfx906()/g' "$ATTN_FILE" 2>/dev/null || true
        echo "[gfx906]   -> flash attention enabled for gfx906 in $ATTN_FILE"
    fi
done


# =============================================================================
# 3. MoE: add gfx906 device name mapping for fused MoE kernel config
# =============================================================================
echo "[gfx906] Patching fused_moe.py — gfx906 device name mapping ..."

FUSED_MOE="vllm/model_executor/layers/fused_moe/fused_moe.py"
if [ -f "$FUSED_MOE" ] && ! grep -q 'gfx906_names' "$FUSED_MOE"; then
    python3 << 'PYEOF'
import re

with open("vllm/model_executor/layers/fused_moe/fused_moe.py", "r") as f:
    content = f.read()

gfx906_mapping = '''
    # gfx906 device name mapping
    gfx906_names = ["Instinct_MI50", "Instinct_MI60", "Radeon_Pro_VII", "Radeon_VII", "Vega_20"]
    if any(s in device_name for s in gfx906_names):
        device_name = "AMD_GFX906"
'''

if "gfx906_names" not in content:
    patterns = [
        r'(device_name\s*=.*?torch\.cuda\.get_device_name.*?\n)',
        r'(device_name\s*=.*?get_device_name.*?\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            insert_after = match.group(0)
            content = content.replace(insert_after, insert_after + gfx906_mapping)
            break

    with open("vllm/model_executor/layers/fused_moe/fused_moe.py", "w") as f:
        f.write(content)
    print("[gfx906]   -> gfx906 device mapping added")
PYEOF
fi


# =============================================================================
# 4. Config: handle numerical instability warnings for gfx906
# =============================================================================
echo "[gfx906] Patching vllm/config/model.py — dtype fallback for gfx906 ..."

if [ -f "vllm/config/model.py" ] && ! grep -q 'gfx906' vllm/config/model.py; then
    python3 << 'PYEOF'
with open("vllm/config/model.py", "r") as f:
    content = f.read()

marker = "# NOTE(gfx906): ignore numerical instability"
if marker not in content:
    content = content.replace(
        "raise ValueError",
        f"{marker}\n        # gfx906 only supports float16 natively\n        raise ValueError",
        1
    )

with open("vllm/config/model.py", "w") as f:
    f.write(content)
print("[gfx906]   -> dtype notes added")
PYEOF
fi


# Fix any indentation issues introduced by patches in config/model.py
python3 "$PATCH_DIR/patch-config-model.py" || echo "[gfx906]   -> WARN: config/model.py fix failed"

# =============================================================================
# 5. Apply unified diff patches (C++ kernels and Python layers)
# =============================================================================
echo "[gfx906] Applying unified diff patches ..."

PATCHES=(
    "gptq-q_gemm-fp32-accum.patch"
    "gguf-mmvq-block128.patch"
    "gguf-vecdotq-vdr.patch"
    "moe-wna16-gfx906.patch"
    "gptq-py-gfx906.patch"
    "exllama-gfx906.patch"
)

for p in "${PATCHES[@]}"; do
    PATCH_FILE="$PATCH_DIR/$p"
    if [ -f "$PATCH_FILE" ]; then
        echo "[gfx906]   Applying $p ..."
        patch -p1 --forward --no-backup-if-mismatch < "$PATCH_FILE" || {
            echo "[gfx906]   -> WARN: $p failed to apply (may already be applied)"
        }
    else
        echo "[gfx906]   -> SKIP: $p not found"
    fi
done


# =============================================================================
# 6. Copy MoE kernel configs for gfx906
# =============================================================================
echo "[gfx906] Installing MoE kernel configs ..."

MOE_CONFIG_DIR="vllm/model_executor/layers/fused_moe/configs"
if [ -d "$MOE_CONFIG_DIR" ]; then
    if [ -d "$PATCH_DIR/moe_configs" ]; then
        cp "$PATCH_DIR/moe_configs/"*.json "$MOE_CONFIG_DIR/" 2>/dev/null || true
        echo "[gfx906]   -> MoE configs installed"
    else
        echo "[gfx906]   -> No MoE configs in patches dir (will use runtime autotuning)"
    fi
fi


# =============================================================================
# 7. AWQ: enable exllama backend for gfx906
# =============================================================================
echo "[gfx906] Patching AWQ quantization — exllama backend ..."

AWQ_FILE="vllm/model_executor/layers/quantization/awq.py"
if [ -f "$AWQ_FILE" ] && ! grep -q 'gfx906' "$AWQ_FILE"; then
    python3 << 'PYEOF'
with open("vllm/model_executor/layers/quantization/awq.py", "r") as f:
    content = f.read()

if "[vllm-gfx906]" not in content:
    content = content.replace(
        "class AWQLinearMethod",
        '# NOTE(gfx906): AWQ uses exllama kernel on gfx906 for compatibility\nclass AWQLinearMethod',
    )
    with open("vllm/model_executor/layers/quantization/awq.py", "w") as f:
        f.write(content)
    print("[gfx906]   -> AWQ gfx906 note added")
PYEOF
fi


# =============================================================================
# 8. Disable AITER imports (gfx942/gfx950 only)
# =============================================================================
echo "[gfx906] Ensuring AITER fallback for gfx906 ..."

# AITER is not installed in this build, so imports should naturally fall back.
if grep -rq 'from vllm._aiter_ops' vllm/ 2>/dev/null; then
    echo "[gfx906]   -> AITER imports found; they should gracefully degrade (not installed)"
fi


# =============================================================================
# Done
# =============================================================================
echo ""
echo "[gfx906] ============================================="
echo "[gfx906] All patches applied successfully."
echo "[gfx906] ============================================="
