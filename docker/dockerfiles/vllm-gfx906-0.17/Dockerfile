# =============================================================================
# vLLM 0.17.1 for AMD gfx906 (Radeon VII / Vega II / MI50 / MI60)
# =============================================================================
#
# Full from-scratch build: ROCm 7.0 + PyTorch (ROCm fork) + Triton + FA + vLLM
# AITER and MORI are excluded (gfx942/gfx950 only).
#
# Build:
#   podman build \
#     -t localhost/vllm-gfx906:0.17.1 \
#     -f docker/dockerfiles/vllm-gfx906-0.17/Dockerfile \
#     --build-arg VLLM_TAG=v0.17.1 \
#     .
#
# Estimated build time: 4-8 hours (PyTorch ~2-4h, FA ~1h, vLLM ~1-2h)
# =============================================================================

# ---------------------------------------------------------------------------
# Build arguments
# ---------------------------------------------------------------------------
ARG BASE_IMAGE=rocm/dev-ubuntu-22.04:7.0-complete

# Component versions (pinned to vLLM v0.17.1 compatibility)
ARG PYTORCH_REPO=https://github.com/ROCm/pytorch.git
ARG PYTORCH_BRANCH=89075173
ARG PYTORCH_VISION_REPO=https://github.com/pytorch/vision.git
ARG PYTORCH_VISION_BRANCH=v0.24.1
# Triton: use nalanzeyu's build (git34747741) which has confirmed gfx906 support.
# We extract it directly from the nalanzeyu image instead of building from source.
ARG TRITON_REPO=https://github.com/ROCm/triton.git
ARG TRITON_BRANCH=RI3.5.x
ARG FA_REPO=https://github.com/Dao-AILab/flash-attention.git
ARG FA_BRANCH=0e60e394

ARG VLLM_TAG=v0.17.1

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
# Base stage
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS base

ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:

# gfx906 ONLY — single arch = fastest possible build
ENV PYTORCH_ROCM_ARCH=gfx906

ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app
WORKDIR /app

# System dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less \
       libgfortran5 libopenmpi-dev libpci-dev libjpeg-dev \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || sleep 5; \
    done \
    && apt-get update -y \
    && apt-get install -y \
       python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
       python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-lib2to3 \
       python-is-python3 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && pip install -U packaging 'cmake<4' ninja wheel 'setuptools<80' pybind11 Cython \
    && rm -rf /var/lib/apt/lists/*


# ---------------------------------------------------------------------------
# Stage: Extract Triton from nalanzeyu image (git34747741, confirmed gfx906 support)
# Building Triton from source with RI3.5.x produces libtriton.so that has
# ConvertTritonAMDGPUToLLVM failing on gfx906. The nalanzeyu build (3.5.0+git34747741)
# works correctly. We repack it as a wheel-compatible directory install.
# ---------------------------------------------------------------------------
FROM docker.io/nalanzeyu/vllm-gfx906:latest AS triton_source

FROM base AS build_triton
# Copy Triton package from nalanzeyu torchenv into a staging dir
RUN --mount=type=bind,from=triton_source,src=/opt/torchenv/lib/python3.12/site-packages/triton,target=/triton_src \
    mkdir -p /app/triton_pkg \
    && cp -r /triton_src /app/triton_pkg/triton \
    && echo "Triton package staged: $(du -sh /app/triton_pkg/triton | cut -f1)"


# ---------------------------------------------------------------------------
# Stage: Build AMD SMI
# ---------------------------------------------------------------------------
FROM base AS build_amdsmi
RUN cd /opt/rocm/share/amd_smi \
    && pip wheel . --wheel-dir=dist \
    && mkdir -p /app/install && cp dist/*.whl /app/install


# ---------------------------------------------------------------------------
# Stage: Build PyTorch (ROCm fork) — gfx906 only
# ---------------------------------------------------------------------------
FROM base AS build_pytorch
ARG PYTORCH_BRANCH
ARG PYTORCH_REPO
ARG PYTORCH_VISION_BRANCH
ARG PYTORCH_VISION_REPO

RUN git clone ${PYTORCH_REPO} pytorch \
    && cd pytorch && git checkout ${PYTORCH_BRANCH} \
    && pip install -r requirements.txt \
    && git submodule update --init --recursive \
    && python3 tools/amd_build/build_amd.py \
    && CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') \
       python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl

RUN git clone ${PYTORCH_VISION_REPO} vision \
    && cd vision && git checkout ${PYTORCH_VISION_BRANCH} \
    && python3 setup.py bdist_wheel --dist-dir=dist \
    && pip install dist/*.whl

RUN mkdir -p /app/install \
    && cp /app/pytorch/dist/*.whl /app/install \
    && cp /app/vision/dist/*.whl /app/install


# ---------------------------------------------------------------------------
# Stage: Build FlashAttention — gfx906 only
# FA has a hardcoded GPU allowlist that excludes gfx906 — patch it before build
# ---------------------------------------------------------------------------
FROM base AS build_fa
ARG FA_BRANCH
ARG FA_REPO
COPY docker/dockerfiles/vllm-gfx906-0.17/patches/patch-fa-gfx906.py /app/patch-fa-gfx906.py
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    pip install /install/*.whl
RUN git clone ${FA_REPO} \
    && cd flash-attention && git checkout ${FA_BRANCH} \
    && git submodule update --init \
    && python3 /app/patch-fa-gfx906.py \
    && GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist \
    && mkdir -p /app/install && cp dist/*.whl /app/install


# ---------------------------------------------------------------------------
# Stage: Extract rocBLAS TensileLibrary for gfx906 from nalanzeyu image
# ROCm 7.0 dropped gfx906 from its rocBLAS library — we source them from the
# ROCm 6.3 nalanzeyu image where they ship inside the PyTorch env
# ---------------------------------------------------------------------------
FROM docker.io/nalanzeyu/vllm-gfx906:latest AS rocblas_gfx906
RUN mkdir -p /tensile_gfx906 \
    && find /opt/torchenv/lib/python3.12/site-packages/torch/lib/rocblas/library/ \
       -name "*gfx906*" -exec cp {} /tensile_gfx906/ \; \
    && echo "Tensile files for gfx906:" && ls /tensile_gfx906/

# ---------------------------------------------------------------------------
# Stage: Collect all dependency wheels (excludes Triton — installed via dir copy)
# ---------------------------------------------------------------------------
FROM base AS deps
RUN mkdir -p /app/debs
RUN --mount=type=bind,from=build_amdsmi,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_pytorch,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs
RUN --mount=type=bind,from=build_fa,src=/app/install/,target=/install \
    cp /install/*.whl /app/debs


# ---------------------------------------------------------------------------
# Stage: Install deps + Build vLLM with gfx906 patches
# ---------------------------------------------------------------------------
FROM base AS build_vllm

# Install all pre-built wheels (PyTorch, FA, AMDsmi — NOT triton, installed below)
RUN --mount=type=bind,from=deps,src=/app/debs,target=/install \
    pip install /install/*.whl

# Install Triton from nalanzeyu package dir (git34747741, confirmed gfx906 support)
RUN --mount=type=bind,from=build_triton,src=/app/triton_pkg/triton,target=/triton_pkg \
    cp -r /triton_pkg /usr/local/lib/python3.12/dist-packages/triton \
    && python3 -c "import triton; print('Triton installed:', triton.__version__)"

# Install uv for fast dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
ENV UV_HTTP_TIMEOUT=500
ENV UV_LINK_MODE=copy

# Clone upstream vLLM
ARG VLLM_TAG
RUN git clone --depth 1 --branch ${VLLM_TAG} \
    https://github.com/vllm-project/vllm.git vllm

# Copy and apply gfx906 patches
# cache-bust: add rocBLAS TensileLibrary for gfx906 from nalanzeyu
COPY docker/dockerfiles/vllm-gfx906-0.17/patches/ /app/patches/
RUN chmod +x /app/patches/apply-gfx906-patches.sh \
    && chmod +x /app/patches/patch-fa-gfx906.py \
    && cd vllm && bash /app/patches/apply-gfx906-patches.sh

# Install vLLM dependencies and build
RUN cd vllm \
    && pip install -r requirements/rocm.txt \
    && python3 setup.py clean --all \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_TAG#v}" \
       python3 setup.py bdist_wheel --dist-dir=dist


# ---------------------------------------------------------------------------
# Final image
# ---------------------------------------------------------------------------
FROM base AS final

# Install all wheels (PyTorch, FA, AMDsmi)
RUN --mount=type=bind,from=deps,src=/app/debs,target=/install \
    pip install /install/*.whl

RUN pip install --upgrade pip huggingface-hub[cli]

# Install Triton from nalanzeyu package dir (gfx906 compatible)
RUN --mount=type=bind,from=build_triton,src=/app/triton_pkg/triton,target=/triton_pkg \
    cp -r /triton_pkg /usr/local/lib/python3.12/dist-packages/triton \
    && python3 -c "import triton; print('Triton installed:', triton.__version__)"

# rocBLAS TensileLibrary for gfx906 (missing from ROCm 7.0, sourced from ROCm 6.3)
RUN --mount=type=bind,from=rocblas_gfx906,src=/tensile_gfx906/,target=/tensile_gfx906 \
    ROCBLAS_LIB=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'rocblas', 'library'))") \
    && mkdir -p "$ROCBLAS_LIB" \
    && cp /tensile_gfx906/* "$ROCBLAS_LIB/" \
    && echo "Installed $(ls /tensile_gfx906/ | wc -l) TensileLibrary files for gfx906"

# Install vLLM wheel + runtime deps
RUN --mount=type=bind,from=build_vllm,src=/app/vllm/dist/,target=/vllm_dist \
    --mount=type=bind,from=build_vllm,src=/app/vllm/requirements/,target=/vllm_reqs \
    pip install -r /vllm_reqs/rocm.txt \
    && pip install /vllm_dist/*.whl

# Copy gfx906 MoE kernel configs
COPY docker/dockerfiles/vllm-gfx906-0.17/patches/moe_configs/ /app/moe_configs/

# ---------------------------------------------------------------------------
# gfx906 environment
# ---------------------------------------------------------------------------
# Required: makes ROCm treat gfx906 as a supported target
ENV HSA_OVERRIDE_GFX_VERSION=9.0.6
# Performance tuning
ENV HIP_FORCE_DEV_KERNARG=1
ENV TOKENIZERS_PARALLELISM=false
ENV SAFETENSORS_FAST_GPU=1
# Disable bfloat16 autocast (gfx906 has no bf16 hardware)
ENV VLLM_ROCM_CUSTOM_PAGED_ATTN=1
# Required for RCCL in ROCm 7.x
ENV HSA_NO_SCRATCH_RECLAIM=1

# MIOpen workarounds
ENV MIOPEN_DEBUG_CONV_DIRECT=0
ENV MIOPEN_DEBUG_CONV_GEMM=0

# ROCm profiler config
RUN echo "ROCTRACER_MAX_EVENTS=10000000" > /app/libkineto.conf
ENV KINETO_CONFIG=/app/libkineto.conf

WORKDIR /app

# Metadata
ARG VLLM_TAG
RUN echo "vLLM: ${VLLM_TAG}" > /app/versions.txt \
    && echo "ROCM_ARCH: gfx906" >> /app/versions.txt \
    && echo "BASE: ${BASE_IMAGE:-rocm/dev-ubuntu-22.04:7.0-complete}" >> /app/versions.txt \
    && python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" >> /app/versions.txt

CMD ["/bin/bash"]

# ---------------------------------------------------------------------------
# OpenAI-compatible entrypoint variant
# ---------------------------------------------------------------------------
FROM final AS vllm-openai
ENTRYPOINT ["vllm", "serve"]
