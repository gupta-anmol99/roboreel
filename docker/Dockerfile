# Base: Ubuntu 22.04 with CUDA 12.1 toolkit (needed to compile pointnet/torch extensions)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH
# Explicit CUDA arch list — required for any torch CUDA extension compiled during
# docker build, because there is no GPU available at build time and PyTorch's
# auto-detection returns an empty list, causing an IndexError.
# Covers: Volta(7.0), Turing(7.5), Ampere(8.0/8.6), Ada(8.9), Hopper(9.0)
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ─── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    wget curl git git-lfs \
    build-essential cmake \
    # OpenGL / display
    libgl1-mesa-glx libgl1-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    libegl1-mesa libgles2 \
    libx11-dev libxau-dev libxdmcp-dev \
    python3-opengl \
    # Display testing tools
    x11-apps mesa-utils \
    # Misc utils
    unzip ca-certificates \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/*

# ─── Miniforge (replaces Miniconda) ──────────────────────────────────────────
# Using Miniforge instead of Miniconda: identical API, no TOS/license prompt
# that breaks non-interactive Docker builds in newer Miniconda versions.
RUN wget -q https://github.com/conda-forge/miniforge/releases/download/26.1.1-3/Miniforge3-26.1.1-3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    /opt/conda/bin/conda clean -afy && \
    /opt/conda/bin/conda init bash

# ─── 'tasks' env (from environment.yml) ──────────────────────────────────────
COPY environment.yml /tmp/environment.yml
COPY docker/split_env.py /tmp/split_env.py

# Split environment.yml into:
#   /tmp/conda_only.yml       — conda packages only (no pip section)
#   /tmp/pip_requirements.txt — pip packages separately
# This avoids "CondaEnvException: Pip failed" which swallows the real error
# when conda drives pip internally.
RUN sed -i '/^prefix:/d' /tmp/environment.yml && \
    /opt/conda/bin/python3 /tmp/split_env.py

# Step 1: create the conda env (conda packages only — fast, no conflicts)
RUN /opt/conda/bin/conda env create -f /tmp/conda_only.yml

# Step 2: install pip packages separately.
# --no-deps: environment.yml is a fully-resolved export, no need to re-solve deps.
# --no-capture-output: makes conda show the real pip error instead of swallowing it.
RUN /opt/conda/bin/conda run --no-capture-output -n tasks \
    pip install --no-deps -r /tmp/pip_requirements.txt

# Install mplib 2.1.1 — it requires numpy 1.26.4 but the env has 2.4.4.
# We install mplib first (skipping auto dep-resolution to avoid pip touching numpy),
# then force-downgrade numpy to what mplib needs.
RUN /opt/conda/bin/conda run -n tasks \
    pip install mplib==0.2.1 --no-deps

RUN /opt/conda/bin/conda run -n tasks \
    pip install "numpy==1.26.4" --force-reinstall

# ─── 'GraspGen' env ───────────────────────────────────────────────────────────
# Exact package lock + pip constraints, so this env matches the tested image.
COPY docker/graspgen_conda_explicit.txt /tmp/graspgen_conda_explicit.txt
COPY docker/graspgen_pip_constraints.txt /tmp/graspgen_pip_constraints.txt

RUN /opt/conda/bin/conda create -n GraspGen --file /tmp/graspgen_conda_explicit.txt -y

# PyTorch 2.1.0 + CUDA 12.1 wheels + torch-geometric extensions
RUN /opt/conda/bin/conda run -n GraspGen pip install \
    -c /tmp/graspgen_pip_constraints.txt \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torch-cluster \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Clone GraspGen to /opt/GraspGen at build time.
RUN git clone https://github.com/NVlabs/GraspGen.git /opt/GraspGen && \
    git -C /opt/GraspGen checkout 0c8066e478ece8920a8b624dc2d0544a07ed9f10

RUN /opt/conda/bin/conda run -n GraspGen \
    pip install -c /tmp/graspgen_pip_constraints.txt omegaconf && \
    /opt/conda/bin/conda run -n GraspGen \
    pip install -c /tmp/graspgen_pip_constraints.txt -e /opt/GraspGen

# Compile PointNet++ CUDA ops (requires NVCC from the cuda-devel base image)
WORKDIR /opt/GraspGen
RUN /opt/conda/bin/conda run -n GraspGen \
    bash ./install_pointnet.sh

# ─── nvdiffrast ───────────────────────────────────────────────────────────────
RUN /opt/conda/bin/conda run -n GraspGen \
    pip install -c /tmp/graspgen_pip_constraints.txt setuptools wheel ninja

RUN /opt/conda/bin/conda run --no-capture-output -n GraspGen \
    pip install -c /tmp/graspgen_pip_constraints.txt \
    git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation

# ─── diff-gaussian-rasterization (from mip-splatting) ────────────────────────
RUN git clone https://github.com/autonomousvision/mip-splatting.git /opt/mip-splatting && \
    git -C /opt/mip-splatting checkout dda02ab5ecf45d6edb8c540d9bb65c7e451345a9

WORKDIR /opt/mip-splatting/submodules/diff-gaussian-rasterization
RUN /opt/conda/bin/conda run --no-capture-output -n GraspGen \
    python setup.py install

# ─── GraspGenModels (cloned inside GraspGen repo) ─────────────────────────────
RUN git clone https://huggingface.co/adithyamurali/GraspGenModels \
    /opt/GraspGen/GraspGenModels && \
    git -C /opt/GraspGen/GraspGenModels checkout ec1ccbb5eec0680db669246ac312a3636f16ee43

# ─── Shell + entrypoint setup ─────────────────────────────────────────────────
# Auto-activate tasks env for interactive shells
RUN echo "conda activate tasks" >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
