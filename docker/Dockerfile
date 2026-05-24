# Base: NVIDIA CUDA 12.8 Ubuntu 22.04 (Blackwell support)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# System tools
RUN apt-get update && apt-get install -y \
    git wget curl vim sudo \
    build-essential \
    python3.10 python3-pip python3-venv \
    libgl1-mesa-glx libglfw3 libosmesa6 \
    libegl1 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# JAX with CUDA 12 (Blackwell sm_120 compatible)
RUN pip install --upgrade "jax[cuda12]" && \
    pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# MuJoCo + MJX
RUN pip install mujoco mujoco-mjx

# PGTT dependencies
RUN pip install \
    brax flax optax \
    numpy scipy matplotlib opencv-python tqdm \
    dm_control jaxopt

# Workspace
WORKDIR /workspace

# Clone PGTT (mounted at runtime if user wants to edit)
RUN git clone https://github.com/NtagkasAlex/phase_guided_terrain_traversal.git pgtt

# Project mount point
RUN mkdir -p /workspace/go2-stair-climber

# Environment variables
ENV PYTHONPATH=/workspace/pgtt:/workspace/go2-stair-climber:$PYTHONPATH
ENV JAX_PLATFORMS=cuda
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

# Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
