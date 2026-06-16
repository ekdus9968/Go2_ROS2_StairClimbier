FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Python 3.11 via deadsnakes
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git wget curl vim sudo \
    build-essential \
    libgl1-mesa-glx libglfw3 libosmesa6 \
    libegl1 libopengl0 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN python -m ensurepip --upgrade && \
    python -m pip install --upgrade pip

# JAX with CUDA 12 (Blackwell sm_120)
RUN python -m pip install --upgrade "jax[cuda12]" \
    nvidia-cudnn-cu12 nvidia-cublas-cu12

# MuJoCo stack
RUN python -m pip install \
    mujoco mujoco-mjx mujoco-warp warp-lang

# mujoco_playground (Python 3.11+ requirment)
RUN python -m pip install --ignore-installed git+https://github.com/google-deepmind/mujoco_playground.git

# PGTT dependency
RUN python -m pip install \
    brax flax optax \
    numpy scipy matplotlib opencv-python tqdm \
    dm_control jaxopt

# Workspace
WORKDIR /workspace
RUN git clone https://github.com/NtagkasAlex/phase_guided_terrain_traversal.git pgtt
RUN mkdir -p /workspace/go2-stair-climber

# Environment
ENV PYTHONPATH=/workspace/pgtt:/workspace/go2-stair-climber:$PYTHONPATH
ENV JAX_PLATFORMS=cuda
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
