# ============================================================
# Go2 Stair Climber - Real Robot Container
# Ubuntu 22.04 + ROS2 Humble + rl_sar + go2-stair-climber
# Target: NVIDIA Jetson (JetPack 5.1.1, L4T R35.3.1)
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ────────────────────────────────────────────
# 1. 
# ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    curl wget git vim \
    build-essential cmake \
    software-properties-common \
    lsb-release gnupg2 \
    sudo tzdata \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────
# 2. ROS2 Humble installation
# ────────────────────────────────────────────
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-ament-cmake \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────
# 3. rl_sar dependency
# ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libyaml-cpp-dev \
    libeigen3-dev \
    libboost-all-dev \
    libspdlog-dev \
    libfmt-dev \
    libtbb-dev \
    liblcm-dev \
    libopenblas-dev \
    libopenmpi-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────
# 4. rl_sar ROS2 dependency
# ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-humble-teleop-twist-keyboard \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-control-toolbox \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-joy \
    ros-humble-demo-nodes-cpp \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────
# 5. ours ROS2 dependency (Nav2 + MPPI)
# ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ros-humble-nav2-controller \
    ros-humble-nav2-mppi-controller \
    ros-humble-nav2-velocity-smoother \
    ros-humble-nav2-lifecycle-manager \
    ros-humble-tf2-tools \
    ros-humble-tf2-ros \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────
# 6. Python package (YOLO, camera etc)
# ────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    scipy \
    ultralytics \
    pyrealsense2

# ────────────────────────────────────────────
# 7. PyTorch CPU (aarch64 official support)
#    LibTorch: install_pytorch_jetson.sh - 
#    Python torch auto
# ────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

# ────────────────────────────────────────────
# 8. Jetson check
#    install_pytorch_jetson.sh -> Jetson check
# ────────────────────────────────────────────
RUN echo "# R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, EABI: aarch64, DATE: Sun Mar 19 15:19:21 UTC 2023" \
    > /etc/nv_tegra_release

# ────────────────────────────────────────────
# 9. Unitree SDK2 Python (Go2 communication)
# ────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# CycloneDDS (Unitree SDK dependency)
WORKDIR /opt
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds.git -b releases/0.10.x && \
    cd cyclonedds && \
    mkdir build install && \
    cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=../install && \
    cmake --build . --target install

ENV CYCLONEDDS_HOME=/opt/cyclonedds/install
ENV CMAKE_PREFIX_PATH=${CYCLONEDDS_HOME}:${CMAKE_PREFIX_PATH}

RUN pip3 install --no-cache-dir cyclonedds==0.10.2

# unitree_sdk2_python
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python && \
    cd /opt/unitree_sdk2_python && \
    pip3 install --no-cache-dir --no-deps -e .

ENV PYTHONPATH="/opt/unitree_sdk2_python:${PYTHONPATH}"

# ────────────────────────────────────────────
# 10. rl_sar clone + build
#     (deploy: cmake_build/bin/rl_real_go2)
# ────────────────────────────────────────────
WORKDIR /workspace

RUN git clone --recursive https://github.com/fan-ziqi/rl_sar.git

WORKDIR /workspace/rl_sar

RUN chmod +x scripts/*.sh build.sh && \
    IS_JETSON=true ./build.sh -m

# ────────────────────────────────────────────
# 11. go2-stair-climber clone + build
# ────────────────────────────────────────────
WORKDIR /workspace

RUN git clone https://github.com/ekdus9968/Go2_ROS2_StairClimbier.git go2-stair-climber && \
    cd go2-stair-climber && \
    git checkout detection-stairs/state

WORKDIR /workspace/go2-stair-climber/ros2_ws

RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# ────────────────────────────────────────────
# 12. env set
# ────────────────────────────────────────────
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/go2-stair-climber/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

WORKDIR /workspace

CMD ["bash"]