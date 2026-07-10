# Go2 Stair Climber

Reinforcement learning based stair-climbing locomotion for Unitree Go2 Education robot,
following an oxygen-tethered patient up stairs.

## Hardware

- Unitree Go2 Education
- Intel RealSense D435i (depth + RGB + IMU)
- Hesai XT16 LiDAR (16-channel)
- 3D-printed mock oxygen tank (lightweight)

## Architecture
[PGTT (MuJoCo MJX)]       — Locomotion policy training
[Isaac Sim]                — Sensor sim + ROS2 bridge validation
[ROS2 Humble]              — Patient tracking, mode SM, tether constraint
[Unitree SDK2]             — Real deployment
[PGTT (MuJoCo MJX)]       — Locomotion policy training
[Isaac Sim]                — Sensor sim + ROS2 bridge validation
[ROS2 Humble]              — Patient tracking, mode SM, tether constraint
[Unitree SDK2]             — Real deployment
See `docs/architecture.md` for details.

## Requirements

- Ubuntu 22.04 (or WSL2 on Windows 11)
- NVIDIA GPU (Blackwell sm_120 tested: RTX 5060 Ti)
- NVIDIA Driver 576+ (Windows side for WSL2)
- Docker + NVIDIA Container Toolkit
- 32GB RAM, 8GB+ VRAM

## Quick Start

```bash
# Clone
git clone https://github.com/<your-id>/go2-stair-climber.git
cd go2-stair-climber

## Docker Image

Prebuilt image available on Docker Hub:
\`\`\`bash
docker pull ekdus9968/go2-stair-climber:v0.2
\`\`\`

### Run with GPU and X11 (Linux/WSL2)
\`\`\`bash
docker run -it --rm --gpus all --network=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/workspace/go2-stair-climber \
    ekdus9968/go2-stair-climber:v0.2 bash
\`\`\`

### Inside container
\`\`\`bash
# Verify GPU
python -c "import jax; print(jax.devices())"

# Run PGTT demo
cd /workspace/pgtt
python deploy/deploy_heightmap.py --robot go2 --stairs --level level13 --vx 0.5
\`\`\`

### Known limitations
- PyTorch not included (needed for `deploy_heightmap.py` only)
- To enable deploy inside container: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- This will be added in future image versions

# Run training environment
docker compose run --rm pgtt-train
```

## Project Status

- [x] PHASE 1: Environment setup (PGTT, MuJoCo MJX, JAX GPU on WSL2)
- [ ] PHASE 2: Pretrained policy evaluation
- [ ] PHASE 3: Isaac Sim URDF with sensor attachments
- [ ] PHASE 4: ROS2 integration
- [ ] PHASE 5: Real hardware deployment

## References

- [PGTT](https://github.com/NtagkasAlex/phase_guided_terrain_traversal) — Locomotion baseline
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) — Sim validation

## License

TBD

## Stair Detection Pipeline (ROS 2)

Five-node sensor fusion pipeline for stair detection, located in `ros2_ws/src/stair_detector/`.

**Architecture:**
- `lidar_stair_detector.py` — detects step-height jumps in LiDAR point cloud
- `camera_stair_detector.py` — detects depth edge jumps from RealSense depth image
- `yolo_run.py` — custom-trained YOLO model (`best.pt`), detects stairs visually
- `fusion_stair.py` — combines LiDAR + camera into a single stair signal with confidence score
- `stair_consensus.py` — combines fusion output + YOLO into final `/stair/robot_mode` (STAIR / APPROACH / NONE)

**Status (as of July 2026):**
- LiDAR, YOLO, fusion, and consensus nodes verified working end-to-end against a real recorded rosbag
- Camera depth edge detection not yet triggering on real data — depth image conversion bug fixed (RealSense publishes 16UC1/mm, not 32FC1/m), but edge-detection thresholds still need tuning against real sensor values
- Human detection branch not yet implemented (planned: separate YOLO model + consensus logic for normal/approach/stair mode switching)

**Running the pipeline:**
```bash
cd ros2_ws
colcon build --packages-select stair_detector
source install/setup.bash

ros2 run stair_detector lidar_stair_detector &
ros2 run stair_detector camera_stair_detector &
ros2 run stair_detector yolo_stair_detector --ros-args -p model_path:=/path/to/best.pt &
ros2 run stair_detector fusion_stair &
ros2 run stair_detector stair_consensus &
```

**Testing with a recorded bag (no simulation required):**
```bash
ros2 bag play /path/to/bag
ros2 topic echo /stair/robot_mode
```
