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
>>>>>>> e12468e (Initial: project structure, docs, gitignore)
