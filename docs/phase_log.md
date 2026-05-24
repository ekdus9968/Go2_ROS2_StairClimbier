# Project Phase Log

## PHASE 1 — Environment Setup ✅

### Decisions
- Host OS: Windows 11 + WSL2 Ubuntu 22.04 (no dual boot)
- GPU: RTX 5060 Ti 8GB (Blackwell sm_120)
- Training framework: PGTT (MuJoCo MJX) — pivoted from Isaac Lab due to PhysX GPU pipeline issue on Blackwell + WSL2
- Conda env: `pgtt` (Python 3.10)
- Key packages: JAX cuda12, mujoco 3.8.1, mujoco-mjx

### Issues encountered
- Isaac Lab PhysX GPU solver failed on Blackwell GPU (CPU fallback caused hang) — abandoned
- JAX cuDNN initialization error in WSL2 — solved by installing `nvidia-cudnn-cu12`
- NVIDIA driver 596.36 caused PhysX failures — downgraded to 576.88

### Verified working
- `python deploy/deploy_heightmap.py --robot go2 --stairs --level level13 --vx 0.5` runs
- MuJoCo viewer displays Go2 via WSLg
- GPU utilization confirmed (30W power draw, P3 state)

## PHASE 2 — Pretrained Policy Evaluation (in progress)

### Available pretrained policies (Go2)
- `policy_go2_pgtt_level03_run0` through `level20_run0`
- Methods: pgtt, baseline, wild

### Goals
- Evaluate each policy on stair heights 1~20cm
- Identify best baseline for our scenario (8~18cm stairs)
- Decide which level to use for PHASE 4 fine-tuning
