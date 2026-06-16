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

## PHASE 2 — Pretrained Policy Evaluation ✅

### Method
- Visual evaluation via `deploy/deploy_heightmap.py`
- Run on host conda env (Docker container has JAX/PyTorch conflict issue)

### Result
- `policy_go2_pgtt_level13_run0` confirmed working
- Go2 climbs stairs with `--vx 0.5`
- Detailed quantitative evaluation deferred (not blocking PHASE 3)

### Decision
- **Baseline for PHASE 4**: `policy_go2_pgtt_level13_run0`
- Method: `pgtt`
- Rationale: Highest difficulty among CLI-supported levels

### Notes
- `evaluate_multiple.py` requires `checks_stairs/` checkpoint format (different from `policies/`)
- Quantitative eval will be done after PHASE 4 fine-tune
