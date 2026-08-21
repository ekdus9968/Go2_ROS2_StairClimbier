# TASH — Stair-Aware Human-Following Quadruped

Autonomous perception, tracking, and planning system enabling a Unitree Go2 quadruped robot to follow a patient across environments including stairs. Built on ROS 2 Humble with a modular multi-sensor pipeline (LiDAR, camera, IMU), particle filter tracking, and a custom MPPI planner.

---

## Motivation

Existing research on assistive robotics treats **human-following** and **stair traversal** as separate problems. Wheeled assistive platforms cannot operate on stairs, and quadruped locomotion research typically focuses on locomotion alone without integrated human tracking. This project addresses the gap by combining both capabilities in a single real-time system, targeting elderly mobility support scenarios where the robot accompanies patients across mixed environments while carrying auxiliary loads (oxygen tanks, medical supplies).

---

## System Architecture

```
Sensors (LiDAR + Camera + IMU)
        ↓
Person Detection Pipeline          Stair Detection Pipeline
  ├─ LiDAR clustering                 ├─ Roll correction
  │   + sub-region search             ├─ Person exclusion
  ├─ Camera keypoint verification     ├─ KDE peak detection
  │   + partial-visibility mode       └─ Longest valid sequence
  ├─ Sensor fusion                                ↓
  └─ Particle filter tracking     ─────────→  State Machine
        ↓                                         ↓
   Predictions ─────────────────────→  MPPI Path Planner
                                              ↓
                                        /cmd_vel
```

---

## Key Contributions

- **Sub-region search clustering** — recovers person position from over-merged DBSCAN clusters where person and stair points fuse into a single cluster
- **Partial-visibility mode classifier** — explicit `full_body` / `lower_body_only` / `torso_only` detection modes handle the occlusion patterns typical of vertical navigation
- **Predictive tracking-planning integration** — particle filter predictions of future person positions feed directly into the MPPI cost function for anticipatory following
- **IMU-based roll correction** — stabilizes LiDAR data during robot locomotion while preserving pitch for natural stair-climbing posture

---

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Robot | Unitree Go2 | Quadruped platform |
| LiDAR | Hesai QT128 (128-channel, ±52.6° vertical FOV, 10 Hz) | Primary 3D perception |
| Camera | Intel RealSense D435i (640×480 RGB) | Person verification |
| IMU | Onboard | Pose estimation, roll correction |
| Onboard compute | NVIDIA Jetson (real robot) | Real-time inference |
| Development | Desktop + NVIDIA GPU | Development and MPPI training |

---

## Software Stack

- **OS**: Ubuntu 22.04 (WSL2 for development)
- **Middleware**: ROS 2 Humble
- **Simulator**: Gazebo Classic 11
- **Container**: Docker (`rl_sar` image)
- **Locomotion**: `rl_sar` pre-trained RL policy (used as-is)
- **Perception**: Python 3.10, NumPy, SciPy, scikit-learn
- **Vision**: Ultralytics YOLOv11-pose, OpenCV
- **Planning**: PyTorch (GPU-parallel MPPI via `pytorch_mppi`)
- **Coordinates**: `tf2_ros`

---

## Package Structure

```
ros2_ws/src/
├── stair_simulation/              # Gazebo world (stairs + human actor)
├── go2_with_payload/              # Extended URDF (Go2 + LiDAR + camera + payload)
├── high_level_planner/            # odom_to_tf infrastructure
├── person_detector_lidar/         # LiDAR clustering + sub-region search
├── person_detector_camera/        # YOLOv11-pose + partial-visibility mode
├── person_detector_fusion/        # LiDAR-camera matching
├── person_tracker/                # Particle filter (300 particles)
├── stair_detector_lidar/          # KDE-based stair detection
├── stair_mppi_controller/         # MPPI planner (in progress)
└── state_machine/                 # State coordination (in progress)
```

---

## Perception Pipeline Details

### LiDAR Person Detection

1. **Roll correction**: Extract `base_link` roll from TF, apply inverse rotation to point cloud (pitch and yaw preserved)
2. **Filtering**: Angle (±35°), distance (0.6–6 m), ground removal (z > -0.30 m)
3. **DBSCAN clustering**: `eps=0.20`, `min_samples=10`
4. **Direct shape check**: Height 0.5–2.0 m, width/depth 0.15–0.7 m
5. **Sub-region search** (for over-merged clusters):
   - Identify clusters where dimensions exceed 1.5× human size
   - Build 20 cm XY density grid
   - Select top-5 densest cells
   - Apply person-sized 3D box at each candidate location
   - Re-verify shape within box

**Output**: `/person_candidates` (PoseArray, `hesai_link` frame, ~3 Hz)

### Camera Verification

1. Project LiDAR person position into camera frame via TF
2. Compute image coordinates using camera intrinsics
3. Distance-adaptive cropping: 800×800 px (<1.5 m), 640×640 (1.5–3 m), 500×500 (>3 m)
4. Run YOLOv11-pose on crop
5. Select detection nearest to LiDAR-predicted image location (±200 px threshold)
6. **Mode classification**:
   - Group A (lower body): knees (kp 13, 14), ankles (kp 15, 16)
   - Group B (torso): hips (kp 11, 12)
   - `full_body`: A ≥ 3 AND B ≥ 2
   - `lower_body_only`: A ≥ 3
   - `torso_only`: B ≥ 2
   - Otherwise: `not_verified`

**Output**: `/person_verified` (PoseArray + mode, ~3 Hz)

### Fusion

- Match LiDAR and camera detections within 30 cm XY threshold
- Camera timeout: 1 second
- Output: `/person_detected` (PoseArray, 10 Hz)

### Particle Filter Tracking

- **State**: 6D `[x, y, z, vx, vy, vz]`
- **Particles**: 300 (NumPy, CPU)
- **Motion model**: Constant velocity + Gaussian noise (pos σ = 5 cm XY / 2 cm Z; vel σ = 10 cm/s XY / 5 cm/s Z)
- **Observation model**: Multi-modal Gaussian with minimum weight floor (0.01) for outlier robustness
- **Resampling**: Systematic, triggered when ESS < 50% of particle count
- **Prediction**: Particles propagated forward K steps to provide future-state estimates for the planner

**Output**: `/person_tracked` (PoseWithCovarianceStamped, 10 Hz), `/particle_cloud` (debug)

---

## Stair Detection Details

Independent LiDAR-based pipeline running in parallel to person detection.

1. **Roll correction** (pitch preserved to reflect natural climbing posture)
2. **ROI filtering**: Forward 5 m, ±2.5 m lateral, -0.5 to 2.0 m height
3. **Person exclusion**: Remove points within 0.5 m radius of `/person_tracked`
4. **KDE on Z-values**: `scipy.stats.gaussian_kde`, bandwidth 0.02, evaluated on 1000-point grid
5. **Peak detection**: `scipy.signal.find_peaks`, threshold 10% of max density, minimum peak distance 5 cm
6. **Step refinement**: For each peak, extract points within ±3 cm and compute actual z (mean), XY center, size
7. **Longest valid sequence**:
   - Sort peaks by z
   - From each starting peak, build sequences satisfying:
     - z-gap ∈ [0.05, 0.30] m
     - XY-progression ∈ [0.05, 0.60] m
   - Select longest sequence
8. **Regularity check**: Step height std/mean < 0.5
9. **Direction**: yaw = atan2(dy, dx) from first to last step

**Output**: `/stair_detected_lidar` (Bool, ~3 Hz)

Camera-based stair detection (teammate contribution) runs in parallel using a YOLOv11 model fine-tuned on the RGB-D stair dataset from PMC12693958 combined with our custom Go2-specific dataset. The state machine transitions to STAIR mode when both LiDAR and camera detectors return True.

---

## Path Planning (MPPI)

**Framework**: PyTorch-based `pytorch_mppi` running GPU-parallel trajectory evaluation.

- **State**: `[x, y, z, yaw]` in world frame
- **Control**: `[linear_vel_x, angular_vel_z]`
- **Motion model**:
  - Flat: standard differential-drive dynamics
  - Stair mode: forward velocity produces vertical displacement proportional to `tan(stair_angle)`
- **Horizon**: 20 steps (2 seconds ahead)
- **Samples**: 1000 candidate trajectories per timestep
- **Cost function**:
  - Follow distance 2.0–2.5 m band (weight 100)
  - Camera FOV coverage ±35° (weight 200)
  - Control smoothness (weight 5)
  - Stair heading alignment when in STAIR mode (weight 50)
- **Prediction integration**: Particle filter propagates 300 particles forward at each planning step; MPPI evaluates candidates against predicted person positions rather than current observation alone

**Output**: `/cmd_vel` (Twist)

> **Note**: Basic MPPI functionality (trajectory sampling, GPU cost evaluation, control publishing) has been validated. Full multi-objective integration with particle-filter predictions is in active development.

---

## State Machine

```
IDLE → PERSON_DETECTED → APPROACH → STAIR → FLAT
```

| State | Condition |
|---|---|
| IDLE | Waiting, no detection |
| PERSON_DETECTED | `/person_tracked` publishing |
| APPROACH | Following on flat ground |
| STAIR | Both `/stair_detected_lidar` and `/stair_detected_camera` return True |
| FLAT | IMU pitch returns below threshold after stair traversal |

---

## Setup

### Prerequisites

- Ubuntu 22.04 (or WSL2)
- Docker
- NVIDIA GPU + drivers (for MPPI and YOLO)
- ROS 2 Humble

### Installation

```bash
# Clone repository
git clone <repo-url> go2-stair-climber
cd go2-stair-climber

# Build Docker container
docker start rl_sar
docker exec -it rl_sar bash

# Inside container
cd /workspace/go2-stair-climber/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Python Dependencies

```bash
pip install ultralytics pytorch-mppi scipy scikit-learn --break-system-packages
```

---

## Usage

### Full Pipeline (Simulation)

```bash
# Terminal 1: Gazebo simulation + rl_sar locomotion
ros2 launch stair_simulation stair_world_with_payload.launch.py x:=-5.0

# Terminal 2: LiDAR person detection
ros2 launch person_detector_lidar lidar_clustering.launch.py

# Terminal 3: Camera person verification
ros2 launch person_detector_camera camera_keypoint.launch.py

# Terminal 4: Sensor fusion
ros2 launch person_detector_fusion fusion.launch.py

# Terminal 5: Particle filter tracking
ros2 launch person_tracker tracker.launch.py

# Terminal 6: LiDAR stair detection
ros2 launch stair_detector_lidar stair_detector.launch.py

# Terminal 7: RViz
rviz2  # Fixed Frame: world
```

### Rosbag Playback (for development)

```bash
ros2 bag play rosbags/stair_detection_tilt --loop
```

### RViz Displays

- Fixed Frame: `world`
- Displays:
  - `/hesai/hesai_lidar_controller/out` (raw LiDAR PointCloud2)
  - `/lidar_clusters` (colored clusters, PointCloud2 with intensity)
  - `/person_candidates` (LiDAR PoseArray)
  - `/person_verified` (Camera PoseArray)
  - `/person_detected` (Fusion PoseArray)
  - `/person_tracked` (PoseWithCovariance)
  - `/particle_cloud` (Particles PointCloud2)
  - `/person_debug_image` (Camera with keypoint overlay)

---

## Topics

### Sensor Inputs
- `/hesai/hesai_lidar_controller/out` — LiDAR (PointCloud2, 10 Hz)
- `/d435i/d435i/image_raw` — Camera RGB (Image)
- `/d435i/d435i/camera_info` — Camera intrinsics (CameraInfo)
- `/odom` — Robot odometry (Odometry)
- `/imu` — IMU (Imu)

### Detection Pipeline
- `/person_candidates` — LiDAR candidates (PoseArray, ~3 Hz)
- `/person_verified` — Camera verified (PoseArray, ~3 Hz)
- `/person_detected` — Fusion output (PoseArray, 10 Hz)
- `/person_tracked` — Particle filter estimate (PoseWithCovarianceStamped, 10 Hz)

### Stair Detection
- `/stair_detected_lidar` — LiDAR stair detection (Bool, ~3 Hz)
- `/stair_detected_camera` — Camera stair detection (Bool, teammate contribution)

### Control
- `/cmd_vel` — MPPI planner output (Twist)

### Debug
- `/lidar_clusters` — Colored cluster visualization (PointCloud2)
- `/particle_cloud` — Particle positions (PointCloud2)
- `/person_debug_image` — Camera image with keypoint overlay (Image)

---

## Limitations

- **Simulation-only validation**: Real-robot deployment was originally planned but was not completed due to team changes near project end. Sim-to-real gap remains untested.
- **Stationary patient**: Gazebo actor collision plugin limitations prevented programmatic patient movement. Tracking with an independently moving person is not directly validated.
- **Long-range detection**: Person detection reliability degrades beyond ~3 m due to LiDAR sparsity, even with sub-region search.
- **MPPI in progress**: Full multi-objective cost function with particle-filter predictions is under active development.
- **Frame consistency with moving robot**: The current particle filter operates in LiDAR frame; explicit ego-motion compensation is required when both robot and person move simultaneously. Real-robot deployment will require SLAM-based localization (e.g., LIO-SAM) for consistent world-frame reasoning.
- **State machine coverage**: Primary state flow is implemented; LOST recovery, multi-person handling, and reverse transitions are not yet covered.

---

## Future Work

- **Direction 1**: Systematic evaluation of sub-region search against baselines (adaptive DBSCAN, HDBSCAN, plane-fitting-based separation) as a standalone perception contribution
- **Direction 2**: Custom stair detection architecture (depth-aware feature fusion, geometric consistency heads) beyond fine-tuning
- **Direction 3**: Battery-aware reinforcement learning control for legged robots — encoding the shrinking effective actuator envelope (from SoC and current-dependent voltage drop) directly into the RL policy's observation and reward. Not yet addressed in the legged robotics literature despite parallel work in HEV energy management and mobile robot recharge planning.

---

## References

- Fan, Z. et al. `rl_sar`: Simulation and real-world deployment framework for RL-based legged locomotion. [GitHub](https://github.com/fan-ziqi/rl_sar)
- Ester, M. et al. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *KDD*.
- Jocher, G. et al. (2024). Ultralytics YOLOv11. [GitHub](https://github.com/ultralytics/ultralytics)
- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Silverman, B. W. (1986). *Density Estimation for Statistics and Data Analysis*. Chapman and Hall.
- Williams, G. et al. (2017). Model Predictive Path Integral Control: From Theory to Parallel Computation. *Journal of Guidance, Control, and Dynamics*, 40(2).
- Islam, F. et al. (2024). RGB-D Dataset for Stair Detection and Classification. PMC12693958.

---

## License

TBD

---

## Contact

[Your name/contact]
