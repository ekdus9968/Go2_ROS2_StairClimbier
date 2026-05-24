
# System Architecture

## Layered Design
PHASE 1-2: PGTT (MuJoCo MJX)
└── Locomotion policy training (stock Go2)
PHASE 3: Isaac Sim URDF
└── go2_isaac.urdf with D435i + Hesai + tank
PHASE 4: Sensor sim + ROS2
├── Isaac Sim (1 env, PhysX CPU acceptable)
└── ROS2 nodes (separate container)
PHASE 5: Real hardware
└── Unitree SDK2 + PGTT deploy/
## Core Principle: Strict Separation

Locomotion policy receives ONLY:
- `cmd_vel` (vx, vy, yaw_rate)
- proprioception (joint state)
- heightmap (terrain perception)

It does NOT know about: patient, tether, mode, sensors.

All scenario logic lives in ROS2 high-level layer.

## Mode State Machine

| Mode | Trigger | Robot offset (patient body frame) |
|------|---------|-----------------------------------|
| FLAT | Default | (-0.4, 0.0) |
| APPROACH | Stair detected ahead, patient still on flat | (-0.4, +0.15) |
| STAIR | Patient z rising + stair detected | (-0.5, +0.30) |

## Tether Constraint (STAIR mode)

- Slant distance ≤ 1.0 m (hard limit)
- Target: 0.6 ~ 0.8 m
- Vertical gap ≤ 2 steps
- Horizontal gap ≤ 0.8 m
- Patient lag: 2 steps when climbing

## Reward Design

See `reward_design.md`.
