#!/usr/bin/env python3
"""Biped patient gait generator with waypoint trajectory."""
import sys
import termios
import tty
import select
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = [
    'base_x_joint', 'base_z_joint', 'base_pitch_joint',
    'L_hip_joint', 'L_knee_joint', 'L_ankle_joint',
    'R_hip_joint', 'R_knee_joint', 'R_ankle_joint',
]

STAND_HEIGHT = 0.0
STEP_DISTANCE = 0.30
STAIR_TREAD = 0.28
STAIR_RISER = 0.07

HIP0, KNEE0, ANK0 = 0.0, 0.12, -0.12

CMD_TOPIC = '/patient/patient_trajectory_controller/joint_trajectory'

# Waypoint trajectory: (x, z, mode)
# mode: 'walk' = 평지 걷기, 'climb' = 계단 오르기
WAYPOINTS = [
    (3.0, 0.0, 'walk'),     # 계단 앞까지
    (8.88, 1.44, 'climb'),  # 계단 21칸 올라 2층
    (10.0, 1.44, 'walk'),   # 2층 안쪽
]


class BipedGaitGenerator(Node):
    def __init__(self):
        super().__init__('biped_gait_generator')
        self.pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.bx = 1.0   # spawn 위치
        self.bz = STAND_HEIGHT
        self.lead = 'R'
        self.get_logger().info(
            "\nBiped Patient Teleop\n"
            "W: step fwd  S: step back\n"
            "X: stair up  Z: stair down\n"
            "T: run waypoint trajectory\n"
            "Space: stand  P: pose  R: reset  Ctrl+C: quit\n"
        )
    
    def stand_pose(self):
        return {
            'base_x_joint': self.bx, 'base_z_joint': self.bz, 'base_pitch_joint': 0.0,
            'L_hip_joint': HIP0, 'L_knee_joint': KNEE0, 'L_ankle_joint': ANK0,
            'R_hip_joint': HIP0, 'R_knee_joint': KNEE0, 'R_ankle_joint': ANK0,
        }
    
    def leg(self, q, side, hip, knee, ank):
        q[f'{side}_hip_joint'] = hip
        q[f'{side}_knee_joint'] = knee
        q[f'{side}_ankle_joint'] = ank
    
    def publish(self, keyframes):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        for pose, t in keyframes:
            pt = JointTrajectoryPoint()
            pt.positions = [float(pose[j]) for j in JOINT_NAMES]
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            traj.points.append(pt)
        self.pub.publish(traj)
    
    def stand(self):
        self.publish([(self.stand_pose(), 0.6)])
    
    def walk_step(self, direction=1):
        lead, trail = self.lead, ('L' if self.lead == 'R' else 'R')
        d = direction
        adv = STEP_DISTANCE * d
        k1 = self.stand_pose()
        self.leg(k1, lead, 0.55 * d, 0.85, -0.20)
        self.leg(k1, trail, -0.20 * d, KNEE0, ANK0)
        k1['base_x_joint'] = self.bx + adv * 0.5
        k2 = self.stand_pose()
        self.leg(k2, lead, 0.20 * d, KNEE0, ANK0)
        self.leg(k2, trail, -0.20 * d, KNEE0, ANK0)
        k2['base_x_joint'] = self.bx + adv
        self.bx += adv
        k3 = self.stand_pose()
        self.publish([(k1, 0.5), (k2, 1.0), (k3, 1.4)])
        self.lead = trail
        self.get_logger().info(f"walk {'fwd' if d>0 else 'back'} bx={self.bx:.2f}")
        return 1.4  # 모션 시간
    
    def climb_stair(self, up=True):
        lead, trail = self.lead, ('L' if self.lead == 'R' else 'R')
        rise = STAIR_RISER if up else -STAIR_RISER
        k1 = self.stand_pose()
        self.leg(k1, lead, 0.95, 1.7, -0.30)
        self.leg(k1, trail, -0.05, 0.25, -0.20)
        k1['base_pitch_joint'] = 0.10
        self.bz += rise
        self.bx += STAIR_TREAD * 0.5
        k2 = self.stand_pose()
        self.leg(k2, lead, 0.30, 0.35, -0.20)
        self.leg(k2, trail, -0.30, 0.35, -0.20)
        k2['base_pitch_joint'] = 0.05
        k3 = self.stand_pose()
        self.leg(k3, trail, 0.85, 1.6, -0.30)
        self.leg(k3, lead, 0.0, KNEE0, ANK0)
        k3['base_pitch_joint'] = 0.05
        self.bx += STAIR_TREAD * 0.5
        k4 = self.stand_pose()
        self.publish([(k1, 0.6), (k2, 1.3), (k3, 2.0), (k4, 2.6)])
        self.lead = trail
        self.get_logger().info(f"stair {'up' if up else 'down'} bx={self.bx:.2f} bz={self.bz:.2f}")
        return 2.6
    
    def run_trajectory(self):
        """Execute the full waypoint trajectory."""
        self.get_logger().info(f"Starting waypoint trajectory from bx={self.bx:.2f}")
        
        for target_x, target_z, mode in WAYPOINTS:
            self.get_logger().info(f"Target: x={target_x}, z={target_z}, mode={mode}")
            
            if mode == 'walk':
                # 평지 walk — STEP_DISTANCE씩
                while self.bx < target_x - 0.05:
                    dx_remaining = target_x - self.bx
                    if dx_remaining < STEP_DISTANCE:
                        # 마지막 작은 step
                        break
                    duration = self.walk_step(1)
                    time.sleep(duration + 0.1)
                    rclpy.spin_once(self, timeout_sec=0.01)
            
            elif mode == 'climb':
                # 계단 climb — STAIR_TREAD씩
                while self.bx < target_x - 0.05 and self.bz < target_z - 0.02:
                    duration = self.climb_stair(up=True)
                    time.sleep(duration + 0.1)
                    rclpy.spin_once(self, timeout_sec=0.01)
        
        self.get_logger().info(f"Trajectory complete! Final: bx={self.bx:.2f}, bz={self.bz:.2f}")
    
    def reset(self):
        self.bx = 1.0
        self.bz = STAND_HEIGHT
        self.lead = 'R'
        self.publish([(self.stand_pose(), 1.0)])
        self.get_logger().info("reset")
    
    def print_pose(self):
        self.get_logger().info(f"bx={self.bx:.2f} bz={self.bz:.2f} lead={self.lead}")
    
    def get_key(self, timeout=0.1):
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            key = sys.stdin.read(1) if rlist else ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            key = self.get_key()
            if not key:
                continue
            k = key.lower()
            if k == 'w': self.walk_step(1)
            elif k == 's': self.walk_step(-1)
            elif k == 'x': self.climb_stair(True)
            elif k == 'z': self.climb_stair(False)
            elif k == 't': self.run_trajectory()
            elif key == ' ': self.stand()
            elif k == 'p': self.print_pose()
            elif k == 'r': self.reset()
            elif key == '\x03': break


def main():
    rclpy.init()
    node = BipedGaitGenerator()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
