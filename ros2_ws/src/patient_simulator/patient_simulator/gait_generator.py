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
    (0.70, 0.00, 'walk'),    # 계단 앞 (step1 앞면이 x=1.02라 그 직전)
    (1.16, 0.07, 'climb'),   # step1
    (1.44, 0.14, 'climb'),   # step2
    (1.72, 0.21, 'climb'),   # step3
    (2.00, 0.28, 'climb'),   # step4
    (2.28, 0.35, 'climb'),   # step5
    (2.56, 0.42, 'climb'),   # step6
    (2.84, 0.49, 'climb'),   # step7
    (3.12, 0.56, 'climb'),   # step8
    (3.40, 0.63, 'climb'),   # step9
    (3.68, 0.70, 'climb'),   # step10
    (3.96, 0.77, 'climb'),   # step11
    (4.24, 0.84, 'climb'),   # step12
    (4.52, 0.91, 'climb'),   # step13
    (4.80, 0.98, 'climb'),   # step14
    (5.08, 1.05, 'climb'),   # step15
    (5.36, 1.12, 'climb'),   # step16
    (5.64, 1.19, 'climb'),   # step17
    (5.92, 1.26, 'climb'),   # step18
    (6.20, 1.33, 'climb'),   # step19
    (6.48, 1.40, 'climb'),   # step20
    (6.76, 1.44, 'climb'),   # step21 (마지막, riser 0.04)
    (7.40, 1.44, 'walk'),    # 2층 진입 (platform 앞 가장자리 6.90 통과)
    (10.0, 1.44, 'walk'),    # 2층 안쪽
]


class BipedGaitGenerator(Node):
    def __init__(self):
        super().__init__('biped_gait_generator')
        self.pub = self.create_publisher(JointTrajectory, CMD_TOPIC, 10)
        self.bx = 0.0   # spawn 위치
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
        self.publish([(k1, 1.0), (k2, 2.0), (k3, 2.8)])
        self.lead = trail
        self.get_logger().info(f"walk {'fwd' if d>0 else 'back'} bx={self.bx:.2f}")
        return 2.8  # 모션 시간 (elderly pace)
    
    def climb_stair(self, up=True):
        lead, trail = self.lead, ('L' if self.lead == 'R' else 'R')
        rise = STAIR_RISER if up else -STAIR_RISER
        
        # Phase 1: lead leg lift only (no body movement, trail planted)
        k1 = {
            'base_x_joint': self.bx,
            'base_z_joint': self.bz,
            'base_pitch_joint': 0.05,
            f'{lead}_hip_joint': 0.95, f'{lead}_knee_joint': 1.7, f'{lead}_ankle_joint': -0.30,
            f'{trail}_hip_joint': HIP0, f'{trail}_knee_joint': KNEE0, f'{trail}_ankle_joint': ANK0,
        }
        
        # Phase 2: lead foot placed on new step, body half-shifted
        k2 = {
            'base_x_joint': self.bx + STAIR_TREAD * 0.3,
            'base_z_joint': self.bz + rise * 0.5,
            'base_pitch_joint': 0.10,
            f'{lead}_hip_joint': 0.20, f'{lead}_knee_joint': 0.60, f'{lead}_ankle_joint': -0.40,
            f'{trail}_hip_joint': -0.30, f'{trail}_knee_joint': 0.05, f'{trail}_ankle_joint': -0.10,
        }
        
        # Phase 3: body shifts onto new step, trail leg lifts
        k3 = {
            'base_x_joint': self.bx + STAIR_TREAD * 0.7,
            'base_z_joint': self.bz + rise * 0.8,
            'base_pitch_joint': 0.05,
            f'{lead}_hip_joint': 0.0, f'{lead}_knee_joint': KNEE0, f'{lead}_ankle_joint': ANK0,
            f'{trail}_hip_joint': 0.85, f'{trail}_knee_joint': 1.7, f'{trail}_ankle_joint': -0.30,
        }
        
        # Phase 4: full stand on new step
        self.bx += STAIR_TREAD
        self.bz += rise
        k4 = self.stand_pose()
        
        self.publish([(k1, 1.2), (k2, 2.6), (k3, 4.0), (k4, 5.2)])
        self.lead = trail
        self.get_logger().info(f"stair {'up' if up else 'down'} bx={self.bx:.2f} bz={self.bz:.2f}")
        return 5.2
    
    def run_trajectory(self):
        """Execute the full waypoint trajectory."""
        self.get_logger().info(f"Starting waypoint trajectory from bx={self.bx:.2f}")
        
        for target_x, target_z, mode in WAYPOINTS:
            self.get_logger().info(f"Target: x={target_x}, z={target_z}, mode={mode}")
            
            if mode == 'walk':
                # 평지 walk — STEP_DISTANCE씩, 목표까지 반복
                while self.bx < target_x - 0.05:
                    duration = self.walk_step(1)
                    time.sleep(duration + 0.1)
                    rclpy.spin_once(self, timeout_sec=0.01)
            
            elif mode == 'climb':
                # 계단 1칸 (한 waypoint = 한 step)
                duration = self.climb_stair(up=True)
                time.sleep(duration + 0.1)
                rclpy.spin_once(self, timeout_sec=0.01)
                # 마지막 step의 riser는 0.04이지만 climb_stair는 항상 0.07 더함
                # target_z와 self.bz 차이를 보정
                if abs(self.bz - target_z) > 0.005:
                    self.bz = target_z  # 강제 동기화
        
        self.get_logger().info(f"Trajectory complete! Final: bx={self.bx:.2f}, bz={self.bz:.2f}")
    
    def reset(self):
        self.bx = 0.0
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
