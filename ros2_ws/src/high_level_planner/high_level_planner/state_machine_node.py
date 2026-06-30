#!/usr/bin/env python3
"""
State machine for Go2 follower.

States:
  IDLE         (0): no person
  FOLLOW       (1): person detected, MPPI follow
  STAIR_CLIMB  (2): on stairs, himloco handles climbing

Transitions:
  IDLE -> FOLLOW       : person_visible
  FOLLOW -> STAIR_CLIMB : stair_detected AND patient on stairs (pelvis_z > threshold)
  STAIR_CLIMB -> FOLLOW : NOT stair_detected (reached flat ground)
  FOLLOW -> IDLE       : person lost for PERSON_LOST_TIMEOUT_SEC
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from go2_interfaces.msg import RobotMode

PERSON_LOST_TIMEOUT_SEC = 2.0
PATIENT_ON_STAIRS_Z_THRESHOLD = 0.05  # pelvis z > this → patient is climbing


class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine')

        self.mode = RobotMode.IDLE
        self.person_visible = False
        self.stair_detected = False
        self.patient_pelvis_z = 0.0
        self.last_person_seen_time = None

        self.create_subscription(
            Bool, 'person_follow/target_valid', self.person_valid_cb, 10
        )
        self.create_subscription(
            Bool, '/stair_detected', self.stair_cb, 10
        )
        self.create_subscription(
            Odometry, '/patient/odom', self.patient_cb, 10
        )

        self.pub_mode = self.create_publisher(RobotMode, '/robot_mode', 10)

        self.create_timer(0.1, self.update)

        self.get_logger().info('StateMachine started in IDLE mode')

    def person_valid_cb(self, msg: Bool):
        self.person_visible = msg.data
        if msg.data:
            self.last_person_seen_time = self.get_clock().now()

    def stair_cb(self, msg: Bool):
        self.stair_detected = msg.data

    def patient_cb(self, msg: Odometry):
        self.patient_pelvis_z = msg.pose.pose.position.z

    def update(self):
        patient_on_stairs = self.patient_pelvis_z > PATIENT_ON_STAIRS_Z_THRESHOLD

        if self.mode == RobotMode.IDLE:
            if self.person_visible:
                self._set_mode(RobotMode.FOLLOW)

        elif self.mode == RobotMode.FOLLOW:
            # Lost person → IDLE
            if not self.person_visible and self.last_person_seen_time is not None:
                elapsed = (self.get_clock().now() - self.last_person_seen_time).nanoseconds / 1e9
                if elapsed > PERSON_LOST_TIMEOUT_SEC:
                    self._set_mode(RobotMode.IDLE)
                    return
            # Patient on stairs + stair ahead → STAIR_CLIMB
            if self.stair_detected and patient_on_stairs:
                self._set_mode(RobotMode.STAIR_CLIMB)

        elif self.mode == RobotMode.STAIR_CLIMB:
            # No more stairs → back to FOLLOW
            if not self.stair_detected:
                self._set_mode(RobotMode.FOLLOW)

        msg = RobotMode()
        msg.mode = self.mode
        self.pub_mode.publish(msg)

    def _set_mode(self, new_mode):
        names = {0: 'IDLE', 1: 'FOLLOW', 2: 'STAIR_CLIMB'}
        if new_mode != self.mode:
            self.get_logger().info(f'Mode: {names[self.mode]} -> {names[new_mode]}')
        self.mode = new_mode


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
