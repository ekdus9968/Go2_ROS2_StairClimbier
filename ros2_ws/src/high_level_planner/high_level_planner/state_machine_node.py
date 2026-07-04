#!/usr/bin/env python3
"""
State machine for Go2 follower — Simplified.

States:
  IDLE   (0): no person detected
  FOLLOW (1): person detected, MPPI follow

STAIR_CLIMB deferred — himloco handles stair terrain via cmd_vel.
Stair detector runs as monitor only (future).
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from go2_interfaces.msg import RobotMode

PERSON_LOST_TIMEOUT_SEC = 2.0


class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine')
        self.mode = RobotMode.IDLE
        self.person_visible = False
        self.last_person_seen_time = None

        self.create_subscription(
            Bool, 'person_follow/target_valid', self.person_valid_cb, 10
        )
        self.pub_mode = self.create_publisher(RobotMode, '/robot_mode', 10)
        self.create_timer(0.1, self.update)

        self.get_logger().info('StateMachine started in IDLE mode')

    def person_valid_cb(self, msg):
        self.person_visible = msg.data
        if msg.data:
            self.last_person_seen_time = self.get_clock().now()

    def update(self):
        if self.mode == RobotMode.IDLE:
            if self.person_visible:
                self._set_mode(RobotMode.FOLLOW)
        elif self.mode == RobotMode.FOLLOW:
            if not self.person_visible and self.last_person_seen_time is not None:
                elapsed = (self.get_clock().now() - self.last_person_seen_time).nanoseconds / 1e9
                if elapsed > PERSON_LOST_TIMEOUT_SEC:
                    self._set_mode(RobotMode.IDLE)

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
