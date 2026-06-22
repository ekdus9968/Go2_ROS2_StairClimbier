#!/usr/bin/env python3
"""
state_machine_node.py

Publishes /robot_mode based on current conditions.
This is a SKELETON your teammate fills in the transition logic.

The node exposes the mode enum and the topic contract so both
follow_controller_node and go2_nav_bridge can be coded against it now,
before the full SM logic is written.

Modes:
  IDLE        (0) robot stopped, no active behavior
  FOLLOW      (1) human-following active
  STAIR_CLIMB (2) stair climbing active (rl_sar policy)

Transitions (to be implemented by your teammate):
  IDLE -> FOLLOW      : person detected + no stairs ahead
  FOLLOW -> STAIR_CLIMB : stairs detected ahead of robot
  STAIR_CLIMB -> FOLLOW : stairs cleared
  any -> IDLE         : e-stop, timeout, or explicit command

Topics subscribed:
  person_follow/target_valid  std_msgs/Bool     (person detector)
  /stair_detected             std_msgs/Bool     (stair detector TBD)
  /estop                      std_msgs/Bool     (emergency stop)

Topics published:
  /robot_mode                 go2_interfaces/RobotMode
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from go2_interfaces.msg import RobotMode

# State transition timeouts
PERSON_LOST_TIMEOUT_SEC = 2.0   # go IDLE if person not seen for this long

class StateMachineNode(Node):

    def __init__(self):
        super().__init__('state_machine')

        # State
        self.mode = RobotMode.IDLE
        self.person_visible = False
        self.stair_detected = False
        self.estop = False
        self.last_person_seen_time = None

        # Subscribers
        self.create_subscription(
            Bool,
            'person_follow/target_valid',
            self.person_valid_cb,
            10
        )
        self.create_subscription(
            Bool,
            '/stair_detected',
            self.stair_cb,
            10
        )
        self.create_subscription(
            Bool,
            '/estop',
            self.estop_cb,
            10
        )

        # Publisher
        self.pub_mode = self.create_publisher(RobotMode, '/robot_mode', 10)

        # Timer: run state machine at 10 Hz
        self.create_timer(0.1, self.update)

        self.get_logger().info('StateMachineNode started in IDLE mode.')

    #
    # Subscribers
    #
    def person_valid_cb(self, msg: Bool):
        self.person_visible = msg.data
        if msg.data:
            self.last_person_seen_time = self.get_clock().now()

    def stair_cb(self, msg: Bool):
        self.stair_detected = msg.data

    def estop_cb(self, msg: Bool):
        self.estop = msg.data
        if msg.data:
            self.get_logger().warn('E-STOP received forcing IDLE')
            self._set_mode(RobotMode.IDLE)

    #
    # State machine update
    #
    def update(self):
        """
        Called at 10 Hz. Evaluate transition conditions and update mode.

        TODO (your teammate): fill in the real transition logic here.
        The stubs below show the intended structure.
        """

        if self.estop:
            self._set_mode(RobotMode.IDLE)
            return

        if self.mode == RobotMode.IDLE:
            self._from_idle()

        elif self.mode == RobotMode.FOLLOW:
            self._from_follow()

        elif self.mode == RobotMode.STAIR_CLIMB:
            self._from_stair_climb()

        # Publish current mode every cycle so subscribers always have fresh state
        msg = RobotMode()
        msg.mode = self.mode
        self.pub_mode.publish(msg)

    def _from_idle(self):
        """Transitions out of IDLE."""
        if self.person_visible:
            self.get_logger().info('Person detected switching to FOLLOW')
            self._set_mode(RobotMode.FOLLOW)

    def _from_follow(self):
        """Transitions out of FOLLOW."""

        # Person lost for too long go idle
        if not self.person_visible and self.last_person_seen_time is not None:
            elapsed = (self.get_clock().now() - self.last_person_seen_time).nanoseconds / 1e9
            if elapsed > PERSON_LOST_TIMEOUT_SEC:
                self.get_logger().info('Person lost switching to IDLE')
                self._set_mode(RobotMode.IDLE)
                return

        # Stairs ahead hand off to rl_sar
        # TODO: your teammate adds the actual stair detection condition here
        if self.stair_detected:
            self.get_logger().info('Stairs detected switching to STAIR_CLIMB')
            self._set_mode(RobotMode.STAIR_CLIMB)

    def _from_stair_climb(self):
        """Transitions out of STAIR_CLIMB."""
        # TODO: your teammate defines "stairs cleared" condition
        if not self.stair_detected:
            self.get_logger().info('Stairs cleared switching to FOLLOW')
            self._set_mode(RobotMode.FOLLOW)

    def _set_mode(self, new_mode: int):
        if new_mode != self.mode:
            names = {
                RobotMode.IDLE: 'IDLE',
                RobotMode.FOLLOW: 'FOLLOW',
                RobotMode.STAIR_CLIMB: 'STAIR_CLIMB'
            }
            self.get_logger().info(
                f'Mode: {names.get(self.mode)} -> {names.get(new_mode)}'
            )
        self.mode = new_mode

def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
