#!/usr/bin/env python3
"""
Simple cmd_vel-based follow controller.
Subscribes to person_follow/target (base_link frame) and publishes cmd_vel
to maintain follow_distance behind the person.

Only active when /robot_mode == FOLLOW.
"""
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from go2_interfaces.msg import RobotMode


class SimpleFollowNode(Node):

    def __init__(self):
        super().__init__('simple_follow')

        self.declare_parameter('follow_distance_m', 1.2)
        self.declare_parameter('follow_tolerance_m', 0.15)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_linear', 0.6)
        self.declare_parameter('kp_angular', 1.5)
        self.declare_parameter('target_timeout_sec', 1.0)
        self.declare_parameter('control_rate_hz', 10.0)

        self.follow_dist = self.get_parameter('follow_distance_m').value
        self.follow_tol = self.get_parameter('follow_tolerance_m').value
        self.max_lin = self.get_parameter('max_linear_speed').value
        self.max_ang = self.get_parameter('max_angular_speed').value
        self.kp_lin = self.get_parameter('kp_linear').value
        self.kp_ang = self.get_parameter('kp_angular').value
        self.target_timeout = self.get_parameter('target_timeout_sec').value
        rate = self.get_parameter('control_rate_hz').value

        self.target_x = None
        self.target_y = None
        self.target_valid = False
        self.last_target_time = 0.0
        self.mode = RobotMode.IDLE

        self.create_subscription(PoseStamped, 'person_follow/target', self.target_cb, 10)
        self.create_subscription(Bool, 'person_follow/target_valid', self.valid_cb, 10)
        self.create_subscription(RobotMode, '/robot_mode', self.mode_cb, 10)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'SimpleFollow: follow_dist={self.follow_dist}m, '
            f'max_lin={self.max_lin}m/s, max_ang={self.max_ang}rad/s'
        )

    def target_cb(self, msg: PoseStamped):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.last_target_time = time.time()

    def valid_cb(self, msg: Bool):
        self.target_valid = msg.data

    def mode_cb(self, msg: RobotMode):
        self.mode = msg.mode

    def control_loop(self):
        if self.mode != RobotMode.FOLLOW:
            return

        if not self.target_valid or self.target_x is None:
            self._stop()
            return

        if time.time() - self.last_target_time > self.target_timeout:
            self._stop()
            return

        dist = math.sqrt(self.target_x ** 2 + self.target_y ** 2)
        bearing = math.atan2(self.target_y, self.target_x)

        # Linear: drive forward until follow_distance reached
        error_dist = dist - self.follow_dist
        if abs(error_dist) < self.follow_tol:
            vx = 0.0
        else:
            vx = self.kp_lin * error_dist
            vx = max(-self.max_lin, min(self.max_lin, vx))

        # Angular: rotate to face person
        wz = self.kp_ang * bearing
        wz = max(-self.max_ang, min(self.max_ang, wz))

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.pub_cmd.publish(cmd)

        self.get_logger().info(
            f'follow: dist={dist:.2f}m bearing={math.degrees(bearing):.0f}deg '
            f'-> vx={vx:.2f} wz={wz:.2f}',
            throttle_duration_sec=0.5
        )

    def _stop(self):
        cmd = Twist()
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleFollowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
