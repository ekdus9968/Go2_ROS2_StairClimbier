#!/usr/bin/env python3
"""
Fake detector using p3d odometry — simulation only.
Uses /odom (Go2) and /patient/odom to compute patient position in base_link.
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool


class FakeDetectorNode(Node):

    def __init__(self):
        super().__init__('fake_detector')
        self.declare_parameter('camera_hfov_rad', 1.5708)
        self.declare_parameter('max_detection_range_m', 8.0)
        self.declare_parameter('publish_rate_hz', 10.0)

        self.hfov = self.get_parameter('camera_hfov_rad').value
        self.max_range = self.get_parameter('max_detection_range_m').value
        rate = self.get_parameter('publish_rate_hz').value

        self.go2_pose = None
        self.patient_pose = None

        self.create_subscription(Odometry, '/odom', self.go2_cb, 10)
        self.create_subscription(Odometry, '/patient/odom', self.patient_cb, 10)

        self.pub_target = self.create_publisher(PoseStamped, 'person_follow/target', 10)
        self.pub_valid = self.create_publisher(Bool, 'person_follow/target_valid', 10)

        self.create_timer(1.0 / rate, self.publish_target)

        self.get_logger().info(
            f'FakeDetector: HFOV={math.degrees(self.hfov):.0f}deg, max_range={self.max_range}m'
        )

    def go2_cb(self, msg):
        self.go2_pose = msg.pose.pose

    def patient_cb(self, msg):
        self.patient_pose = msg.pose.pose

    def publish_target(self):
        if self.go2_pose is None or self.patient_pose is None:
            self.pub_valid.publish(Bool(data=False))
            return

        q = self.go2_pose.orientation
        go2_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )

        dx = self.patient_pose.position.x - self.go2_pose.position.x
        dy = self.patient_pose.position.y - self.go2_pose.position.y
        dist = math.sqrt(dx**2 + dy**2)

        if dist > self.max_range:
            self.pub_valid.publish(Bool(data=False))
            return

        cos_y = math.cos(-go2_yaw)
        sin_y = math.sin(-go2_yaw)
        x_base = cos_y * dx - sin_y * dy
        y_base = sin_y * dx + cos_y * dy

        if x_base <= 0.0:
            self.pub_valid.publish(Bool(data=False))
            return

        bearing = math.atan2(y_base, x_base)
        if abs(bearing) > self.hfov / 2.0:
            self.pub_valid.publish(Bool(data=False))
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x_base
        pose.pose.position.y = y_base
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.pub_target.publish(pose)

        self.pub_valid.publish(Bool(data=True))


def main(args=None):
    rclpy.init(args=args)
    node = FakeDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
