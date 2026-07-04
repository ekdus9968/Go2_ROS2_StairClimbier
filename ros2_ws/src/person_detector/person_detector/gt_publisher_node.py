#!/usr/bin/env python3
"""
Ground Truth publisher — for debugging/validation only.

Uses /odom (Go2 base_link) and /patient/odom (Patient pelvis) to compute
the true patient position in base_link frame. NOT used for control —
only for comparing against sensor-based detection.
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32


class GTPublisher(Node):

    def __init__(self):
        super().__init__('gt_publisher')

        self.declare_parameter('publish_rate_hz', 10.0)

        rate = self.get_parameter('publish_rate_hz').value

        self.go2_pose = None
        self.patient_pose = None

        self.create_subscription(Odometry, '/odom', self.go2_cb, 10)
        self.create_subscription(Odometry, '/patient/odom', self.patient_cb, 10)

        self.pub_target = self.create_publisher(PoseStamped, '/debug/gt_target', 10)
        self.pub_valid = self.create_publisher(Bool, '/debug/gt_target_valid', 10)
        self.pub_distance = self.create_publisher(Float32, '/debug/gt_distance', 10)

        self.create_timer(1.0 / rate, self.publish_gt)

        self.get_logger().info('GT publisher started (debug only)')

    def go2_cb(self, msg):
        self.go2_pose = msg.pose.pose

    def patient_cb(self, msg):
        self.patient_pose = msg.pose.pose

    def publish_gt(self):
        if self.go2_pose is None or self.patient_pose is None:
            self.pub_valid.publish(Bool(data=False))
            return

        q = self.go2_pose.orientation
        go2_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

        dx = self.patient_pose.position.x - self.go2_pose.position.x
        dy = self.patient_pose.position.y - self.go2_pose.position.y
        dist_xy = math.sqrt(dx ** 2 + dy ** 2)

        cos_y = math.cos(-go2_yaw)
        sin_y = math.sin(-go2_yaw)
        x_base = cos_y * dx - sin_y * dy
        y_base = sin_y * dx + cos_y * dy

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x_base
        pose.pose.position.y = y_base
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self.pub_target.publish(pose)

        self.pub_valid.publish(Bool(data=True))
        self.pub_distance.publish(Float32(data=dist_xy))


def main(args=None):
    rclpy.init(args=args)
    node = GTPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
