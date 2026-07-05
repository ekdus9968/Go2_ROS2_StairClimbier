#!/usr/bin/env python3
"""
stair_mode.py

Reads outputs from the individual detectors and decides when to trigger
stair mode based on distance. Run after testing the two detector files.

Run to test:
  ros2 run stair_detector stair_mode
  ros2 topic echo /stair_detected

Subscribes:
  /stair/lidar_detected     std_msgs/Bool
  /stair/lidar_distance     std_msgs/Float32
  /stair/camera_detected    std_msgs/Bool
  /stair/camera_distance    std_msgs/Float32

Publishes:
  /stair_detected           std_msgs/Bool
  /stair_distance           std_msgs/Float32
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class StairMode(Node):

    def __init__(self):
        super().__init__('stair_mode')

        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('trigger_distance_m', 2.0)
        self.declare_parameter('require_both_sensors', False)

        rate = self.get_parameter('publish_rate_hz').value
        self.trigger_dist = self.get_parameter('trigger_distance_m').value
        self.require_both = self.get_parameter('require_both_sensors').value

        self.lidar_detected = False
        self.lidar_distance = float('nan')
        self.camera_detected = False
        self.camera_distance = float('nan')

        self.create_subscription(Bool, '/stair/lidar_detected', self.lidar_det_cb, 10)
        self.create_subscription(Float32, '/stair/lidar_distance', self.lidar_dist_cb, 10)
        self.create_subscription(Bool, '/stair/camera_detected', self.camera_det_cb, 10)
        self.create_subscription(Float32, '/stair/camera_distance', self.camera_dist_cb, 10)

        self.pub_detected = self.create_publisher(Bool, '/stair_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair_distance', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairMode ready. trigger_dist={self.trigger_dist}m, require_both={self.require_both}'
        )

    def lidar_det_cb(self, msg: Bool):
        self.lidar_detected = msg.data

    def lidar_dist_cb(self, msg: Float32):
        self.lidar_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def camera_det_cb(self, msg: Bool):
        self.camera_detected = msg.data

    def camera_dist_cb(self, msg: Float32):
        self.camera_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def publish_result(self):
        if self.require_both:
            any_detected = self.lidar_detected and self.camera_detected
        else:
            any_detected = self.lidar_detected or self.camera_detected

        distances = [
            d for d in [self.lidar_distance, self.camera_distance]
            if not math.isnan(d)
        ]
        distance = min(distances) if distances else float('nan')

        close_enough = (not math.isnan(distance)) and (distance < self.trigger_dist)
        detected = any_detected and close_enough

        self.pub_detected.publish(Bool(data=detected))
        self.pub_distance.publish(Float32(
            data=distance if not math.isnan(distance) else -1.0
        ))

        if detected:
            self.get_logger().info(
                f'Stair mode triggered at {distance:.2f}m '
                f'(lidar={self.lidar_detected}, camera={self.camera_detected})',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = StairMode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
