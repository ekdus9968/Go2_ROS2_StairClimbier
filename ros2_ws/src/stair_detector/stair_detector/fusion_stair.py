#!/usr/bin/env python3
"""
fusion_stair.py

Fuses outputs from the individual detectors. Run this last after all
three other files work without errors.

All four nodes need to be running at the same time. This one just
subscribes to their outputs and makes the final call.

Both sensors must agree for /stair_detected to go True.
Also publishes a confidence score so you can debug what each sensor sees.

Run to test:
  ros2 run stair_detector fusion_stair
  ros2 topic echo /stair/fusion_confidence

Subscribes:
  /stair/lidar_detected     std_msgs/Bool
  /stair/lidar_distance     std_msgs/Float32
  /stair/camera_detected    std_msgs/Bool
  /stair/camera_distance    std_msgs/Float32

Publishes:
  /stair_detected           std_msgs/Bool
  /stair_distance           std_msgs/Float32
  /stair/fusion_confidence  std_msgs/Float32  (0.0 to 1.0)
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32


class FusionStair(Node):

    def __init__(self):
        super().__init__('fusion_stair')

        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('trigger_distance_m', 2.0)

        rate = self.get_parameter('publish_rate_hz').value
        self.trigger_dist = self.get_parameter('trigger_distance_m').value

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
        self.pub_confidence = self.create_publisher(Float32, '/stair/fusion_confidence', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(f'FusionStair ready. trigger_dist={self.trigger_dist}m')

    def lidar_det_cb(self, msg: Bool):
        self.lidar_detected = msg.data

    def lidar_dist_cb(self, msg: Float32):
        self.lidar_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def camera_det_cb(self, msg: Bool):
        self.camera_detected = msg.data

    def camera_dist_cb(self, msg: Float32):
        self.camera_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def publish_result(self):
        both_agree = self.lidar_detected and self.camera_detected

        if not math.isnan(self.lidar_distance) and not math.isnan(self.camera_distance):
            distance = (self.lidar_distance + self.camera_distance) / 2.0
        elif not math.isnan(self.lidar_distance):
            distance = self.lidar_distance
        elif not math.isnan(self.camera_distance):
            distance = self.camera_distance
        else:
            distance = float('nan')

        close_enough = (not math.isnan(distance)) and (distance < self.trigger_dist)
        detected = both_agree and close_enough

        # 1.0 = both agree and close, 0.5 = one sensor close, 0.2 = one sensor far
        sensors_agree = int(self.lidar_detected) + int(self.camera_detected)
        if detected:
            confidence = 1.0
        elif sensors_agree == 1 and close_enough:
            confidence = 0.5
        elif sensors_agree == 2:
            confidence = 0.4
        elif sensors_agree == 1:
            confidence = 0.2
        else:
            confidence = 0.0

        self.pub_detected.publish(Bool(data=detected))
        self.pub_distance.publish(Float32(
            data=distance if not math.isnan(distance) else -1.0
        ))
        self.pub_confidence.publish(Float32(data=confidence))

        if detected:
            self.get_logger().info(
                f'Fusion: stairs confirmed at {distance:.2f}m (confidence={confidence:.1f})',
                throttle_duration_sec=1.0
            )
        elif sensors_agree > 0:
            self.get_logger().info(
                f'Fusion: partial (lidar={self.lidar_detected}, camera={self.camera_detected}, '
                f'dist={distance:.2f}m, confidence={confidence:.1f})',
                throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = FusionStair()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
