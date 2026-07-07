#!/usr/bin/env python3
"""
stair_consensus.py

Sits in front of fusion_stair.py and yolo_stair_detector.py. Treats each of
their outputs as one vote and decides a 3-state robot mode. Does not control
robot motion - classification only.

  Vote A - fusion_stair.py's output. Already its own internal consensus
           (lidar + camera both agree AND within trigger_distance_m), so
           a "yes" here is already a strong signal.
  Vote B - yolo_stair_detector.py's output. Your custom-trained model's
           yes/no vote. Its distance is always -1.0 (no real depth), so
           it's only used for the vote, never for distance.

  both agree      -> STAIR_MODE
  only one agrees -> APPROACH_MODE
  neither         -> NONE

*** TOPIC COLLISION NOTE ***
fusion_stair.py publishes to /stair_detected and /stair_distance - the same
names this node needs for its OWN final output. Remap fusion_stair.py's
topics at launch:

  ros2 run stair_detector fusion_stair --ros-args \
    -r /stair_detected:=/fusion/stair_detected \
    -r /stair_distance:=/fusion/stair_distance

  ros2 run stair_detector yolo_stair_detector

  ros2 run stair_detector stair_consensus

  ros2 topic echo /stair/robot_mode
  ros2 topic echo /stair_distance

Subscribes:
  /fusion/stair_detected     std_msgs/Bool
  /fusion/stair_distance     std_msgs/Float32
  /stair/yolo_detected       std_msgs/Bool

Publishes:
  /stair/robot_mode          std_msgs/String  ('STAIR_MODE'|'APPROACH_MODE'|'NONE')
  /stair_distance            std_msgs/Float32  (meters, -1 if not available)

NOTE: named /stair/robot_mode (not /robot_mode) to avoid colliding with the
high_level_planner package's state_machine_node, which may eventually own
/robot_mode using the RobotMode.msg custom type once it's built out for real.
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class StairConsensus(Node):

    def __init__(self):
        super().__init__('stair_consensus')

        self.declare_parameter('fusion_detected_topic', '/fusion/stair_detected')
        self.declare_parameter('fusion_distance_topic', '/fusion/stair_distance')
        self.declare_parameter('yolo_detected_topic', '/stair/yolo_detected')
        self.declare_parameter('publish_rate_hz', 5.0)

        fusion_det_topic = self.get_parameter('fusion_detected_topic').value
        fusion_dist_topic = self.get_parameter('fusion_distance_topic').value
        yolo_det_topic = self.get_parameter('yolo_detected_topic').value
        rate = self.get_parameter('publish_rate_hz').value

        self.fusion_detected = False
        self.fusion_distance = float('nan')
        self.yolo_detected = False

        self.create_subscription(Bool, fusion_det_topic, self.fusion_det_cb, 10)
        self.create_subscription(Float32, fusion_dist_topic, self.fusion_dist_cb, 10)
        self.create_subscription(Bool, yolo_det_topic, self.yolo_det_cb, 10)

        self.pub_mode = self.create_publisher(String, '/stair/robot_mode', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair_distance', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairConsensus ready. '
            f'fusion={fusion_det_topic}, yolo={yolo_det_topic}'
        )

    def fusion_det_cb(self, msg: Bool):
        self.fusion_detected = msg.data

    def fusion_dist_cb(self, msg: Float32):
        self.fusion_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def yolo_det_cb(self, msg: Bool):
        self.yolo_detected = msg.data

    def publish_result(self):
        agree_count = int(self.fusion_detected) + int(self.yolo_detected)

        if agree_count == 2:
            mode = 'STAIR_MODE'
        elif agree_count == 1:
            mode = 'APPROACH_MODE'
        else:
            mode = 'NONE'

        distance = self.fusion_distance

        self.pub_mode.publish(String(data=mode))
        self.pub_distance.publish(Float32(
            data=distance if not math.isnan(distance) else -1.0
        ))

        if mode != 'NONE':
            self.get_logger().info(
                f'{mode} (fusion={self.fusion_detected}, yolo={self.yolo_detected}, '
                f'dist={distance if not math.isnan(distance) else -1:.2f}m)',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().debug(
                f'NONE (fusion={self.fusion_detected}, yolo={self.yolo_detected})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = StairConsensus()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()