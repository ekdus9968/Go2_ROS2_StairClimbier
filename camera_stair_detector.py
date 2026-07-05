#!/usr/bin/env python3
"""
camera_stair_detector.py

Depth camera only stair detector, run this on its own to test before fusion.

Looks at a vertical column of pixels in the depth image and counts sharp
depth jumps. Stairs show up as repeated sudden changes scanning downward.

Run to test:
  ros2 run stair_detector camera_stair_detector
  ros2 topic echo /stair/camera_detected

Publishes:
  /stair/camera_detected    std_msgs/Bool
  /stair/camera_distance    std_msgs/Float32  (meters, -1 if not detected)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge


class CameraStairDetector(Node):

    def __init__(self):
        super().__init__('camera_stair_detector')

        # *** CHANGE THIS: run `ros2 topic list | grep -i depth` to find your topic ***
        self.declare_parameter('depth_topic', '/depth_camera/depth/image_raw')
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('detection_range_m', 2.5)
        self.declare_parameter('min_steps_for_stairs', 2)
        self.declare_parameter('edge_threshold_m', 0.05)

        depth_topic = self.get_parameter('depth_topic').value
        rate = self.get_parameter('publish_rate_hz').value
        self.det_range = self.get_parameter('detection_range_m').value
        self.min_steps = self.get_parameter('min_steps_for_stairs').value
        self.edge_thresh = self.get_parameter('edge_threshold_m').value

        self.bridge = CvBridge()
        self.detected = False
        self.distance = float('nan')

        self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.pub_detected = self.create_publisher(Bool, '/stair/camera_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair/camera_distance', 10)
        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(f'CameraStairDetector ready on {depth_topic}')

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().debug(f'depth image conversion failed: {e}')
            self.detected = False
            return

        h, w = depth.shape

        # center column, bottom 2/3 of image where stairs appear
        col = depth[h // 3:, w // 2 - 5: w // 2 + 5]
        col_median = np.nanmedian(col, axis=1)

        valid = (
            ~np.isnan(col_median) &
            (col_median > 0.3) &
            (col_median < self.det_range)
        )
        col_valid = col_median[valid]

        if len(col_valid) < 20:
            self.detected = False
            return

        diffs = np.diff(col_valid)
        edges = np.abs(diffs) > self.edge_thresh
        num_edges = int(edges.sum())

        if num_edges >= self.min_steps * 2:
            self.detected = True
            self.distance = float(np.min(col_valid))
        else:
            self.detected = False
            self.distance = float('nan')

    def publish_result(self):
        self.pub_detected.publish(Bool(data=self.detected))
        self.pub_distance.publish(Float32(
            data=self.distance if not math.isnan(self.distance) else -1.0
        ))
        if self.detected:
            self.get_logger().info(
                f'Camera: stairs at {self.distance:.2f}m',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = CameraStairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
