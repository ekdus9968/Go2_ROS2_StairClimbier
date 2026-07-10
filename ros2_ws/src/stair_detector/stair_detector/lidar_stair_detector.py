#!/usr/bin/env python3
"""
lidar_stair_detector.py

LiDAR only stair detector, run this on its own to test before fusion.

Bins points by forward distance, checks height jumps between bins.
If enough jumps match stair step heights, stairs are detected.

Run to test:
  ros2 run stair_detector lidar_stair_detector
  ros2 topic echo /stair/lidar_detected

Publishes:
  /stair/lidar_detected     std_msgs/Bool
  /stair/lidar_distance     std_msgs/Float32  (meters, -1 if not detected)
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32


class LidarStairDetector(Node):

    def __init__(self):
        super().__init__('lidar_stair_detector')
        self.declare_parameter('lidar_topic', '/hesai/hesai_lidar_controller/out')
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('detection_range_m', 2.5)
        self.declare_parameter('step_height_min_m', 0.04)
        self.declare_parameter('step_height_max_m', 0.20)
        self.declare_parameter('min_steps_for_stairs', 2)
        self.declare_parameter('front_cone_width_m', 0.5)

        lidar_topic = self.get_parameter('lidar_topic').value
        rate = self.get_parameter('publish_rate_hz').value
        self.det_range = self.get_parameter('detection_range_m').value
        self.h_min = self.get_parameter('step_height_min_m').value
        self.h_max = self.get_parameter('step_height_max_m').value
        self.min_steps = self.get_parameter('min_steps_for_stairs').value
        self.cone_width = self.get_parameter('front_cone_width_m').value

        self.detected = False
        self.distance = float('nan')

        self.create_subscription(PointCloud2, lidar_topic, self.lidar_cb, 10)
        self.pub_detected = self.create_publisher(Bool, '/stair/lidar_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair/lidar_distance', 10)
        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(f'LidarStairDetector ready on {lidar_topic}')

    def lidar_cb(self, msg: PointCloud2):
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
        except Exception as e:
            self.get_logger().debug(f'point cloud read failed: {e}')
            self.detected = False
            return

        if points.shape[0] < 100:
            self.detected = False
            return

        mask = (
            (points[:, 0] > 0.3) &
            (points[:, 0] < self.det_range) &
            (np.abs(points[:, 1]) < self.cone_width)
        )
        front = points[mask]

        if front.shape[0] < 50:
            self.detected = False
            return

        # 10cm bins along x, median z per bin
        bins = np.arange(0.3, self.det_range, 0.1)
        z_per_bin = []
        for i in range(len(bins) - 1):
            in_bin = (front[:, 0] >= bins[i]) & (front[:, 0] < bins[i + 1])
            if in_bin.sum() > 5:
                z_per_bin.append(float(np.median(front[in_bin, 2])))
            else:
                z_per_bin.append(float('nan'))

        z_arr = np.array(z_per_bin)
        valid = ~np.isnan(z_arr)

        if valid.sum() < self.min_steps + 2:
            self.detected = False
            return

        z_valid = z_arr[valid]
        jumps = np.diff(z_valid)
        step_jumps = (jumps > self.h_min) & (jumps < self.h_max)
        num_steps = int(step_jumps.sum())

        if num_steps >= self.min_steps:
            self.detected = True
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            valid_centers = bin_centers[valid][:-1]
            step_positions = valid_centers[step_jumps]
            self.distance = (
                float(step_positions.min()) if len(step_positions) > 0
                else float('nan')
            )
        else:
            self.detected = False
            self.distance = float('nan')

    def publish_result(self):
        import math
        self.pub_detected.publish(Bool(data=self.detected))
        self.pub_distance.publish(Float32(
            data=self.distance if not math.isnan(self.distance) else -1.0
        ))
        if self.detected:
            self.get_logger().info(
                f'LiDAR: stairs at {self.distance:.2f}m',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = LidarStairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
