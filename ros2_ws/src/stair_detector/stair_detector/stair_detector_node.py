#!/usr/bin/env python3
"""
Stair detector with LiDAR + depth camera fusion.

LiDAR (Hesai): Detects ground discontinuities ahead (rough region candidates)
Camera (D435i depth): Verifies step pattern and height

Both must agree → /stair_detected = True

Subscribes:
  /hesai/points          sensor_msgs/PointCloud2
  /d435i/d435i/depth/image_raw  sensor_msgs/Image

Publishes:
  /stair_detected        std_msgs/Bool
  /stair_distance        std_msgs/Float32   (m, NaN if not detected)
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge


class StairDetector(Node):

    def __init__(self):
        super().__init__('stair_detector')

        # Parameters
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('detection_range_m', 2.5)
        self.declare_parameter('step_height_min_m', 0.04)
        self.declare_parameter('step_height_max_m', 0.20)
        self.declare_parameter('min_steps_for_stairs', 2)
        self.declare_parameter('lidar_topic', '/hesai/hesai_lidar_controller/out')
        self.declare_parameter('depth_topic', '/d435i/d435i/depth/image_raw')

        rate = self.get_parameter('publish_rate_hz').value
        self.det_range = self.get_parameter('detection_range_m').value
        self.h_min = self.get_parameter('step_height_min_m').value
        self.h_max = self.get_parameter('step_height_max_m').value
        self.min_steps = self.get_parameter('min_steps_for_stairs').value
        lidar_topic = self.get_parameter('lidar_topic').value
        depth_topic = self.get_parameter('depth_topic').value

        self.bridge = CvBridge()
        self.lidar_detected = False
        self.lidar_distance = float('nan')
        self.camera_detected = False
        self.camera_distance = float('nan')

        self.create_subscription(PointCloud2, lidar_topic, self.lidar_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, 10)

        self.pub_detected = self.create_publisher(Bool, '/stair_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair_distance', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairDetector ready. Range={self.det_range}m, '
            f'h=[{self.h_min},{self.h_max}]m, min_steps={self.min_steps}'
        )

    def lidar_cb(self, msg: PointCloud2):
        """Detect stair pattern from LiDAR point cloud.
        
        Strategy: project front-region points to XZ plane, find z transitions
        consistent with stair steps.
        """
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
        except Exception:
            self.lidar_detected = False
            return

        if points.shape[0] < 100:
            self.lidar_detected = False
            return

        # Filter: front region only (LiDAR is mounted on Go2, +X forward)
        # Front cone: x in [0.3, det_range], y in [-0.5, 0.5]
        mask = (
            (points[:, 0] > 0.3) & (points[:, 0] < self.det_range) &
            (np.abs(points[:, 1]) < 0.5)
        )
        front = points[mask]

        if front.shape[0] < 50:
            self.lidar_detected = False
            return

        # Bin by x distance, find z distribution per bin
        bins = np.arange(0.3, self.det_range, 0.1)  # 10cm bins
        z_per_bin = []
        for i in range(len(bins) - 1):
            in_bin = (front[:, 0] >= bins[i]) & (front[:, 0] < bins[i + 1])
            if in_bin.sum() > 5:
                z_per_bin.append(np.median(front[in_bin, 2]))
            else:
                z_per_bin.append(np.nan)

        z_arr = np.array(z_per_bin)
        valid = ~np.isnan(z_arr)
        if valid.sum() < self.min_steps + 2:
            self.lidar_detected = False
            return

        # Count z-jumps consistent with step heights
        z_valid = z_arr[valid]
        jumps = np.diff(z_valid)
        step_jumps = (jumps > self.h_min) & (jumps < self.h_max)
        num_steps = step_jumps.sum()

        if num_steps >= self.min_steps:
            self.lidar_detected = True
            # Closest stair distance
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            valid_centers = bin_centers[valid][:-1]  # match diff length
            step_positions = valid_centers[step_jumps]
            self.lidar_distance = float(step_positions.min()) if len(step_positions) > 0 else float('nan')
        else:
            self.lidar_detected = False
            self.lidar_distance = float('nan')

    def depth_cb(self, msg: Image):
        """Detect stair pattern from depth image.
        
        Strategy: look at vertical center column, find discrete depth jumps
        consistent with stair edges.
        """
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception:
            self.camera_detected = False
            return

        h, w = depth.shape
        # Center column (10px wide), bottom 2/3 of image (closer ground)
        col = depth[h // 3:, w // 2 - 5: w // 2 + 5]
        col_median = np.nanmedian(col, axis=1)  # per row median

        # Smooth & find discrete jumps
        valid = ~np.isnan(col_median) & (col_median > 0.3) & (col_median < self.det_range)
        col_valid = col_median[valid]

        if len(col_valid) < 30:
            self.camera_detected = False
            return

        # Look at depth gradient — staircase shows up as alternating slopes
        diffs = np.diff(col_valid)
        # In a typical stair view: depth decreases then suddenly increases (riser)
        # Or steady decrease without big jumps (slope)
        # Count "edge" pixels — abrupt depth changes
        edges = np.abs(diffs) > 0.05  # 5cm change between adjacent pixels
        num_edges = edges.sum()

        # Many edges suggest stair pattern
        if num_edges >= self.min_steps * 2:
            self.camera_detected = True
            # Closest stair = minimum depth in the column
            self.camera_distance = float(np.min(col_valid))
        else:
            self.camera_detected = False
            self.camera_distance = float('nan')

    def publish_result(self):
        """Fusion: both must agree."""
        detected = self.lidar_detected and self.camera_detected

        # Distance: average of two if both valid
        if detected:
            distance = (self.lidar_distance + self.camera_distance) / 2.0
        else:
            distance = float('nan')

        self.pub_detected.publish(Bool(data=detected))

        d_msg = Float32()
        d_msg.data = distance if not math.isnan(distance) else -1.0
        self.pub_distance.publish(d_msg)

        if detected:
            self.get_logger().info(
                f'Stair detected at {distance:.2f}m '
                f'(LiDAR={self.lidar_distance:.2f}, Cam={self.camera_distance:.2f})',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = StairDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
