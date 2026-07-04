#!/usr/bin/env python3
"""
Stair geometry verifier — hard trigger for STAIR_CLIMB mode.

Design (from architecture.md):
  YOLO gives bbox → we project LiDAR points into camera frustum
  → filter points inside bbox → normal-based classification
  → extract first riser distance, tread depth, riser height, alignment

Currently a skeleton — LiDAR-only path for simulation without YOLO.
Will add YOLO ROI gating once yolo_stair_detector_node is ready.

Subscribes:
  /hesai/hesai_lidar_controller/out  sensor_msgs/PointCloud2
  /yolo/stair_detections  vision_msgs/Detection2DArray  (TODO)

Publishes:
  /stair_geometry  custom msg with:
    - detected: Bool
    - first_riser_distance: float32 (m, from Go2 base)
    - tread_depth: float32 (m)
    - riser_height: float32 (m)
    - heading_alignment_error: float32 (rad)
    - valid: Bool (all params within valid range)
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32


class StairGeometryVerifier(Node):

    def __init__(self):
        super().__init__('stair_geometry_verifier')

        # Valid range for stair parameters
        self.declare_parameter('tread_min_m', 0.20)
        self.declare_parameter('tread_max_m', 0.35)
        self.declare_parameter('riser_min_m', 0.08)
        self.declare_parameter('riser_max_m', 0.20)
        self.declare_parameter('detection_range_m', 3.0)
        self.declare_parameter('heading_tolerance_deg', 15.0)
        self.declare_parameter('normal_horizontal_thresh', 0.85)  # dot(normal, z) > this = horizontal
        self.declare_parameter('normal_vertical_thresh', 0.15)  # abs(dot(normal, z)) < this = vertical
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('lidar_topic', '/hesai/hesai_lidar_controller/out')

        self.tread_min = self.get_parameter('tread_min_m').value
        self.tread_max = self.get_parameter('tread_max_m').value
        self.riser_min = self.get_parameter('riser_min_m').value
        self.riser_max = self.get_parameter('riser_max_m').value
        self.det_range = self.get_parameter('detection_range_m').value
        self.heading_tol = math.radians(self.get_parameter('heading_tolerance_deg').value)
        self.n_h_thresh = self.get_parameter('normal_horizontal_thresh').value
        self.n_v_thresh = self.get_parameter('normal_vertical_thresh').value
        rate = self.get_parameter('publish_rate_hz').value
        lidar_topic = self.get_parameter('lidar_topic').value

        # Latest computed geometry
        self.latest_detected = False
        self.latest_first_riser_dist = float('nan')
        self.latest_tread = float('nan')
        self.latest_riser = float('nan')
        self.latest_heading_err = float('nan')
        self.latest_valid = False

        self.create_subscription(PointCloud2, lidar_topic, self.lidar_cb, 10)

        self.pub_valid = self.create_publisher(Bool, '/stair_geometry/valid', 10)
        self.pub_dist = self.create_publisher(Float32, '/stair_geometry/first_riser_distance', 10)
        self.pub_tread = self.create_publisher(Float32, '/stair_geometry/tread_depth', 10)
        self.pub_riser = self.create_publisher(Float32, '/stair_geometry/riser_height', 10)
        self.pub_heading = self.create_publisher(Float32, '/stair_geometry/heading_error', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairGeometryVerifier ready. tread=[{self.tread_min},{self.tread_max}]m, '
            f'riser=[{self.riser_min},{self.riser_max}]m, range={self.det_range}m'
        )

    def lidar_cb(self, msg: PointCloud2):
        """Normal-based classification of ROI point cloud.
        
        Steps:
        1. Read points into numpy
        2. Filter to front region (later: YOLO bbox ROI)
        3. Compute local normals (k-NN)
        4. Classify horizontal (tread) vs vertical (riser)
        5. Cluster tread/riser by height
        6. Extract first_riser_distance, tread_depth, riser_height
        7. Validate against param ranges
        """
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
        except Exception:
            self._reset()
            return

        if points.shape[0] < 200:
            self._reset()
            return

        # Filter to front region (temporary — will be YOLO bbox)
        mask = (
            (points[:, 0] > 0.3) & (points[:, 0] < self.det_range) &
            (np.abs(points[:, 1]) < 0.5) &
            (points[:, 2] > -0.3) & (points[:, 2] < 1.5)
        )
        roi = points[mask]

        if roi.shape[0] < 100:
            self._reset()
            return

        # === TODO: Normal computation ===
        # For each point, find k nearest neighbors, fit plane, extract normal
        # Currently placeholder — needs open3d or scipy KDTree
        # Skeleton implementation using scipy
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(roi)
            k = 15
            _, idx = tree.query(roi, k=k)
            normals = np.zeros_like(roi)
            for i in range(roi.shape[0]):
                neigh = roi[idx[i]]
                centered = neigh - neigh.mean(axis=0)
                cov = centered.T @ centered / k
                w, v = np.linalg.eigh(cov)
                normals[i] = v[:, 0]  # smallest eigenvalue = normal
        except ImportError:
            self.get_logger().warn('scipy required for normal computation')
            self._reset()
            return

        # Classify: horizontal (tread) vs vertical (riser)
        n_z = np.abs(normals[:, 2])
        horizontal_mask = n_z > self.n_h_thresh
        vertical_mask = n_z < self.n_v_thresh
        tread_points = roi[horizontal_mask]
        riser_points = roi[vertical_mask]

        if tread_points.shape[0] < 20 or riser_points.shape[0] < 20:
            self._reset()
            return

        # Cluster treads by z (horizontal surfaces at different heights)
        tread_z_sorted = np.sort(tread_points[:, 2])
        # Simple binning: find distinct z levels
        z_bins = self._cluster_1d(tread_z_sorted, gap=0.05)
        num_treads = len(z_bins)

        if num_treads < 2:
            self._reset()
            return

        # Riser height = diff between consecutive tread z levels
        riser_heights = np.diff([z.mean() for z in z_bins])
        median_riser = float(np.median(riser_heights))

        # Tread depth = distance between consecutive risers in x direction
        # For each tread level, find x extent
        tread_centers_x = []
        for z_group in z_bins:
            in_group_pts = tread_points[(tread_points[:, 2] > z_group.min() - 0.02) &
                                        (tread_points[:, 2] < z_group.max() + 0.02)]
            if len(in_group_pts) > 5:
                tread_centers_x.append(float(np.median(in_group_pts[:, 0])))
        if len(tread_centers_x) < 2:
            self._reset()
            return
        tread_depths = np.diff(sorted(tread_centers_x))
        median_tread = float(np.median(tread_depths))

        # First riser distance = minimum x in riser points
        first_riser_dist = float(riser_points[:, 0].min())

        # Heading error = angle of first riser plane normal wrt robot forward (x-axis)
        # Simplification: assume first riser is well-aligned if riser normals project mainly on x
        first_riser_mask = riser_points[:, 0] < first_riser_dist + 0.05
        if first_riser_mask.sum() > 5:
            fr_norms = normals[vertical_mask][first_riser_mask]
            mean_normal = fr_norms.mean(axis=0)
            heading_err = math.atan2(mean_normal[1], abs(mean_normal[0]))
        else:
            heading_err = 0.0

        # Validate
        valid = (
            self.tread_min <= median_tread <= self.tread_max and
            self.riser_min <= median_riser <= self.riser_max and
            abs(heading_err) < self.heading_tol
        )

        self.latest_detected = True
        self.latest_first_riser_dist = first_riser_dist
        self.latest_tread = median_tread
        self.latest_riser = median_riser
        self.latest_heading_err = heading_err
        self.latest_valid = valid

    def _cluster_1d(self, sorted_values, gap):
        """Cluster 1D sorted array into groups separated by gap."""
        if len(sorted_values) == 0:
            return []
        groups = [[sorted_values[0]]]
        for v in sorted_values[1:]:
            if v - groups[-1][-1] > gap:
                groups.append([v])
            else:
                groups[-1].append(v)
        return [np.array(g) for g in groups]

    def _reset(self):
        self.latest_detected = False
        self.latest_first_riser_dist = float('nan')
        self.latest_tread = float('nan')
        self.latest_riser = float('nan')
        self.latest_heading_err = float('nan')
        self.latest_valid = False

    def publish_result(self):
        self.pub_valid.publish(Bool(data=self.latest_valid))
        self.pub_dist.publish(Float32(
            data=self.latest_first_riser_dist if not math.isnan(self.latest_first_riser_dist) else -1.0
        ))
        self.pub_tread.publish(Float32(
            data=self.latest_tread if not math.isnan(self.latest_tread) else -1.0
        ))
        self.pub_riser.publish(Float32(
            data=self.latest_riser if not math.isnan(self.latest_riser) else -1.0
        ))
        self.pub_heading.publish(Float32(
            data=self.latest_heading_err if not math.isnan(self.latest_heading_err) else 0.0
        ))

        if self.latest_valid:
            self.get_logger().info(
                f'Valid stair: dist={self.latest_first_riser_dist:.2f}m, '
                f'tread={self.latest_tread:.2f}m, riser={self.latest_riser:.2f}m, '
                f'heading_err={math.degrees(self.latest_heading_err):.1f}°',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = StairGeometryVerifier()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
