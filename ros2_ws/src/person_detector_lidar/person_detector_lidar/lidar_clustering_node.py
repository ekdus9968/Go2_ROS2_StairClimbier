#!/usr/bin/env python3
"""
LiDAR Clustering Node for Person Detection.

Pipeline:
1. Receive PointCloud2 from Hesai LiDAR
2. Filter by angle (-35 to +35 degrees, front)
3. Filter by distance (0.5m to 3m)
4. Ground/stair plane removal (RANSAC)
5. DBSCAN clustering
6. Human-shape filter (bounding box + aspect ratio + point density)
7. Publish person candidates

Input:  /hesai/hesai_lidar_controller/out (sensor_msgs/PointCloud2)
Output: /person_candidates (geometry_msgs/PoseArray) - human-shape candidates
Debug:  /lidar_clusters (sensor_msgs/PointCloud2) - all clusters colored
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Header

from sklearn.cluster import DBSCAN

import tf2_ros
from tf2_ros import TransformException


class LidarClusteringNode(Node):
    def __init__(self):
        super().__init__('lidar_clustering_node')

        # ==========================
        # Parameters
        # ==========================
        # Input filtering
        self.declare_parameter('angle_min_deg', -35.0)
        self.declare_parameter('angle_max_deg', 35.0)
        self.declare_parameter('distance_min', 0.5)
        self.declare_parameter('distance_max', 3.0)

        # Ground removal (RANSAC)
        self.declare_parameter('ground_ransac_threshold', 0.05)  # 5 cm
        self.declare_parameter('ground_ransac_iterations', 100)

        # DBSCAN
        self.declare_parameter('dbscan_epsilon', 0.20)  # 20 cm
        self.declare_parameter('dbscan_min_points_ref', 500.0)
        # min_points = ref / (distance^2)

        # Human shape filter
        self.declare_parameter('human_height_min', 1.4)
        self.declare_parameter('human_height_max', 1.9)
        self.declare_parameter('human_width_min', 0.25)
        self.declare_parameter('human_width_max', 0.7)
        self.declare_parameter('human_depth_min', 0.2)
        self.declare_parameter('human_depth_max', 0.6)
        self.declare_parameter('aspect_ratio_min', 2.0)  # height/width

        # Frames
        self.declare_parameter('lidar_frame', 'hesai_link')
        self.declare_parameter('world_frame', 'world')

        # Load parameters
        self.angle_min = np.deg2rad(
            self.get_parameter('angle_min_deg').value)
        self.angle_max = np.deg2rad(
            self.get_parameter('angle_max_deg').value)
        self.dist_min = self.get_parameter('distance_min').value
        self.dist_max = self.get_parameter('distance_max').value

        self.ground_threshold = self.get_parameter(
            'ground_ransac_threshold').value
        self.ground_iterations = self.get_parameter(
            'ground_ransac_iterations').value

        self.dbscan_eps = self.get_parameter('dbscan_epsilon').value
        self.min_points_ref = self.get_parameter(
            'dbscan_min_points_ref').value

        self.h_min = self.get_parameter('human_height_min').value
        self.h_max = self.get_parameter('human_height_max').value
        self.w_min = self.get_parameter('human_width_min').value
        self.w_max = self.get_parameter('human_width_max').value
        self.d_min = self.get_parameter('human_depth_min').value
        self.d_max = self.get_parameter('human_depth_max').value
        self.aspect_min = self.get_parameter('aspect_ratio_min').value

        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.world_frame = self.get_parameter('world_frame').value

        # ==========================
        # TF listener
        # ==========================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ==========================
        # Publishers / Subscribers
        # ==========================
        # Best-effort QoS for LiDAR (high rate, tolerable to drop)
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            PointCloud2,
            '/hesai/hesai_lidar_controller/out',
            self.cloud_callback,
            qos,
        )

        self.pub_candidates = self.create_publisher(
            PoseArray, '/person_candidates', 10)
        self.pub_debug_clusters = self.create_publisher(
            PointCloud2, '/lidar_clusters', 10)

        self.get_logger().info('LiDAR Clustering Node started')
        self.get_logger().info(
            f'Angle range: [{np.rad2deg(self.angle_min):.1f}, '
            f'{np.rad2deg(self.angle_max):.1f}] deg')
        self.get_logger().info(
            f'Distance range: [{self.dist_min:.1f}, '
            f'{self.dist_max:.1f}] m')

    # ==========================
    # Main callback
    # ==========================
    def cloud_callback(self, msg: PointCloud2):
        # PointCloud2 -> numpy array (N, 3)
        points = self.pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            return

        # Step 1: Angle + distance filter (in LiDAR frame)
        points = self.filter_by_angle_distance(points)
        if len(points) < 10:
            self.get_logger().debug(
                f'After angle/distance filter: {len(points)} points')
            return

        # Step 2: Ground/stair plane removal
        points = self.remove_ground_plane(points)
        if len(points) < 10:
            return

        # Step 3: DBSCAN clustering
        cluster_labels = self.dbscan_cluster(points)
        num_clusters = len(set(cluster_labels)) - (
            1 if -1 in cluster_labels else 0)

        if num_clusters == 0:
            return

        # Step 4: Extract cluster info + human-shape filter
        person_candidates = self.filter_human_shape(points, cluster_labels)

        # Publish
        self.publish_candidates(person_candidates, msg.header.stamp)
        self.publish_debug_clusters(points, cluster_labels, msg.header.stamp)

    # ==========================
    # PointCloud2 conversion
    # ==========================
    def pointcloud2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        """Convert PointCloud2 to (N, 3) numpy array of XYZ."""
        try:
            # read_points returns a structured array
            gen = point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            arr = np.array(list(gen), dtype=np.float32)
            if arr.ndim == 1:
                # Handle structured array case
                arr = np.stack(
                    [arr['x'], arr['y'], arr['z']], axis=-1)
            return arr
        except Exception as e:
            self.get_logger().error(f'PointCloud2 conversion failed: {e}')
            return None

    # ==========================
    # Filtering
    # ==========================
    def filter_by_angle_distance(self, points: np.ndarray) -> np.ndarray:
        """Filter points by front angle and distance range."""
        x = points[:, 0]
        y = points[:, 1]

        # Angle from x-axis (forward)
        angles = np.arctan2(y, x)
        # Distance in horizontal plane
        distances = np.sqrt(x**2 + y**2)

        mask = (
            (angles >= self.angle_min) &
            (angles <= self.angle_max) &
            (distances >= self.dist_min) &
            (distances <= self.dist_max)
        )
        return points[mask]

    def remove_ground_plane(self, points: np.ndarray) -> np.ndarray:
        """Remove ground/stair planes using RANSAC.
        
        Simple approach: iteratively fit planes and remove points on them.
        For our case, we mainly want to remove the ground plane.
        """
        if len(points) < 50:
            return points

        remaining = points.copy()
        # One-pass RANSAC for ground plane
        best_inliers_mask = self.ransac_plane(
            remaining, self.ground_threshold, self.ground_iterations)

        if best_inliers_mask is None:
            return remaining

        # Return non-ground points
        return remaining[~best_inliers_mask]

    def ransac_plane(self, points: np.ndarray, threshold: float,
                     iterations: int) -> np.ndarray:
        """Simple RANSAC plane fitting. Returns inliers mask or None."""
        if len(points) < 3:
            return None

        best_count = 0
        best_mask = None

        n = len(points)
        rng = np.random.default_rng()

        for _ in range(iterations):
            # Random 3 points
            idx = rng.choice(n, size=3, replace=False)
            p1, p2, p3 = points[idx]

            # Plane normal from cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            # Prefer horizontal planes (ground/stairs)
            # Normal z-component should be near 1 (horizontal plane)
            if abs(normal[2]) < 0.7:
                continue

            # Plane equation: n . (p - p1) = 0
            d = -np.dot(normal, p1)
            distances = np.abs(points @ normal + d)

            inlier_mask = distances < threshold
            count = int(inlier_mask.sum())

            if count > best_count:
                best_count = count
                best_mask = inlier_mask

        # Require a minimum inlier count to accept
        if best_count < 50:
            return None
        return best_mask

    # ==========================
    # Clustering
    # ==========================
    def dbscan_cluster(self, points: np.ndarray) -> np.ndarray:
        """DBSCAN clustering. Returns cluster labels (-1 = noise)."""
        if len(points) < 5:
            return np.full(len(points), -1)

        # min_samples set as small default; per-cluster refinement in filter step
        db = DBSCAN(eps=self.dbscan_eps, min_samples=10)
        labels = db.fit_predict(points)
        return labels

    # ==========================
    # Human shape filter
    # ==========================
    def filter_human_shape(self, points: np.ndarray,
                            labels: np.ndarray) -> list:
        """Filter clusters by human shape. Returns list of (center, info)."""
        candidates = []

        unique_labels = set(labels)
        unique_labels.discard(-1)  # remove noise

        for label in unique_labels:
            cluster_pts = points[labels == label]

            if len(cluster_pts) < 10:
                continue

            # Bounding box
            min_xyz = cluster_pts.min(axis=0)
            max_xyz = cluster_pts.max(axis=0)
            size = max_xyz - min_xyz  # [dx, dy, dz]

            width = size[0]
            depth = size[1]
            height = size[2]

            # Filter by size
            if not (self.h_min <= height <= self.h_max):
                continue
            # Width can be width or depth (person can be facing any direction)
            wh_dim = min(width, depth)
            if not (self.w_min <= wh_dim <= self.w_max):
                continue
            other_dim = max(width, depth)
            if not (self.d_min <= other_dim <= self.d_max):
                continue

            # Aspect ratio (height / horizontal min)
            if wh_dim > 0:
                aspect = height / wh_dim
                if aspect < self.aspect_min:
                    continue

            # Distance-adaptive min_points check
            center = cluster_pts.mean(axis=0)
            distance = np.linalg.norm(center[:2])  # horizontal distance
            min_points = max(30, int(self.min_points_ref / (distance**2)))

            if len(cluster_pts) < min_points:
                self.get_logger().debug(
                    f'Cluster {label}: {len(cluster_pts)} pts < {min_points} '
                    f'required at {distance:.2f}m')
                continue

            # Point density check
            volume = width * depth * height
            if volume > 0:
                density = len(cluster_pts) / volume
                min_density = self.min_points_ref / (distance**2)
                if density < min_density * 0.5:  # relaxed threshold
                    continue

            candidates.append({
                'label': label,
                'center': center,
                'size': size,
                'num_points': len(cluster_pts),
                'distance': distance,
            })

        return candidates

    # ==========================
    # Publishing
    # ==========================
    def publish_candidates(self, candidates: list, stamp):
        """Publish person candidates as PoseArray."""
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.lidar_frame

        for c in candidates:
            pose = Pose()
            pose.position.x = float(c['center'][0])
            pose.position.y = float(c['center'][1])
            pose.position.z = float(c['center'][2])
            pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.pub_candidates.publish(msg)

        if len(candidates) > 0:
            self.get_logger().info(
                f'Published {len(candidates)} person candidate(s)')
            for c in candidates:
                self.get_logger().debug(
                    f'  Label {c["label"]}: pos=({c["center"][0]:.2f}, '
                    f'{c["center"][1]:.2f}, {c["center"][2]:.2f}) '
                    f'dist={c["distance"]:.2f}m pts={c["num_points"]}')

    def publish_debug_clusters(self, points: np.ndarray,
                                labels: np.ndarray, stamp):
        """Publish clustered points with cluster label as intensity."""
        header = Header()
        header.stamp = stamp
        header.frame_id = self.lidar_frame

        # Combine points with labels as intensity
        pts_with_label = np.column_stack(
            (points, labels.astype(np.float32)))

        # Only publish clustered (non-noise) points
        mask = labels >= 0
        if not mask.any():
            return

        pts_pub = pts_with_label[mask]

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32,
                        count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32,
                        count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32,
                        count=1),
            PointField(name='intensity', offset=12,
                        datatype=PointField.FLOAT32, count=1),
        ]

        cloud_msg = point_cloud2.create_cloud(
            header, fields, pts_pub.tolist())
        self.pub_debug_clusters.publish(cloud_msg)


def main():
    rclpy.init()
    node = LidarClusteringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
