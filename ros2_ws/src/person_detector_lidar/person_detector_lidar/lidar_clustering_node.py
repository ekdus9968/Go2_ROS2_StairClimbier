#!/usr/bin/env python3
"""
LiDAR Person Clustering Node with roll correction.

Roll correction:
- Roll: corrected (remove side-to-side wobble)
- Pitch: preserved (natural robot posture)
- Yaw: preserved (heading)

Pipeline:
1. Roll correction
2. Angle + distance filter
3. Ground removal
4. DBSCAN clustering
5. Shape check + sub-region search
6. Publish person candidates
"""

import numpy as np
from collections import Counter
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

        # Filters
        self.declare_parameter('angle_min_deg', -35.0)
        self.declare_parameter('angle_max_deg', 35.0)
        self.declare_parameter('distance_min', 0.6)
        self.declare_parameter('distance_max', 6.0)

        # Ground
        self.declare_parameter('ground_z_threshold', -0.30)

        # DBSCAN
        self.declare_parameter('dbscan_epsilon', 0.20)
        self.declare_parameter('dbscan_min_points_ref', 500.0)

        # Human shape
        self.declare_parameter('human_height_min', 0.5)
        self.declare_parameter('human_height_max', 2.0)
        self.declare_parameter('human_width_min', 0.15)
        self.declare_parameter('human_width_max', 0.7)
        self.declare_parameter('human_depth_min', 0.15)
        self.declare_parameter('human_depth_max', 0.7)

        # Sub-region
        self.declare_parameter('subregion_grid_size', 0.2)
        self.declare_parameter('subregion_min_points', 30)
        self.declare_parameter('subregion_top_k', 5)
        self.declare_parameter('subregion_box_size', 0.5)
        self.declare_parameter('cluster_large_threshold', 1.5)

        # Roll correction
        self.declare_parameter('use_roll_correction', True)
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('world_frame', 'world')

        # Frames
        self.declare_parameter('lidar_frame', 'hesai_link')

        # Load
        self.angle_min = np.deg2rad(self.get_parameter('angle_min_deg').value)
        self.angle_max = np.deg2rad(self.get_parameter('angle_max_deg').value)
        self.dist_min = self.get_parameter('distance_min').value
        self.dist_max = self.get_parameter('distance_max').value
        self.ground_z = self.get_parameter('ground_z_threshold').value
        self.dbscan_eps = self.get_parameter('dbscan_epsilon').value

        self.h_min = self.get_parameter('human_height_min').value
        self.h_max = self.get_parameter('human_height_max').value
        self.w_min = self.get_parameter('human_width_min').value
        self.w_max = self.get_parameter('human_width_max').value
        self.d_min = self.get_parameter('human_depth_min').value
        self.d_max = self.get_parameter('human_depth_max').value

        self.sub_grid = self.get_parameter('subregion_grid_size').value
        self.sub_min_pts = self.get_parameter('subregion_min_points').value
        self.sub_top_k = self.get_parameter('subregion_top_k').value
        self.sub_box = self.get_parameter('subregion_box_size').value
        self.large_thresh = self.get_parameter('cluster_large_threshold').value

        self.use_roll = self.get_parameter('use_roll_correction').value
        self.base_frame = self.get_parameter('base_frame').value
        self.world_frame = self.get_parameter('world_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PointCloud2, '/hesai/hesai_lidar_controller/out',
            self.cloud_cb, qos)

        self.pub_candidates = self.create_publisher(
            PoseArray, '/person_candidates', 10)
        self.pub_clusters = self.create_publisher(
            PointCloud2, '/lidar_clusters', 10)

        self.get_logger().info(
            f'Lidar Clustering Node started '
            f'(roll correction: {self.use_roll})')

    def cloud_cb(self, msg: PointCloud2):
        arr = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.column_stack((
            np.array(arr['x']),
            np.array(arr['y']),
            np.array(arr['z']),
        )).astype(np.float32)

        self.get_logger().info(f'[STAGE0] Raw points: {len(points)}')
        if len(points) < 100:
            return

        # Roll correction
        roll = 0.0
        if self.use_roll:
            corrected, roll = self.apply_roll_correction(points)
            if corrected is not None:
                points = corrected

        # STAGE 1
        angle = np.arctan2(points[:, 1], points[:, 0])
        dist = np.linalg.norm(points[:, :2], axis=1)
        mask = ((angle >= self.angle_min) & (angle <= self.angle_max) &
                (dist >= self.dist_min) & (dist <= self.dist_max))
        points = points[mask]
        self.get_logger().info(
            f'[STAGE1] After angle/dist filter: {len(points)}')

        # STAGE 2
        points = self.remove_ground_plane(points)
        self.get_logger().info(
            f'[STAGE2] After ground removal: {len(points)} '
            f'(roll={np.rad2deg(roll):.1f}°)')
        if len(points) < 30:
            return

        # STAGE 3
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=10).fit(points)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}
        self.get_logger().info(f'[STAGE3] Clusters: {len(unique_labels)}')

        # STAGE 4
        candidates = []
        cluster_pts_for_pub = []
        cluster_lbls_for_pub = []

        for label in unique_labels:
            cluster = points[labels == label]
            center = cluster.mean(axis=0)
            size = cluster.max(axis=0) - cluster.min(axis=0)
            cluster_dist = np.linalg.norm(center[:2])

            self.get_logger().info(
                f'  Cluster {label}: {len(cluster)} pts, '
                f'size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f}), '
                f'center=({center[0]:.2f},{center[1]:.2f},'
                f'{center[2]:.2f}), dist={cluster_dist:.2f}')

            if not (self.dist_min <= cluster_dist <= self.dist_max):
                continue

            if self.is_person_shape(size):
                self.get_logger().info('    ACCEPTED (direct)')
                candidates.append(center)
                cluster_pts_for_pub.append(cluster)
                cluster_lbls_for_pub.extend([label] * len(cluster))
                continue

            is_large = (size[0] > self.w_max * self.large_thresh or
                         size[1] > self.d_max * self.large_thresh)
            if is_large:
                self.get_logger().info('    Large, searching sub-regions')
                sub_centers = self.find_person_subregions(cluster)
                for sc in sub_centers:
                    self.get_logger().info(
                        f'    ACCEPTED (sub-region) at '
                        f'({sc[0]:.2f},{sc[1]:.2f},{sc[2]:.2f})')
                    candidates.append(sc)
                    cluster_pts_for_pub.append(cluster)
                    cluster_lbls_for_pub.extend([label] * len(cluster))
            else:
                self.get_logger().info('    REJECTED: shape')

        self.get_logger().info(
            f'[STAGE4] Person candidates: {len(candidates)}')

        if len(candidates) > 0:
            pose_array = PoseArray()
            pose_array.header.stamp = msg.header.stamp
            pose_array.header.frame_id = self.lidar_frame
            for c in candidates:
                pose = Pose()
                pose.position.x = float(c[0])
                pose.position.y = float(c[1])
                pose.position.z = float(c[2])
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)
            self.pub_candidates.publish(pose_array)
            self.get_logger().info(
                f'Published {len(candidates)} candidate(s)')

        if len(cluster_pts_for_pub) > 0:
            all_pts = np.vstack(cluster_pts_for_pub)
            all_lbl = np.array(cluster_lbls_for_pub, dtype=np.float32)
            pts_with_label = np.column_stack((all_pts, all_lbl))
            self.publish_cluster_cloud(msg.header, pts_with_label)

    def remove_ground_plane(self, points):
        if len(points) < 10:
            return points
        mask = points[:, 2] > self.ground_z
        return points[mask]

    def is_person_shape(self, size):
        w, d, h = size
        if not (self.h_min <= h <= self.h_max):
            return False
        if not (self.w_min <= w <= self.w_max):
            return False
        if not (self.d_min <= d <= self.d_max):
            return False
        return True

    def find_person_subregions(self, cluster_points):
        x = cluster_points[:, 0]
        y = cluster_points[:, 1]

        gx = np.floor(x / self.sub_grid).astype(int)
        gy = np.floor(y / self.sub_grid).astype(int)

        cells = Counter(zip(gx.tolist(), gy.tolist()))

        person_centers = []
        seen_positions = []

        for (grid_x, grid_y), count in cells.most_common(self.sub_top_k):
            if count < self.sub_min_pts:
                continue

            center_x = grid_x * self.sub_grid + self.sub_grid / 2
            center_y = grid_y * self.sub_grid + self.sub_grid / 2

            too_close = False
            for sx, sy in seen_positions:
                if abs(sx - center_x) < 0.3 and abs(sy - center_y) < 0.3:
                    too_close = True
                    break
            if too_close:
                continue

            box_mask = (
                (np.abs(cluster_points[:, 0] - center_x) < self.sub_box) &
                (np.abs(cluster_points[:, 1] - center_y) < self.sub_box)
            )
            box_points = cluster_points[box_mask]

            if len(box_points) < 30:
                continue

            b_size = box_points.max(axis=0) - box_points.min(axis=0)

            if self.is_person_shape(b_size):
                center = box_points.mean(axis=0)
                person_centers.append(center)
                seen_positions.append((center_x, center_y))

        return person_centers

    def publish_cluster_cloud(self, header, points_with_label):
        cloud_header = Header()
        cloud_header.stamp = header.stamp
        cloud_header.frame_id = self.lidar_frame
        fields = [
            PointField(name='x', offset=0,
                        datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                        datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                        datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12,
                        datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg = point_cloud2.create_cloud(
            cloud_header, fields, points_with_label.tolist())
        self.pub_clusters.publish(cloud_msg)

    # ==========================
    # Roll correction
    # ==========================
    def apply_roll_correction(self, points):
        """Apply inverse ROLL only (pitch and yaw preserved)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', once=True)
            return None, 0.0

        q = tf.transform.rotation
        roll, _, _ = self.quat_to_euler(q.x, q.y, q.z, q.w)

        if abs(roll) < 0.01:
            return points, roll

        cr, sr = np.cos(-roll), np.sin(-roll)
        Rx = np.array([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ])
        corrected = (Rx @ points.T).T
        return corrected, roll

    def quat_to_euler(self, x, y, z, w):
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw


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