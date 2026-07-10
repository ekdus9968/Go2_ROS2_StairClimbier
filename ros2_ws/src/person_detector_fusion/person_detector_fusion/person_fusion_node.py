#!/usr/bin/env python3
"""
Person Detection Fusion Node.

Combines:
  - LiDAR person candidates (/person_candidates)
  - Camera verified persons (/person_verified)

Fusion logic:
  1. Keep latest LiDAR candidates
  2. Keep latest Camera verified poses
  3. For each LiDAR candidate, check if matched with camera verified
     (distance < match_threshold)
  4. Publish matched candidates as final detections

Input:
  /person_candidates (geometry_msgs/PoseArray) - LiDAR
  /person_verified (geometry_msgs/PoseArray) - Camera

Output:
  /person_detected (geometry_msgs/PoseArray) - Fused final detections
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose


class PersonFusionNode(Node):
    def __init__(self):
        super().__init__('person_fusion_node')

        # Parameters
        self.declare_parameter('match_threshold', 0.3)  # 30 cm
        self.declare_parameter('camera_timeout', 1.0)  # 1 sec

        self.match_thresh = self.get_parameter('match_threshold').value
        self.camera_timeout = self.get_parameter('camera_timeout').value

        # Latest data
        self.latest_lidar_candidates = None
        self.latest_lidar_stamp = None
        self.latest_camera_verified = None
        self.latest_camera_stamp = None

        # Subscribers
        self.create_subscription(
            PoseArray, '/person_candidates',
            self.lidar_cb, 10)
        self.create_subscription(
            PoseArray, '/person_verified',
            self.camera_cb, 10)

        # Publisher
        self.pub_detected = self.create_publisher(
            PoseArray, '/person_detected', 10)

        # Fusion timer (10 Hz)
        self.create_timer(0.1, self.fusion_step)

        self.get_logger().info('Person Fusion Node started')
        self.get_logger().info(
            f'Match threshold: {self.match_thresh} m, '
            f'Camera timeout: {self.camera_timeout} s')

    # ==========================
    # Callbacks
    # ==========================
    def lidar_cb(self, msg: PoseArray):
        self.latest_lidar_candidates = msg
        self.latest_lidar_stamp = self.get_clock().now()

    def camera_cb(self, msg: PoseArray):
        self.latest_camera_verified = msg
        self.latest_camera_stamp = self.get_clock().now()

    # ==========================
    # Fusion
    # ==========================
    def fusion_step(self):
        # Need LiDAR data
        if self.latest_lidar_candidates is None:
            return
        if len(self.latest_lidar_candidates.poses) == 0:
            return

        # Check Camera data timeout
        now = self.get_clock().now()
        camera_valid = False
        if self.latest_camera_stamp is not None:
            time_since_camera = (
                now - self.latest_camera_stamp).nanoseconds / 1e9
            if time_since_camera < self.camera_timeout:
                camera_valid = True

        if not camera_valid or self.latest_camera_verified is None:
            self.get_logger().debug(
                'No valid camera verified data - skipping fusion')
            return
        if len(self.latest_camera_verified.poses) == 0:
            return

        # Match LiDAR candidates with Camera verified
        detected_poses = []
        for lidar_pose in self.latest_lidar_candidates.poses:
            l_pos = np.array([
                lidar_pose.position.x,
                lidar_pose.position.y,
                lidar_pose.position.z])

            for camera_pose in self.latest_camera_verified.poses:
                c_pos = np.array([
                    camera_pose.position.x,
                    camera_pose.position.y,
                    camera_pose.position.z])

                # Distance in x-y plane
                dist = np.linalg.norm(l_pos[:2] - c_pos[:2])

                if dist < self.match_thresh:
                    # Matched
                    detected_poses.append(lidar_pose)
                    self.get_logger().info(
                        f'MATCHED person at '
                        f'({l_pos[0]:.2f}, {l_pos[1]:.2f}, {l_pos[2]:.2f}) '
                        f'dist={dist:.2f}m')
                    break  # only one match per LiDAR candidate

        # Publish detected
        if len(detected_poses) > 0:
            msg = PoseArray()
            msg.header = self.latest_lidar_candidates.header
            msg.poses = detected_poses
            self.pub_detected.publish(msg)


def main():
    rclpy.init()
    node = PersonFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()