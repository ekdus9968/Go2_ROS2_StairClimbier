#!/usr/bin/env python3
"""
Particle Filter Node for Person Tracking.

Pipeline:
1. Initialize particles uniformly over search space
2. Every step (~10 Hz):
   - Predict: constant velocity + noise
   - Update: if /person_detected available, adjust weights
   - Resample: systematic + ESS threshold
   - Estimate: weighted mean position + covariance
3. Publish tracked position

Input:
  /person_detected (geometry_msgs/PoseArray) - Fused detection

Output:
  /person_tracked (geometry_msgs/PoseWithCovarianceStamped) - Tracked pose
  /particle_cloud (sensor_msgs/PointCloud2) - Debug particles
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import (
    PoseArray, Pose, PoseWithCovarianceStamped)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from person_tracker.particle_filter import ParticleFilter


class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

        # Parameters
        self.declare_parameter('num_particles', 300)
        self.declare_parameter('update_rate', 10.0)  # Hz

        # Search space (initialization)
        self.declare_parameter('search_x_min', -5.0)
        self.declare_parameter('search_x_max', 10.0)
        self.declare_parameter('search_y_min', -3.0)
        self.declare_parameter('search_y_max', 3.0)
        self.declare_parameter('search_z_min', 0.0)
        self.declare_parameter('search_z_max', 2.0)

        # Noise
        self.declare_parameter('pos_noise_xy', 0.05)
        self.declare_parameter('pos_noise_z', 0.02)
        self.declare_parameter('vel_noise_xy', 0.10)
        self.declare_parameter('vel_noise_z', 0.05)

        # Observation
        self.declare_parameter('obs_sigma', 0.10)
        self.declare_parameter('min_weight', 0.01)

        # Resampling
        self.declare_parameter('ess_ratio_threshold', 0.5)

        # Frames
        self.declare_parameter('frame_id', 'hesai_link')

        # Load
        n = self.get_parameter('num_particles').value
        rate = self.get_parameter('update_rate').value

        x_min = self.get_parameter('search_x_min').value
        x_max = self.get_parameter('search_x_max').value
        y_min = self.get_parameter('search_y_min').value
        y_max = self.get_parameter('search_y_max').value
        z_min = self.get_parameter('search_z_min').value
        z_max = self.get_parameter('search_z_max').value

        pos_xy = self.get_parameter('pos_noise_xy').value
        pos_z = self.get_parameter('pos_noise_z').value
        vel_xy = self.get_parameter('vel_noise_xy').value
        vel_z = self.get_parameter('vel_noise_z').value

        obs_sig = self.get_parameter('obs_sigma').value
        min_w = self.get_parameter('min_weight').value

        ess_ratio = self.get_parameter('ess_ratio_threshold').value

        self.frame_id = self.get_parameter('frame_id').value

        # Initialize Particle Filter
        self.pf = ParticleFilter(
            num_particles=n,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            z_range=(z_min, z_max),
            pos_noise_xy=pos_xy,
            pos_noise_z=pos_z,
            vel_noise_xy=vel_xy,
            vel_noise_z=vel_z,
            obs_sigma=obs_sig,
            min_weight=min_w,
            ess_ratio_threshold=ess_ratio,
        )
        self.pf.initialize_uniform()

        self.dt = 1.0 / rate
        self.latest_observation = None

        # Subscribers
        self.create_subscription(
            PoseArray, '/person_detected',
            self.detected_cb, 10)

        # Publishers
        self.pub_tracked = self.create_publisher(
            PoseWithCovarianceStamped, '/person_tracked', 10)
        self.pub_particles = self.create_publisher(
            PointCloud2, '/particle_cloud', 10)

        # Timer
        self.create_timer(self.dt, self.step)

        self.get_logger().info(
            f'Particle Filter Node started: n={n}, rate={rate}Hz')

    # ==========================
    # Callbacks
    # ==========================
    def detected_cb(self, msg: PoseArray):
        """Store latest observation (closest one if multiple)."""
        if len(msg.poses) == 0:
            return

        # Use first pose (closest)
        pose = msg.poses[0]
        self.latest_observation = np.array([
            pose.position.x,
            pose.position.y,
            pose.position.z,
        ], dtype=np.float32)

    # ==========================
    # Main step
    # ==========================
    def step(self):
        # Prediction
        self.pf.predict(self.dt)

        # Update if observation available
        if self.latest_observation is not None:
            self.pf.update(self.latest_observation)
            self.get_logger().info(
                f'Update with obs: {self.latest_observation}')
            self.latest_observation = None

        # Resample if needed
        resampled = self.pf.resample_if_needed()

        # Estimate
        pos, vel, cov = self.pf.estimate()

        if pos is None:
            self.get_logger().warn('Estimate returned None')
            return

        self.publish_tracked(pos, vel, cov)
        self.publish_particles()

    # ==========================
    # Publishing
    # ==========================
    def publish_tracked(self, pos, vel, cov):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])
        msg.pose.pose.orientation.w = 1.0

        # Covariance (6x6): only position part filled
        cov_flat = np.zeros(36, dtype=np.float64)
        # Position covariance in top-left 3x3
        for i in range(3):
            for j in range(3):
                cov_flat[i * 6 + j] = cov[i, j]
        msg.pose.covariance = cov_flat.tolist()

        self.pub_tracked.publish(msg)

    def publish_particles(self):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        # Combine positions with weights
        points = self.pf.particles[:, :3]  # (n, 3)
        weights = self.pf.weights  # (n,)

        # Concatenate (x, y, z, weight)
        pts = np.column_stack((points, weights))

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
            header, fields, pts.tolist())
        self.pub_particles.publish(cloud_msg)


def main():
    rclpy.init()
    node = ParticleFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()