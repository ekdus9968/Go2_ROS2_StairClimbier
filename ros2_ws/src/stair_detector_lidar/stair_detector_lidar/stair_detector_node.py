#!/usr/bin/env python3
"""
Stair Detector Node with roll-only tilt correction.

Applies inverse ROLL rotation to LiDAR points (yaw and pitch preserved).
- Roll: correct (remove side-to-side wobble)
- Pitch: preserve (natural stair-climbing posture)
- Yaw: preserve (heading info)

Input:
  /hesai/hesai_lidar_controller/out
  /person_tracked (optional)
  TF: world -> base

Output:
  /stair_detected_lidar
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

import tf2_ros
from tf2_ros import TransformException

from stair_detector_lidar.stair_algorithm import StairDetector


class StairDetectorNode(Node):
    def __init__(self):
        super().__init__('stair_detector_node')

        # Params
        self.declare_parameter('forward_min', 0.3)
        self.declare_parameter('forward_max', 5.0)
        self.declare_parameter('side_range', 2.5)
        self.declare_parameter('z_min', -0.5)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('person_exclusion_radius', 0.5)
        self.declare_parameter('person_timeout', 1.0)
        self.declare_parameter('kde_bandwidth', 0.02)
        self.declare_parameter('peak_height_ratio', 0.1)
        self.declare_parameter('peak_min_distance', 0.05)
        self.declare_parameter('z_slice_thickness', 0.03)
        self.declare_parameter('min_points_per_step', 20)
        self.declare_parameter('min_steps', 3)
        self.declare_parameter('step_height_min', 0.05)
        self.declare_parameter('step_height_max', 0.30)
        self.declare_parameter('height_std_ratio', 0.5)
        self.declare_parameter('xy_min_distance', 0.05)
        self.declare_parameter('xy_max_distance', 0.60)

        # Roll correction
        self.declare_parameter('use_roll_correction', True)
        self.declare_parameter('base_frame', 'base')
        self.declare_parameter('world_frame', 'world')

        params = {
            'forward_min': self.get_parameter('forward_min').value,
            'forward_max': self.get_parameter('forward_max').value,
            'side_range': self.get_parameter('side_range').value,
            'z_min': self.get_parameter('z_min').value,
            'z_max': self.get_parameter('z_max').value,
            'person_exclusion_radius': self.get_parameter(
                'person_exclusion_radius').value,
            'kde_bandwidth': self.get_parameter('kde_bandwidth').value,
            'peak_height_ratio': self.get_parameter(
                'peak_height_ratio').value,
            'peak_min_distance': self.get_parameter(
                'peak_min_distance').value,
            'z_slice_thickness': self.get_parameter(
                'z_slice_thickness').value,
            'min_points_per_step': self.get_parameter(
                'min_points_per_step').value,
            'min_steps': self.get_parameter('min_steps').value,
            'step_height_min': self.get_parameter('step_height_min').value,
            'step_height_max': self.get_parameter('step_height_max').value,
            'height_std_ratio': self.get_parameter(
                'height_std_ratio').value,
            'xy_min_distance': self.get_parameter('xy_min_distance').value,
            'xy_max_distance': self.get_parameter('xy_max_distance').value,
        }
        self.person_timeout = self.get_parameter('person_timeout').value
        self.use_roll = self.get_parameter('use_roll_correction').value
        self.base_frame = self.get_parameter('base_frame').value
        self.world_frame = self.get_parameter('world_frame').value

        self.detector = StairDetector(**params)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_person_pos = None
        self.latest_person_stamp = None

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            PointCloud2, '/hesai/hesai_lidar_controller/out',
            self.cloud_cb, qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/person_tracked',
            self.person_cb, 10)

        self.pub_detected = self.create_publisher(
            Bool, '/stair_detected_lidar', 10)

        self.frame_count = 0
        self.get_logger().info(
            f'Stair Detector Node started (roll correction: {self.use_roll})')

    def person_cb(self, msg):
        self.latest_person_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])
        self.latest_person_stamp = self.get_clock().now()

    def cloud_cb(self, msg: PointCloud2):
        arr = point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.column_stack((
            np.array(arr['x']),
            np.array(arr['y']),
            np.array(arr['z']),
        )).astype(np.float32)

        self.frame_count += 1

        roll = 0.0
        if self.use_roll:
            corrected, roll = self.apply_roll_correction(points)
            if corrected is not None:
                points = corrected

        person_pos = None
        if self.latest_person_pos is not None:
            age = (self.get_clock().now() -
                    self.latest_person_stamp).nanoseconds / 1e9
            if age < self.person_timeout:
                person_pos = self.latest_person_pos

        detected, info = self.detector.detect(points, person_pos)
        self.pub_detected.publish(Bool(data=detected))

        roll_str = f'roll={np.rad2deg(roll):.1f}°'
        if detected:
            self.get_logger().info(
                f'[Frame {self.frame_count}] STAIRS: '
                f'{info["num_steps"]} steps, '
                f'h={info["avg_step_height"]:.3f}m, '
                f'dir={np.rad2deg(info["direction_yaw"]):.1f}°, '
                f'{roll_str}')
        else:
            person_str = 'w/p' if person_pos is not None else 'no p'
            self.get_logger().info(
                f'[Frame {self.frame_count}] no stairs ({person_str}): '
                f'{info.get("reason", "unknown")[:50]}, '
                f'peaks={info.get("num_peaks", 0)}, '
                f'{roll_str}')

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
    node = StairDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()