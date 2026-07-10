#!/usr/bin/env python3
"""
Camera Keypoint Detection Node for Person Verification.

Pipeline:
1. Receive person candidates from LiDAR (/person_candidates)
2. Project LiDAR 3D position -> Camera image coordinates
3. Crop 640x640 region around projected point
4. Run YOLOv11-pose on cropped region
5. Verify person by keypoint groups:
   - Group A (required): lower body (knees 13-14, ankles 15-16), 3+ detected
   - Group B (alternative): torso (hips 11-12), 2+ detected
6. Publish verified persons

Input:
  /person_candidates (geometry_msgs/PoseArray) - from LiDAR
  /d435i/d435i/image_raw (sensor_msgs/Image) - RGB
  /d435i/d435i/camera_info (sensor_msgs/CameraInfo) - intrinsics
  TF: hesai_link -> d435i_link

Output:
  /person_verified (geometry_msgs/PoseArray) - verified person positions
  /person_debug_image (sensor_msgs/Image) - annotated image (debug)
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException

from ultralytics import YOLO


# COCO keypoint indices
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_EAR = 3
KP_RIGHT_EAR = 4
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 14
KP_LEFT_ANKLE = 15
KP_RIGHT_ANKLE = 16

# Group definitions
GROUP_A_LOWER = [KP_LEFT_KNEE, KP_RIGHT_KNEE,
                  KP_LEFT_ANKLE, KP_RIGHT_ANKLE]  # 4 keypoints
GROUP_B_TORSO = [KP_LEFT_HIP, KP_RIGHT_HIP]  # 2 keypoints


class CameraKeypointNode(Node):
    def __init__(self):
        super().__init__('camera_keypoint_node')

        # Parameters
        self.declare_parameter('yolo_model', 'yolo11n-pose.pt')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('crop_size', 640)

        self.declare_parameter('camera_frame', 'd435i_link')
        self.declare_parameter('lidar_frame', 'hesai_link')

        # Verification thresholds
        self.declare_parameter('group_a_required', 3)  # 3+ lower body kpts
        self.declare_parameter('group_b_required', 2)  # 2+ torso kpts

        # Load params
        model_path = self.get_parameter('yolo_model').value
        self.conf_thresh = self.get_parameter('conf_threshold').value
        self.crop_size = self.get_parameter('crop_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.lidar_frame = self.get_parameter('lidar_frame').value
        self.group_a_req = self.get_parameter('group_a_required').value
        self.group_b_req = self.get_parameter('group_b_required').value

        # Load YOLO
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info('YOLO loaded')

        # CV Bridge
        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics (from CameraInfo)
        self.camera_matrix = None
        self.image_width = None
        self.image_height = None

        # Latest image
        self.latest_image = None
        self.latest_image_stamp = None

        # QoS
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.create_subscription(
            CameraInfo, '/d435i/d435i/camera_info',
            self.camera_info_cb, qos)
        self.create_subscription(
            Image, '/d435i/d435i/image_raw',
            self.image_cb, qos)
        self.create_subscription(
            PoseArray, '/person_candidates',
            self.candidates_cb, 10)

        # Publishers
        self.pub_verified = self.create_publisher(
            PoseArray, '/person_verified', 10)
        self.pub_debug = self.create_publisher(
            Image, '/person_debug_image', 10)

        self.get_logger().info('Camera Keypoint Node started')

    # ==========================
    # Callbacks
    # ==========================
    def camera_info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(
                f'Camera info: {msg.width}x{msg.height}, '
                f'fx={self.camera_matrix[0,0]:.1f}')

    def image_cb(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_image_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')

    def candidates_cb(self, msg: PoseArray):
        if self.camera_matrix is None:
            self.get_logger().warn('No camera info yet')
            return
        if self.latest_image is None:
            self.get_logger().warn('No image yet')
            return
        if len(msg.poses) == 0:
            return

        # Get TF: lidar_frame -> camera_frame
        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame, self.lidar_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        # Rotation matrix + translation
        q = tf.transform.rotation
        t = tf.transform.translation
        R = self.quat_to_matrix(q.x, q.y, q.z, q.w)
        T = np.array([t.x, t.y, t.z])

        # Process each candidate
        verified_poses = []
        debug_image = self.latest_image.copy()

        for i, pose in enumerate(msg.poses):
            # LiDAR frame -> Camera frame
            p_lidar = np.array([
                pose.position.x, pose.position.y, pose.position.z])
            p_camera = R @ p_lidar + T

            # Camera optical frame convention:
            # x=right, y=down, z=forward
            # But d435i_link is typically x=forward, y=left, z=up
            # We need to check which frame we're working with

            # Assume d435i_link is REP-103 (x=forward, y=left, z=up)
            # Transform to camera optical frame:
            # x_opt = -y_link (right)
            # y_opt = -z_link (down)
            # z_opt =  x_link (forward)
            p_opt = np.array([-p_camera[1], -p_camera[2], p_camera[0]])

            # Skip if behind camera
            if p_opt[2] <= 0.1:
                self.get_logger().debug(
                    f'Candidate {i}: behind camera (z={p_opt[2]:.2f})')
                continue

            # Project to image plane
            u = (self.camera_matrix[0, 0] * p_opt[0] / p_opt[2]
                 + self.camera_matrix[0, 2])
            v = (self.camera_matrix[1, 1] * p_opt[1] / p_opt[2]
                 + self.camera_matrix[1, 2])

            u = int(u)
            v = int(v)

            self.get_logger().info(
                f'Candidate {i}: LiDAR ({p_lidar[0]:.2f},{p_lidar[1]:.2f},'
                f'{p_lidar[2]:.2f}) -> Image ({u},{v})')

            # Skip if outside image
            if not (0 <= u < self.image_width
                    and 0 <= v < self.image_height):
                self.get_logger().info(
                    f'  Outside image ({u},{v})')
                continue

            # Crop region (640x640 around projected point)
            # Distance-based crop size
            distance = np.linalg.norm(p_camera)
            if distance < 1.5:
                half = 400  # 800x800
            elif distance < 3.0:
                half = 320  # 640x640
            else:
                half = 250  # 500x500
            
            self.get_logger().info(
                f'  Distance: {distance:.2f}m, crop: {half*2}x{half*2}')
            
            u_min = max(0, u - half)
            u_max = min(self.image_width, u + half)
            v_min = max(0, v - half)
            v_max = min(self.image_height, v + half)

            crop = self.latest_image[v_min:v_max, u_min:u_max]
            if crop.size == 0:
                continue

            # Run YOLO
            results = self.model(
                crop, conf=self.conf_thresh, verbose=False)
            if len(results) == 0:
                continue

            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                continue

            # Get first person's keypoints
            # (Assume closest to center is target)
            # Multiple persons: pick one closest to LiDAR predicted position
            crop_center_x = half
            crop_center_y = half
            max_kpt_distance = 200  # pixels

            # GPU -> CPU
            # all_keypoints = result.keypoints.data  # (num_persons, 17, 3)
            all_keypoints = result.keypoints.data.cpu().numpy()
            
            best_person_idx = -1
            best_distance = float('inf')

            for p_idx in range(len(all_keypoints)):
                person_kpts = all_keypoints[p_idx]
                # Valid keypoints
                valid_mask = person_kpts[:, 2] > self.conf_thresh
                if valid_mask.sum() == 0:
                    continue
                
                # Keypoint center
                valid_kpts = person_kpts[valid_mask]
                kpt_center_x = valid_kpts[:, 0].mean()
                kpt_center_y = valid_kpts[:, 1].mean()
                
                # Distance from crop center (LiDAR expected)
                dist = np.sqrt(
                    (kpt_center_x - crop_center_x)**2 +
                    (kpt_center_y - crop_center_y)**2)
                
                if dist < best_distance:
                    best_distance = dist
                    best_person_idx = p_idx
            
            if best_person_idx < 0:
                self.get_logger().info(f'  No valid person detected')
                continue
            
            if best_distance > max_kpt_distance:
                self.get_logger().info(
                    f'  Best person too far from LiDAR: {best_distance:.0f}px')
                continue
            
            self.get_logger().info(
                f'  Selected person {best_person_idx} '
                f'(dist from LiDAR: {best_distance:.0f}px)')
            
            keypoints = all_keypoints[best_person_idx]

            # Verification
            verified, reason, mode, confidence = self.verify_person(keypoints)
            
            self.get_logger().info(
                f'  Mode: {mode}, Confidence: {confidence:.2f}')

            # Draw on debug image
            self.draw_debug(debug_image, u_min, v_min, u_max, v_max,
                            keypoints, verified, reason)

            if verified:
                # Add to verified list (keep LiDAR position)
                verified_poses.append(pose)
                self.get_logger().info(f'  VERIFIED: {reason}')
            else:
                self.get_logger().info(f'  NOT verified: {reason}')

        # Publish verified
        if len(verified_poses) > 0:
            verified_msg = PoseArray()
            verified_msg.header = msg.header
            verified_msg.poses = verified_poses
            self.pub_verified.publish(verified_msg)

        # Publish debug image
        # Publish debug image
        try:
            self.get_logger().info(f'>>> Publishing debug image')
            img_uint8 = debug_image.astype(np.uint8)
            debug_msg = Image()
            debug_msg.header.stamp = msg.header.stamp
            debug_msg.header.frame_id = self.camera_frame
            debug_msg.height = img_uint8.shape[0]
            debug_msg.width = img_uint8.shape[1]
            debug_msg.encoding = 'bgr8'
            debug_msg.is_bigendian = 0
            debug_msg.step = img_uint8.shape[1] * 3
            debug_msg.data = img_uint8.tobytes()
            self.pub_debug.publish(debug_msg)
            self.get_logger().info(f'<<< Debug published')
        except Exception as e:
            import traceback
            self.get_logger().error(
                f'Debug image publish failed: {e}\n{traceback.format_exc()}')

    # ==========================
    # Verification
    # ==========================
    def verify_person(self, keypoints) -> tuple:
        """Check keypoint groups for verification.
        Returns (verified: bool, reason: str, mode: str, confidence: float)
        """
        # Group A: lower body (knees, ankles)
        group_a_detected = []
        group_a_conf = []
        for idx in GROUP_A_LOWER:
            if keypoints[idx, 2] > self.conf_thresh:
                group_a_detected.append(idx)
                group_a_conf.append(keypoints[idx, 2])

        # Group B: torso (hips)
        group_b_detected = []
        group_b_conf = []
        for idx in GROUP_B_TORSO:
            if keypoints[idx, 2] > self.conf_thresh:
                group_b_detected.append(idx)
                group_b_conf.append(keypoints[idx, 2])

        a_count = len(group_a_detected)
        b_count = len(group_b_detected)
        a_conf_avg = np.mean(group_a_conf) if a_count > 0 else 0.0
        b_conf_avg = np.mean(group_b_conf) if b_count > 0 else 0.0

        # Determine mode
        if a_count >= self.group_a_req and b_count >= self.group_b_req:
            mode = 'full_body'
            confidence = (a_conf_avg + b_conf_avg) / 2
            verified = True
            reason = (f'{mode}: A={a_count}/{self.group_a_req} '
                       f'B={b_count}/{self.group_b_req}')
        elif a_count >= self.group_a_req:
            mode = 'lower_body_only'
            confidence = a_conf_avg
            verified = True
            reason = f'{mode}: A={a_count}/{self.group_a_req}'
        elif b_count >= self.group_b_req:
            mode = 'torso_only'
            confidence = b_conf_avg
            verified = True
            reason = f'{mode}: B={b_count}/{self.group_b_req}'
        else:
            mode = 'not_verified'
            confidence = 0.0
            verified = False
            reason = (f'{mode}: A={a_count}/{self.group_a_req} '
                       f'B={b_count}/{self.group_b_req}')

        return verified, reason, mode, confidence

    # ==========================
    # Utils
    # ==========================
    def quat_to_matrix(self, x, y, z, w) -> np.ndarray:
        """Quaternion to rotation matrix."""
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    def draw_debug(self, image, u_min, v_min, u_max, v_max,
                    keypoints, verified, reason):
        """Draw crop region + keypoints on debug image."""
        color = (0, 255, 0) if verified else (0, 0, 255)
        cv2.rectangle(image, (u_min, v_min), (u_max, v_max), color, 2)
        cv2.putText(image, reason, (u_min, v_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints (offset by crop origin)
        for i in range(17):
            x, y, conf = keypoints[i]
            if conf > self.conf_thresh:
                px = int(x) + u_min
                py = int(y) + v_min
                # Color by group
                if i in GROUP_A_LOWER:
                    kp_color = (0, 255, 255)  # yellow for lower
                elif i in GROUP_B_TORSO:
                    kp_color = (255, 0, 255)  # magenta for torso
                else:
                    kp_color = (255, 255, 255)  # white
                cv2.circle(image, (px, py), 4, kp_color, -1)


def main():
    rclpy.init()
    node = CameraKeypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()