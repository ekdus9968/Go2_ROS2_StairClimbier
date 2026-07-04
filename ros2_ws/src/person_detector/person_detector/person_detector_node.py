#!/usr/bin/env python3
"""
Person detector using YOLOv11n and Go2's D435i (RGB + depth).

- YOLOv11n COCO person class (no custom training)
- Depth camera for accurate distance measurement
- Follow distance: oxygen tank bottom (0.3775m) ↔ patient head top (1.65m)
  diagonal 2.0m → XY plane target ~1.543m

Subscribes:
  /d435i/d435i/image_raw          sensor_msgs/Image
  /d435i/d435i/depth/image_raw    sensor_msgs/Image

Publishes:
  person_follow/target            geometry_msgs/PoseStamped  (base_link frame)
  person_follow/target_valid      std_msgs/Bool
"""

import math
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# D435i sim camera intrinsics (from URDF: 640x480, HFOV=1.5708)
CAMERA_FX = 320.0
CAMERA_FY = 320.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0

# Camera position in base_link frame (from URDF)
CAM_OFFSET_X = 0.1423
CAM_OFFSET_Y = 0.0
CAM_OFFSET_Z = 0.2035


class PersonDetectorNode(Node):

    def __init__(self):
        super().__init__('person_detector')

        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('rgb_topic', '/d435i/d435i/image_raw')
        self.declare_parameter('depth_topic', '/d435i/d435i/depth/image_raw')
        self.declare_parameter('use_depth', True)

        model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.use_depth = self.get_parameter('use_depth').value

        if not YOLO_AVAILABLE:
            self.get_logger().fatal('ultralytics not installed. Run: pip3 install ultralytics')
            raise ImportError('ultralytics required')

        self.get_logger().info(f'Loading YOLO from: {model_path}')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        if self.use_depth:
            self.rgb_sub = Subscriber(self, Image, rgb_topic)
            self.depth_sub = Subscriber(self, Image, depth_topic)
            self.sync = ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], queue_size=5, slop=0.1
            )
            self.sync.registerCallback(self.rgb_depth_callback)
            self.get_logger().info(f'Sync: {rgb_topic} + {depth_topic}')
        else:
            self.create_subscription(Image, rgb_topic, self.rgb_only_callback, 10)
            self.get_logger().info(f'RGB only: {rgb_topic}')

        self.pub_target = self.create_publisher(PoseStamped, 'person_follow/target', 10)
        self.pub_valid = self.create_publisher(Bool, 'person_follow/target_valid', 10)

    def detect_person(self, frame):
        """Returns (cx_px, cy_px, y2_bottom_px, bbox_height_px) or None."""
        results = self.model(frame, classes=[0], verbose=False)
        best = None
        best_area = 0.0
        for r in results:
            for box in r.boxes:
                if float(box.conf) < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, y2, y2 - y1)
        return best

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        det = self.detect_person(frame)
        if det is None:
            self.pub_valid.publish(Bool(data=False))
            return

        cx_px, cy_px, _, _ = det

        # Sample depth in a small region around bbox center (avoid noise)
        h, w = depth.shape
        cx_int, cy_int = int(cx_px), int(cy_px)
        if 0 <= cx_int < w and 0 <= cy_int < h:
            patch = depth[max(0, cy_int-5):min(h, cy_int+5),
                          max(0, cx_int-5):min(w, cx_int+5)]
            valid = patch[np.isfinite(patch) & (patch > 0.3) & (patch < 8.0)]
            if len(valid) == 0:
                self.pub_valid.publish(Bool(data=False))
                return
            distance_m = float(np.median(valid))
        else:
            self.pub_valid.publish(Bool(data=False))
            return

        self._publish_target(cx_px, cy_px, distance_m)

    def rgb_only_callback(self, rgb_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge failed: {e}')
            return

        det = self.detect_person(frame)
        if det is None:
            self.pub_valid.publish(Bool(data=False))
            return

        cx_px, cy_px, _, bbox_h = det
        # Fallback: bbox height based estimation (1.6m assumed height)
        distance_m = (1.6 * CAMERA_FY) / bbox_h if bbox_h > 0 else 999.0
        self._publish_target(cx_px, cy_px, distance_m)

    def _publish_target(self, cx_px, cy_px, distance_m):
        """Convert pixel + depth → base_link frame."""
        # Pixel to camera frame (z-forward, x-right, y-down in image convention)
        x_cam = (cx_px - CAMERA_CX) * distance_m / CAMERA_FX
        y_cam = (cy_px - CAMERA_CY) * distance_m / CAMERA_FY
        z_cam = distance_m

        # Camera → base_link (D435i mounted forward-facing on Go2)
        # Camera z-axis = base_link +x, camera x-axis = base_link -y, camera y-axis = base_link -z
        x_base = z_cam + CAM_OFFSET_X
        y_base = -x_cam + CAM_OFFSET_Y
        z_base = -y_cam + CAM_OFFSET_Z

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x_base
        pose.pose.position.y = y_base
        pose.pose.position.z = z_base
        pose.pose.orientation.w = 1.0
        self.pub_target.publish(pose)

        self.pub_valid.publish(Bool(data=True))

        self.get_logger().info(
            f'Person: x={x_base:.2f}m y={y_base:.2f}m z={z_base:.2f}m dist={distance_m:.2f}m',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
