#!/usr/bin/env python3
"""
Person detector using YOLOv8 and Go2's D435i camera (sim).

Subscribes:
  /d435i/d435i/image_raw          sensor_msgs/Image
  /d435i/d435i/depth/image_raw    sensor_msgs/Image  (optional, for accurate depth)

Publishes:
  person_follow/target            geometry_msgs/PoseStamped  (in base_link frame)
  person_follow/target_valid      std_msgs/Bool
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# D435i sim camera intrinsics — Go2 with payload xacro defines 640x480 with HFOV=1.5708
# fx = (width/2) / tan(hfov/2) = 320 / tan(0.7854) = 320
CAMERA_FX = 320.0
CAMERA_FY = 320.0
CAMERA_CX = 320.0
CAMERA_CY = 240.0

# Marionette patient is ~1.7m tall (pelvis_h=0.2 + thigh+shank+foot ~0.8 + headless)
ASSUMED_PERSON_HEIGHT_M = 1.6


def estimate_distance(bbox_height_px: float) -> float:
    if bbox_height_px <= 0:
        return 999.0
    return (ASSUMED_PERSON_HEIGHT_M * CAMERA_FY) / bbox_height_px


def bbox_center_to_xy(cx_px: float, distance_m: float):
    """Convert pixel cx to (x, y) in robot base frame."""
    x = distance_m
    y = -(cx_px - CAMERA_CX) * distance_m / CAMERA_FX
    return x, y


class PersonDetectorNode(Node):

    def __init__(self):
        super().__init__('person_detector')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('image_topic', '/d435i/d435i/image_raw')

        model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        image_topic = self.get_parameter('image_topic').value

        if YOLO_AVAILABLE:
            self.get_logger().info(f'Loading YOLOv8 from: {model_path}')
            self.model = YOLO(model_path)
            self.detect_fn = self._detect_yolo
        else:
            self.get_logger().warn('ultralytics not found - using HOG fallback')
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.detect_fn = self._detect_hog

        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.pub_target = self.create_publisher(
            PoseStamped, 'person_follow/target', 10
        )

        self.pub_valid = self.create_publisher(
            Bool, 'person_follow/target_valid', 10
        )

        self.get_logger().info(f'PersonDetector subscribed to: {image_topic}')

    def _detect_yolo(self, frame):
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
                    best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, y2 - y1)
        return best

    def _detect_hog(self, frame):
        gray = cv2.resize(frame, (320, 240))
        rects, _ = self.hog.detectMultiScale(gray, winStride=(8, 8))
        if len(rects) == 0:
            return None
        rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = rects[0]
        sx = frame.shape[1] / 320.0
        sy = frame.shape[0] / 240.0
        return (x + w / 2.0) * sx, (y + h / 2.0) * sy, h * sy

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        detection = self.detect_fn(frame)
        valid_msg = Bool()

        if detection is None:
            valid_msg.data = False
            self.pub_valid.publish(valid_msg)
            return

        cx_px, cy_px, bbox_h_px = detection
        distance_m = estimate_distance(bbox_h_px)
        x_m, y_m = bbox_center_to_xy(cx_px, distance_m)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = x_m
        pose_msg.pose.position.y = y_m
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pub_target.publish(pose_msg)

        valid_msg.data = True
        self.pub_valid.publish(valid_msg)

        self.get_logger().info(
            f'Person at x={x_m:.2f}m y={y_m:.2f}m (dist={distance_m:.2f}m)',
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
