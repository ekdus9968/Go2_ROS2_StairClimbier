#!/usr/bin/env python3
"""
person_detector_node.py

Subscribes to the Go2's front camera feed, runs person detection,
and publishes the detected person's estimated position relative to
the robot as a PoseStamped on person_follow/target.

Also publishes person_follow/target_valid (Bool) so go2_nav_bridge
knows whether to move or stop.

Detection: uses YOLOv8n (nano) fast enough for onboard CPU.
Falls back to a simple HOG detector if ultralytics isn't installed.

Topics published:
  person_follow/target       geometry_msgs/PoseStamped
  person_follow/target_valid std_msgs/Bool

Topics subscribed:
  /camera/image_raw          sensor_msgs/Image
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

# Try YOLOv8 first, fall back to HOG
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Camera intrinsics for Go2 front camera
# These are approximate replace with values from your calibration if you have them.
# Go2 front camera is roughly 640x480 with ~90 degree HFOV.
CAMERA_FX = 554.0   # focal length x (pixels)
CAMERA_FY = 554.0   # focal length y (pixels)
CAMERA_CX = 320.0   # principal point x
CAMERA_CY = 240.0   # principal point y
ASSUMED_PERSON_HEIGHT_M = 1.7   # meters used to estimate depth from bbox height

def estimate_distance(bbox_height_px: float) -> float:
    """
    Estimate distance to person using the pinhole camera model.
    distance = (real_height * focal_length) / bbox_height_pixels
    """
    if bbox_height_px <= 0:
        return 999.0
    return (ASSUMED_PERSON_HEIGHT_M * CAMERA_FY) / bbox_height_px

def bbox_center_to_xy(cx_px: float, cy_px: float, distance_m: float):
    """
    Convert pixel center of bounding box to (x, y) in robot base frame.
    x = forward (depth), y = left/right
    """
    x = distance_m
    y = -(cx_px - CAMERA_CX) * distance_m / CAMERA_FX  # negative = right of center is +y left
    return x, y

class PersonDetectorNode(Node):

    def __init__(self):
        super().__init__('person_detector')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('publish_rate_hz', 10.0)

        model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        image_topic = self.get_parameter('image_topic').value

        # Detector setup
        if YOLO_AVAILABLE:
            self.get_logger().info(f'Loading YOLOv8 model from: {model_path}')
            self.model = YOLO(model_path)
            self.detect_fn = self._detect_yolo
        else:
            self.get_logger().warn('ultralytics not found falling back to HOG detector. '
                                   'Install with: pip install ultralytics')
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.detect_fn = self._detect_hog

        # ROS2 setup
        self.bridge = CvBridge()

        self.sub_image = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.pub_target = self.create_publisher(
            PoseStamped,
            'person_follow/target',
            10
        )

        self.pub_valid = self.create_publisher(
            Bool,
            'person_follow/target_valid',
            10
        )

        self.get_logger().info('PersonDetectorNode started.')
        self.get_logger().info(f'  Subscribed to: {image_topic}')
        self.get_logger().info(f'  Publishing to: person_follow/target, person_follow/target_valid')

    #
    # Detection backends
    #
    def _detect_yolo(self, frame: np.ndarray):
        """
        Run YOLOv8 inference. Returns (cx, cy, bbox_h) of the largest
        person detection, or None if no person found.
        """
        results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
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
                    best = (
                        (x1 + x2) / 2.0,   # cx
                        (y1 + y2) / 2.0,   # cy
                        y2 - y1             # bbox height
                    )
        return best

    def _detect_hog(self, frame: np.ndarray):
        """
        Run HOG+SVM person detection. Returns (cx, cy, bbox_h) of the
        largest detection, or None.
        """
        gray = cv2.resize(frame, (320, 240))
        rects, _ = self.hog.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        if len(rects) == 0:
            return None

        # Pick largest rect (closest person)
        rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
        x, y, w, h = rects[0]

        # HOG ran on 320x240, scale back to 640x480
        scale_x = frame.shape[1] / 320.0
        scale_y = frame.shape[0] / 240.0
        cx = (x + w / 2.0) * scale_x
        cy = (y + h / 2.0) * scale_y
        bbox_h = h * scale_y

        return cx, cy, bbox_h

    #
    # Image callback
    #
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        detection = self.detect_fn(frame)
        valid_msg = Bool()

        if detection is None:
            # No person found publish invalid
            valid_msg.data = False
            self.pub_valid.publish(valid_msg)
            return

        cx_px, cy_px, bbox_h_px = detection
        distance_m = estimate_distance(bbox_h_px)
        x_m, y_m = bbox_center_to_xy(cx_px, cy_px, distance_m)

        # Publish target pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'base_link'
        pose_msg.pose.position.x = x_m
        pose_msg.pose.position.y = y_m
        pose_msg.pose.position.z = 0.0
        # Orientation: face toward the person
        pose_msg.pose.orientation.w = 1.0

        self.pub_target.publish(pose_msg)

        valid_msg.data = True
        self.pub_valid.publish(valid_msg)

        self.get_logger().debug(
            f'Person detected at x={x_m:.2f}m y={y_m:.2f}m (dist={distance_m:.2f}m)'
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
