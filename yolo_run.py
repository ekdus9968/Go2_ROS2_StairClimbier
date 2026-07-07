#!/usr/bin/env python3
"""
yolo_stair_detector.py

Custom-trained YOLO11 stair detector, run this on its own to test before
fusion. Matches the shape of lidar_stair_detector.py / camera_stair_detector.py
so it can slot into the same fusion pattern later.

Runs your custom-trained .pt model against the RGB camera feed. Publishes a
distance topic for structural consistency with the other two detectors, but
the model has no real depth information, so it always publishes -1.0 there
(same convention the other files already use for "no valid distance").

Run to test:
  ros2 run stair_detector yolo_stair_detector
  ros2 topic echo /stair/yolo_detected

Publishes:
  /stair/yolo_detected    std_msgs/Bool
  /stair/yolo_distance    std_msgs/Float32  (always -1.0, no real depth info)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloStairDetector(Node):

    def __init__(self):
        super().__init__('yolo_stair_detector')

        # confirmed from go2_with_payload.urdf.xacro's d435i_sensor gazebo
        # plugin (libgazebo_ros_camera, namespace=/d435i, camera_name=d435i,
        # no topic overrides -> default gazebo naming applies)
        self.declare_parameter('image_topic', '/d435i/d435i/image_raw')
        self.declare_parameter('model_path', '/path/to/your_weights.pt')
        self.declare_parameter('publish_rate_hz', 5.0)

        # *** CHANGE THIS: verify with model.names after loading -
        # single-class models are almost always 0, but don't assume ***
        self.declare_parameter('yolo_class_id', 0)
        self.declare_parameter('yolo_confidence_threshold', 0.4)

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model_path').value
        rate = self.get_parameter('publish_rate_hz').value
        self.yolo_class_id = self.get_parameter('yolo_class_id').value
        self.yolo_conf_thresh = self.get_parameter('yolo_confidence_threshold').value

        self.bridge = CvBridge()

        self.get_logger().info(f'Loading YOLO model from {model_path}')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model classes: {self.model.names}')

        self.detected = False
        self.distance = -1.0  # always -1.0, matches "no valid distance" convention

        self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.pub_detected = self.create_publisher(Bool, '/stair/yolo_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair/yolo_distance', 10)
        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'YoloStairDetector ready on {image_topic}, '
            f'class_id={self.yolo_class_id}, conf_thresh={self.yolo_conf_thresh}'
        )

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().debug(f'image conversion failed: {e}')
            self.detected = False
            return

        results = self.model.predict(frame, verbose=False)

        best_conf = 0.0
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != self.yolo_class_id:
                    continue
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf

        if best_conf >= self.yolo_conf_thresh:
            self.detected = True
        else:
            self.detected = False

    def publish_result(self):
        self.pub_detected.publish(Bool(data=self.detected))
        self.pub_distance.publish(Float32(data=self.distance))

        if self.detected:
            self.get_logger().info(
                'YOLO: stairs detected',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = YoloStairDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()