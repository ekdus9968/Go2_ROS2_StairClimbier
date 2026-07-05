#!/usr/bin/env python3
"""
stair_detect.py

First-pass stair detector. Uses LiDAR, depth camera, and YOLOv11n.
All three have to agree before /stair_detected goes True.

YOLO spots stairs visually, LiDAR checks for z-jumps in that region,
depth camera checks for edges in the same spot.

*** lidar_topic defaults to /lidar/points but this is probably wrong. ***
*** Run: ros2 topic list | grep -i lidar  then set it in your launch file. ***

If YOLO isn't ready yet, set use_yolo_gate:=false. LiDAR will scan
the full front cone instead, which has more false positives.

Subscribes:
  lidar_topic (param)    sensor_msgs/PointCloud2
  depth_topic (param)    sensor_msgs/Image
  yolo_topic (param)     vision_msgs/Detection2DArray

Publishes:
  /stair_detected        std_msgs/Bool
  /stair_distance        std_msgs/Float32  (meters, -1 if not detected)
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge

try:
    from vision_msgs.msg import Detection2DArray
    VISION_MSGS_AVAILABLE = True
except ImportError:
    VISION_MSGS_AVAILABLE = False


class StairDetectorNode(Node):

    def __init__(self):
        super().__init__('stair_detector')

        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('detection_range_m', 2.5)
        self.declare_parameter('step_height_min_m', 0.04)
        self.declare_parameter('step_height_max_m', 0.20)
        self.declare_parameter('min_steps_for_stairs', 2)
        self.declare_parameter('use_yolo_gate', True)
        self.declare_parameter('yolo_confidence_threshold', 0.4)
        self.declare_parameter('yolo_stair_class_id', 0)
        self.declare_parameter('yolo_gate_timeout_sec', 1.0)

        # *** CHANGE THIS to match your actual LiDAR topic ***
        # find it with: ros2 topic list | grep -i lidar
        self.declare_parameter('lidar_topic', '/lidar/points')
        self.declare_parameter('depth_topic', '/depth_camera/depth/image_raw')
        self.declare_parameter('yolo_topic', '/yolo/stair_detections')

        # approximate Go2 front camera values, update from calibration if you have it
        self.declare_parameter('camera_fx', 554.0)
        self.declare_parameter('camera_fy', 554.0)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        rate = self.get_parameter('publish_rate_hz').value
        self.det_range = self.get_parameter('detection_range_m').value
        self.h_min = self.get_parameter('step_height_min_m').value
        self.h_max = self.get_parameter('step_height_max_m').value
        self.min_steps = self.get_parameter('min_steps_for_stairs').value
        self.use_yolo_gate = self.get_parameter('use_yolo_gate').value
        self.yolo_conf_thresh = self.get_parameter('yolo_confidence_threshold').value
        self.yolo_class_id = self.get_parameter('yolo_stair_class_id').value
        self.yolo_timeout = self.get_parameter('yolo_gate_timeout_sec').value

        lidar_topic = self.get_parameter('lidar_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        yolo_topic = self.get_parameter('yolo_topic').value

        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value
        self.img_w = self.get_parameter('image_width').value
        self.img_h = self.get_parameter('image_height').value

        self.bridge = CvBridge()

        self.lidar_detected = False
        self.lidar_distance = float('nan')
        self.camera_detected = False
        self.camera_distance = float('nan')

        # last YOLO bbox in normalized 0-1 coords, None if no recent detection
        self.yolo_bbox_norm = None
        self.yolo_last_seen = 0.0
        self.yolo_active = False

        self.create_subscription(PointCloud2, lidar_topic, self.lidar_cb, 10)
        self.create_subscription(Image, depth_topic, self.depth_cb, 10)

        if self.use_yolo_gate:
            if VISION_MSGS_AVAILABLE:
                self.create_subscription(
                    Detection2DArray, yolo_topic, self.yolo_cb, 10
                )
                self.get_logger().info(f'YOLO gate enabled on {yolo_topic}')
            else:
                self.get_logger().warn(
                    'vision_msgs not installed, YOLO gate disabled. '
                    'Install with: sudo apt install ros-humble-vision-msgs'
                )
                self.use_yolo_gate = False

        self.pub_detected = self.create_publisher(Bool, '/stair_detected', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair_distance', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairDetectorNode ready. '
            f'LiDAR={lidar_topic}, depth={depth_topic}, '
            f'range={self.det_range}m, '
            f'step_h=[{self.h_min},{self.h_max}]m, '
            f'min_steps={self.min_steps}, '
            f'yolo_gate={self.use_yolo_gate}'
        )

    def yolo_cb(self, msg: 'Detection2DArray'):
        # pick the highest-confidence stair detection and store its bbox
        best_conf = 0.0
        best_bbox = None

        for det in msg.detections:
            for hyp in det.results:
                if int(hyp.hypothesis.class_id) != self.yolo_class_id:
                    continue
                if hyp.hypothesis.score < self.yolo_conf_thresh:
                    continue
                if hyp.hypothesis.score > best_conf:
                    best_conf = hyp.hypothesis.score
                    cx = det.bbox.center.position.x
                    cy = det.bbox.center.position.y
                    bw = det.bbox.size_x
                    bh = det.bbox.size_y
                    best_bbox = (
                        (cx - bw / 2.0) / self.img_w,
                        (cy - bh / 2.0) / self.img_h,
                        (cx + bw / 2.0) / self.img_w,
                        (cy + bh / 2.0) / self.img_h,
                    )

        if best_bbox is not None:
            self.yolo_bbox_norm = best_bbox
            self.yolo_last_seen = self.get_clock().now().nanoseconds / 1e9
            self.yolo_active = True
        else:
            # clear the gate if nothing seen recently
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self.yolo_last_seen > self.yolo_timeout:
                self.yolo_active = False
                self.yolo_bbox_norm = None

    def _yolo_to_lidar_lateral_bounds(self, bbox_norm, distance_m):
        # convert bbox left/right edges to robot-frame y bounds at a given depth
        x_min_norm, _, x_max_norm, _ = bbox_norm
        x_min_px = x_min_norm * self.img_w
        x_max_px = x_max_norm * self.img_w
        y_left = -(x_min_px - self.cx) * distance_m / self.fx
        y_right = -(x_max_px - self.cx) * distance_m / self.fx
        return min(y_left, y_right), max(y_left, y_right)

    def lidar_cb(self, msg: PointCloud2):
        # slice points into 10cm bins along x, check median z per bin for step-height jumps
        # when YOLO gate is on, only look at points inside the flagged region
        try:
            points = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True
            )
        except Exception as e:
            self.get_logger().debug(f'LiDAR read failed: {e}')
            self.lidar_detected = False
            return

        if points.shape[0] < 100:
            self.lidar_detected = False
            return

        if self.use_yolo_gate and self.yolo_active and self.yolo_bbox_norm is not None:
            mid_depth = self.det_range / 2.0
            y_min, y_max = self._yolo_to_lidar_lateral_bounds(
                self.yolo_bbox_norm, mid_depth
            )
            y_min = max(y_min, -1.0)
            y_max = min(y_max, 1.0)
        else:
            if self.use_yolo_gate and not self.yolo_active:
                self.lidar_detected = False
                return
            y_min, y_max = -0.5, 0.5

        mask = (
            (points[:, 0] > 0.3) &
            (points[:, 0] < self.det_range) &
            (points[:, 1] > y_min) &
            (points[:, 1] < y_max)
        )
        front = points[mask]

        if front.shape[0] < 50:
            self.lidar_detected = False
            return

        bins = np.arange(0.3, self.det_range, 0.1)
        z_per_bin = []
        for i in range(len(bins) - 1):
            in_bin = (front[:, 0] >= bins[i]) & (front[:, 0] < bins[i + 1])
            if in_bin.sum() > 5:
                z_per_bin.append(float(np.median(front[in_bin, 2])))
            else:
                z_per_bin.append(float('nan'))

        z_arr = np.array(z_per_bin)
        valid = ~np.isnan(z_arr)

        if valid.sum() < self.min_steps + 2:
            self.lidar_detected = False
            return

        z_valid = z_arr[valid]
        jumps = np.diff(z_valid)
        step_jumps = (jumps > self.h_min) & (jumps < self.h_max)
        num_steps = int(step_jumps.sum())

        if num_steps >= self.min_steps:
            self.lidar_detected = True
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            valid_centers = bin_centers[valid][:-1]
            step_positions = valid_centers[step_jumps]
            self.lidar_distance = (
                float(step_positions.min()) if len(step_positions) > 0
                else float('nan')
            )
        else:
            self.lidar_detected = False
            self.lidar_distance = float('nan')

    def depth_cb(self, msg: Image):
        # stair edges show up as sharp depth jumps along a vertical column
        # use YOLO bbox region if available, otherwise default to image center
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().debug(f'Depth image conversion failed: {e}')
            self.camera_detected = False
            return

        h, w = depth.shape

        if self.use_yolo_gate and self.yolo_active and self.yolo_bbox_norm is not None:
            x_min_norm, y_min_norm, x_max_norm, y_max_norm = self.yolo_bbox_norm
            px_left = int(np.clip(x_min_norm * w, 0, w - 1))
            px_right = int(np.clip(x_max_norm * w, 0, w - 1))
            py_top = int(np.clip(y_min_norm * h, 0, h - 1))
            py_bot = int(np.clip(y_max_norm * h, 0, h - 1))
            cx = (px_left + px_right) // 2
            col_top = py_top + (py_bot - py_top) // 2
            col = depth[col_top:py_bot, max(0, cx - 5):min(w, cx + 5)]
        else:
            if self.use_yolo_gate and not self.yolo_active:
                self.camera_detected = False
                return
            col = depth[h // 3:, w // 2 - 5: w // 2 + 5]

        col_median = np.nanmedian(col, axis=1)
        valid = (
            ~np.isnan(col_median) &
            (col_median > 0.3) &
            (col_median < self.det_range)
        )
        col_valid = col_median[valid]

        if len(col_valid) < 20:
            self.camera_detected = False
            return

        diffs = np.diff(col_valid)
        edges = np.abs(diffs) > 0.05
        num_edges = int(edges.sum())

        if num_edges >= self.min_steps * 2:
            self.camera_detected = True
            self.camera_distance = float(np.min(col_valid))
        else:
            self.camera_detected = False
            self.camera_distance = float('nan')

    def publish_result(self):
        # LiDAR and depth both have to agree; YOLO is implicit since both
        # callbacks return early when yolo_active is False
        detected = self.lidar_detected and self.camera_detected

        if detected and not math.isnan(self.lidar_distance) and not math.isnan(self.camera_distance):
            distance = (self.lidar_distance + self.camera_distance) / 2.0
        elif detected and not math.isnan(self.lidar_distance):
            distance = self.lidar_distance
        else:
            distance = float('nan')

        self.pub_detected.publish(Bool(data=detected))

        d_msg = Float32()
        d_msg.data = distance if not math.isnan(distance) else -1.0
        self.pub_distance.publish(d_msg)

        if detected:
            self.get_logger().info(
                f'Stair detected at {distance:.2f}m '
                f'(lidar={self.lidar_distance:.2f}m, '
                f'depth={self.camera_distance:.2f}m, '
                f'yolo_active={self.yolo_active})',
                throttle_duration_sec=1.0
            )


def main(args=None):
    rclpy.init(args=args)
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
