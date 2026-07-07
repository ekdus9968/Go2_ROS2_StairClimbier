

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class StairConsensus(Node):

    def __init__(self):
        super().__init__('stair_consensus')

        self.declare_parameter('fusion_detected_topic', '/fusion/stair_detected')
        self.declare_parameter('fusion_distance_topic', '/fusion/stair_distance')
        self.declare_parameter('yolo_detected_topic', '/stair/yolo_detected')
        self.declare_parameter('publish_rate_hz', 5.0)

        fusion_det_topic = self.get_parameter('fusion_detected_topic').value
        fusion_dist_topic = self.get_parameter('fusion_distance_topic').value
        yolo_det_topic = self.get_parameter('yolo_detected_topic').value
        rate = self.get_parameter('publish_rate_hz').value

        self.fusion_detected = False
        self.fusion_distance = float('nan')
        self.yolo_detected = False

        self.create_subscription(Bool, fusion_det_topic, self.fusion_det_cb, 10)
        self.create_subscription(Float32, fusion_dist_topic, self.fusion_dist_cb, 10)
        self.create_subscription(Bool, yolo_det_topic, self.yolo_det_cb, 10)

        self.pub_mode = self.create_publisher(String, '/stair/robot_mode', 10)
        self.pub_distance = self.create_publisher(Float32, '/stair_distance', 10)

        self.create_timer(1.0 / rate, self.publish_result)

        self.get_logger().info(
            f'StairConsensus ready. '
            f'fusion={fusion_det_topic}, yolo={yolo_det_topic}'
        )

    def fusion_det_cb(self, msg: Bool):
        self.fusion_detected = msg.data

    def fusion_dist_cb(self, msg: Float32):
        self.fusion_distance = float(msg.data) if msg.data >= 0 else float('nan')

    def yolo_det_cb(self, msg: Bool):
        self.yolo_detected = msg.data

    def publish_result(self):
        agree_count = int(self.fusion_detected) + int(self.yolo_detected)

        if agree_count == 2:
            mode = 'STAIR'
        elif agree_count == 1:
            mode = 'APPROACH'
        else:
            mode = 'NONE'

        distance = self.fusion_distance

        self.pub_mode.publish(String(data=mode))
        self.pub_distance.publish(Float32(
            data=distance if not math.isnan(distance) else -1.0
        ))

        if mode != 'NONE':
            self.get_logger().info(
                f'{mode} (fusion={self.fusion_detected}, yolo={self.yolo_detected}, '
                f'dist={distance if not math.isnan(distance) else -1:.2f}m)',
                throttle_duration_sec=1.0
            )
        else:
            self.get_logger().debug(
                f'NONE (fusion={self.fusion_detected}, yolo={self.yolo_detected})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = StairConsensus()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()