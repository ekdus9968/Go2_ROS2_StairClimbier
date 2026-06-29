#!/usr/bin/env python3
"""Bridge /odom topic to TF broadcast (odom -> base)."""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('parent_frame', 'odom')
        self.declare_parameter('child_frame', 'base')
        
        topic = self.get_parameter('odom_topic').value
        self.parent = self.get_parameter('parent_frame').value
        self.child = self.get_parameter('child_frame').value
        
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, topic, self.cb, 10)
        self.get_logger().info(f'OdomToTF: {topic} -> {self.parent}/{self.child}')
    
    def cb(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent
        t.child_frame_id = self.child
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    rclpy.spin(OdomToTF())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
