#!/usr/bin/env python3
"""
Keyboard Teleop for cmd_vel
W/S: forward/backward
A/D: left/right
Q/E: yaw left/right
Space: stop
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cmd = Twist()
        
        # Speed limits
        self.vx_max = 0.5
        self.vy_max = 0.3
        self.vyaw_max = 1.0
        
        self.get_logger().info('''
Teleop Keyboard:
W/S: forward/backward
A/D: strafe left/right
Q/E: yaw left/right
Space: stop
Ctrl+C: quit
''')
    
    def get_key(self, timeout=0.1):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def run(self):
        while rclpy.ok():
            key = self.get_key()
            
            if key == 'w': self.cmd.linear.x = self.vx_max
            elif key == 's': self.cmd.linear.x = -self.vx_max
            elif key == 'a': self.cmd.linear.y = self.vy_max
            elif key == 'd': self.cmd.linear.y = -self.vy_max
            elif key == 'q': self.cmd.angular.z = self.vyaw_max
            elif key == 'e': self.cmd.angular.z = -self.vyaw_max
            elif key == ' ':
                self.cmd = Twist()
            elif key == '\x03':  # Ctrl+C
                break
            
            self.pub.publish(self.cmd)


def main():
    rclpy.init()
    node = TeleopNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
