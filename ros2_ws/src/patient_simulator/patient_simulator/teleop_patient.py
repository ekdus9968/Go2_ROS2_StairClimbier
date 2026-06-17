#!/usr/bin/env python3
"""환자 키보드 조작.

키 매핑:
    I: Forward
    K: Backward
    J: Turn Left
    L: Turn Right
    U: Strafe Left
    O: Strafe Right
    Space: Stop
    Ctrl+C: Quit
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select


SPEED_LINEAR = 0.5   # m/s
SPEED_ANGULAR = 0.5  # rad/s


class PatientTeleop(Node):
    def __init__(self):
        super().__init__('patient_teleop')
        self.pub = self.create_publisher(Twist, '/patient/cmd_vel', 10)
        self.cmd = Twist()
        
        self.get_logger().info("""
        ==================================
        Patient Keyboard Teleop
        ==================================
        I: Forward
        K: Backward
        J: Turn Left
        L: Turn Right
        U: Strafe Left
        O: Strafe Right
        Space: Stop
        Ctrl+C: Quit
        ==================================
        """)
    
    def get_key(self, timeout=0.1):
        """block key."""
        settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                key = sys.stdin.read(1)
            else:
                key = ''
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key
    
    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()
                
                if key == 'i':
                    self.cmd.linear.x = SPEED_LINEAR
                elif key == 'k':
                    self.cmd.linear.x = -SPEED_LINEAR
                elif key == 'j':
                    self.cmd.angular.z = SPEED_ANGULAR
                elif key == 'l':
                    self.cmd.angular.z = -SPEED_ANGULAR
                elif key == 'u':
                    self.cmd.linear.y = SPEED_LINEAR
                elif key == 'o':
                    self.cmd.linear.y = -SPEED_LINEAR
                elif key == ' ':
                    self.cmd = Twist()  # 정지
                elif key == '\x03':  # Ctrl+C
                    break
                
                self.pub.publish(self.cmd)
        finally:
            # 종료 시 정지
            self.pub.publish(Twist())


def main():
    rclpy.init()
    node = PatientTeleop()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()