#!/usr/bin/env python3
"""MPPI-based follow controller via Nav2 FollowPath action."""
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from nav2_msgs.action import FollowPath
from go2_interfaces.msg import RobotMode


class FollowControllerNode(Node):

    def __init__(self):
        super().__init__('follow_controller')
        self.declare_parameter('follow_distance_m', 1.543)
        self.declare_parameter('follow_tolerance_m', 0.2)
        self.declare_parameter('max_goal_rate_hz', 2.0)
        self.declare_parameter('goal_position_threshold_m', 0.1)
        self.declare_parameter('ema_alpha', 0.6)
        self.declare_parameter('target_timeout_sec', 1.0)

        self.follow_dist = self.get_parameter('follow_distance_m').value
        self.follow_tol = self.get_parameter('follow_tolerance_m').value
        self.max_goal_rate = self.get_parameter('max_goal_rate_hz').value
        self.pos_thresh = self.get_parameter('goal_position_threshold_m').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.target_timeout = self.get_parameter('target_timeout_sec').value

        self.current_odom = None
        self.mode = RobotMode.IDLE
        self.target_valid = False
        self.last_target_time = 0.0
        self.last_goal_pose = None
        self.ema_x = None
        self.ema_y = None
        self._action_goal_handle = None

        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, 'person_follow/target', self.target_cb, 10)
        self.create_subscription(Bool, 'person_follow/target_valid', self.valid_cb, 10)
        self.create_subscription(RobotMode, '/robot_mode', self.mode_cb, 10)

        self.pub_path = self.create_publisher(Path, 'person_follow/path', 10)
        self._action_client = ActionClient(self, FollowPath, 'follow_path')
        self.create_timer(1.0 / self.max_goal_rate, self.control_loop)

        self.get_logger().info(f'FollowController (MPPI) ready. follow_dist={self.follow_dist}m')

    def odom_cb(self, msg):
        self.current_odom = msg

    def target_cb(self, msg):
        rx, ry = msg.pose.position.x, msg.pose.position.y
        if self.ema_x is None:
            self.ema_x, self.ema_y = rx, ry
        else:
            a = self.ema_alpha
            self.ema_x = a * rx + (1 - a) * self.ema_x
            self.ema_y = a * ry + (1 - a) * self.ema_y
        self.last_target_time = time.time()

    def valid_cb(self, msg):
        self.target_valid = msg.data

    def mode_cb(self, msg):
        prev = self.mode
        self.mode = msg.mode
        if prev == RobotMode.FOLLOW and self.mode != RobotMode.FOLLOW:
            self._cancel_goal()

    def control_loop(self):
        if self.mode != RobotMode.FOLLOW or self.current_odom is None:
            return
        if not self.target_valid or self.ema_x is None:
            return
        if time.time() - self.last_target_time > self.target_timeout:
            self._cancel_goal()
            return

        dist = math.sqrt(self.ema_x**2 + self.ema_y**2)
        bearing = math.atan2(self.ema_y, self.ema_x)

        if abs(dist - self.follow_dist) < self.follow_tol:
            self._cancel_goal()
            return

        gx_r = (dist - self.follow_dist) * math.cos(bearing)
        gy_r = (dist - self.follow_dist) * math.sin(bearing)
        goal_pose = self._robot_to_odom(gx_r, gy_r, bearing)

        if self.last_goal_pose is not None:
            dx = goal_pose.pose.position.x - self.last_goal_pose.pose.position.x
            dy = goal_pose.pose.position.y - self.last_goal_pose.pose.position.y
            if math.sqrt(dx**2 + dy**2) < self.pos_thresh:
                return

        self._send_follow_path(goal_pose)
        self.last_goal_pose = goal_pose

    def _robot_to_odom(self, x_r, y_r, yaw_r):
        odom = self.current_odom
        rx, ry = odom.pose.pose.position.x, odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        rob_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y**2 + q.z**2)
        )
        cy, sy = math.cos(rob_yaw), math.sin(rob_yaw)
        ox = rx + cy * x_r - sy * y_r
        oy = ry + sy * x_r + cy * y_r
        oyaw = rob_yaw + yaw_r

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.pose.position.x = ox
        goal.pose.position.y = oy
        goal.pose.orientation.z = math.sin(oyaw / 2.0)
        goal.pose.orientation.w = math.cos(oyaw / 2.0)
        return goal

    def _send_follow_path(self, goal_pose):
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            return
        odom = self.current_odom
        start = PoseStamped()
        start.header = goal_pose.header
        start.pose = odom.pose.pose

        path = Path()
        path.header = goal_pose.header
        path.poses = [start, goal_pose]
        self.pub_path.publish(path)

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = 'FollowPath'
        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_cb)

    def _goal_cb(self, future):
        handle = future.result()
        if handle and handle.accepted:
            self._action_goal_handle = handle

    def _cancel_goal(self):
        if self._action_goal_handle is not None:
            self._action_goal_handle.cancel_goal_async()
            self._action_goal_handle = None


def main():
    rclpy.init()
    rclpy.spin(FollowControllerNode())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
