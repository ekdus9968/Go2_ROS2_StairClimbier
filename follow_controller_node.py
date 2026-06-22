#!/usr/bin/env python3
"""
follow_controller_node.py

Updated version of person_follow_nav's follow_controller_node.
Key changes from original:
  1. No longer reads from UDP socket subscribes to person_follow/target
     (published by person_detector_node) instead.
  2. Gated by /robot_mode topic only active when mode == FOLLOW (1).
     Cancels any active Nav2 goal and stops when mode changes away from FOLLOW.
  3. Same EMA filtering, rate limiting, and tolerance logic as original.

Topics subscribed:
  /odom                      nav_msgs/Odometry
  person_follow/target       geometry_msgs/PoseStamped   (from person_detector)
  person_follow/target_valid std_msgs/Bool               (from person_detector)
  /robot_mode                go2_interfaces/RobotMode    (from state machine)

Topics published:
  person_follow/path         nav_msgs/Path

Actions:
  follow_path                nav2_msgs/action/FollowPath (to controller_server)
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from nav2_msgs.action import FollowPath
import math
import time

# Import our custom message make sure go2_interfaces is built first
from go2_interfaces.msg import RobotMode

class FollowControllerNode(Node):

    def __init__(self):
        super().__init__('follow_controller')

        # Parameters (match original where possible)
        self.declare_parameter('follow_distance_m', 1.2)       # desired distance to person
        self.declare_parameter('follow_tolerance_m', 0.2)      # within this = hold position
        self.declare_parameter('bearing_hold_tolerance_rad', 0.15)
        self.declare_parameter('max_goal_rate_hz', 2.0)        # max FollowPath goals/sec
        self.declare_parameter('goal_position_threshold_m', 0.1)
        self.declare_parameter('goal_yaw_threshold_rad', 0.1)
        self.declare_parameter('ema_alpha', 0.6)               # smoothing factor
        self.declare_parameter('target_hold_sec', 0.7)         # hold last target this long
        self.declare_parameter('target_timeout_sec', 1.0)      # drop target after this

        self.follow_dist = self.get_parameter('follow_distance_m').value
        self.follow_tol = self.get_parameter('follow_tolerance_m').value
        self.bearing_tol = self.get_parameter('bearing_hold_tolerance_rad').value
        self.max_goal_rate = self.get_parameter('max_goal_rate_hz').value
        self.pos_thresh = self.get_parameter('goal_position_threshold_m').value
        self.yaw_thresh = self.get_parameter('goal_yaw_threshold_rad').value
        self.ema_alpha = self.get_parameter('ema_alpha').value
        self.target_hold_sec = self.get_parameter('target_hold_sec').value
        self.target_timeout_sec = self.get_parameter('target_timeout_sec').value

        # State
        self.current_odom = None
        self.active_mode = RobotMode.IDLE   # start idle, wait for state machine
        self.target_valid = False
        self.last_target_time = 0.0
        self.last_goal_time = 0.0
        self.last_goal_pose = None          # last PoseStamped sent to Nav2

        # EMA-smoothed target position (in base_link frame)
        self.ema_x = None
        self.ema_y = None

        self._action_goal_handle = None

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseStamped, 'person_follow/target', self.target_cb, 10)
        self.create_subscription(Bool, 'person_follow/target_valid', self.valid_cb, 10)
        self.create_subscription(RobotMode, '/robot_mode', self.mode_cb, 10)

        # Publishers
        self.pub_path = self.create_publisher(Path, 'person_follow/path', 10)

        # Nav2 action client
        self._action_client = ActionClient(self, FollowPath, 'follow_path')

        # Control loop
        period = 1.0 / self.max_goal_rate
        self.create_timer(period, self.control_loop)

        self.get_logger().info('FollowControllerNode started.')
        self.get_logger().info('  Waiting for /robot_mode = FOLLOW to activate.')

    #
    # Callbacks
    #
    def odom_cb(self, msg: Odometry):
        self.current_odom = msg

    def target_cb(self, msg: PoseStamped):
        """Receive target from person_detector and apply EMA smoothing."""
        raw_x = msg.pose.position.x
        raw_y = msg.pose.position.y

        if self.ema_x is None:
            self.ema_x = raw_x
            self.ema_y = raw_y
        else:
            a = self.ema_alpha
            self.ema_x = a * raw_x + (1 - a) * self.ema_x
            self.ema_y = a * raw_y + (1 - a) * self.ema_y

        self.last_target_time = time.time()

    def valid_cb(self, msg: Bool):
        self.target_valid = msg.data

    def mode_cb(self, msg: RobotMode):
        """
        Respond to state machine mode changes.
        If we were following and mode changes away, cancel the active Nav2 goal.
        """
        prev_mode = self.active_mode
        self.active_mode = msg.mode

        mode_names = {
            RobotMode.IDLE: 'IDLE',
            RobotMode.FOLLOW: 'FOLLOW',
            RobotMode.STAIR_CLIMB: 'STAIR_CLIMB'
        }
        self.get_logger().info(
            f'Mode change: {mode_names.get(prev_mode, "?")} -> {mode_names.get(self.active_mode, "?")}'
        )

        if prev_mode == RobotMode.FOLLOW and self.active_mode != RobotMode.FOLLOW:
            self._cancel_goal()

    #
    # Control loop
    #
    def control_loop(self):
        """Main loop runs at max_goal_rate_hz."""

        # Only run when in FOLLOW mode
        if self.active_mode != RobotMode.FOLLOW:
            return

        if self.current_odom is None:
            return

        # Check if target is still fresh
        now = time.time()
        target_age = now - self.last_target_time
        have_fresh_target = (
            self.target_valid and
            self.ema_x is not None and
            target_age < self.target_hold_sec
        )

        if not have_fresh_target:
            if target_age > self.target_timeout_sec:
                # Target truly gone cancel goal, let go2_nav_bridge watchdog stop robot
                self._cancel_goal()
            return

        # Build goal pose: step back from person by follow_distance_m
        # ema_x is forward distance to person, ema_y is lateral offset
        dist_to_person = math.sqrt(self.ema_x ** 2 + self.ema_y ** 2)
        bearing = math.atan2(self.ema_y, self.ema_x)

        # Desired robot position: follow_dist behind the person along bearing
        goal_x_robot_frame = (dist_to_person - self.follow_dist) * math.cos(bearing)
        goal_y_robot_frame = (dist_to_person - self.follow_dist) * math.sin(bearing)

        # Within tolerance hold position, cancel goal
        if (abs(dist_to_person - self.follow_dist) < self.follow_tol and
                abs(bearing) < self.bearing_tol):
            self._cancel_goal()
            return

        # Convert to odom frame using current robot pose
        goal_pose = self._robot_to_odom(goal_x_robot_frame, goal_y_robot_frame, bearing)

        # Rate and delta limiting don't spam Nav2
        time_since_last = now - self.last_goal_time
        min_period = 1.0 / self.max_goal_rate

        if time_since_last < min_period:
            return

        if self.last_goal_pose is not None:
            dx = goal_pose.pose.position.x - self.last_goal_pose.pose.position.x
            dy = goal_pose.pose.position.y - self.last_goal_pose.pose.position.y
            dist_delta = math.sqrt(dx ** 2 + dy ** 2)
            if dist_delta < self.pos_thresh:
                return

        # Send goal to Nav2
        self._send_follow_path_goal(goal_pose)
        self.last_goal_time = now
        self.last_goal_pose = goal_pose

    #
    # Nav2 interaction
    #
    def _robot_to_odom(self, x_robot: float, y_robot: float, yaw: float) -> PoseStamped:
        """
        Transform a point from robot base_link frame to odom frame
        using the current odometry pose.
        """
        odom = self.current_odom
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y

        # Get robot yaw from odom quaternion
        q = odom.pose.pose.orientation
        robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        )

        # Rotate and translate
        cos_y = math.cos(robot_yaw)
        sin_y = math.sin(robot_yaw)
        ox = rx + cos_y * x_robot - sin_y * y_robot
        oy = ry + sin_y * x_robot + cos_y * y_robot
        oyaw = robot_yaw + yaw

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'odom'
        goal.pose.position.x = ox
        goal.pose.position.y = oy
        goal.pose.position.z = 0.0

        # Yaw to quaternion
        goal.pose.orientation.z = math.sin(oyaw / 2.0)
        goal.pose.orientation.w = math.cos(oyaw / 2.0)

        return goal

    def _send_follow_path_goal(self, goal_pose: PoseStamped):
        """Build a single-waypoint Path and send as FollowPath action goal."""
        if not self._action_client.wait_for_server(timeout_sec=0.1):
            self.get_logger().warn('controller_server not available yet')
            return

        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'odom'
        path.poses = [goal_pose]

        self.pub_path.publish(path)

        goal_msg = FollowPath.Goal()
        goal_msg.path = path
        goal_msg.controller_id = 'FollowPath'

        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('FollowPath goal rejected by controller_server')
            return
        self._action_goal_handle = handle

    def _cancel_goal(self):
        if self._action_goal_handle is not None:
            self._action_goal_handle.cancel_goal_async()
            self._action_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    node = FollowControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
