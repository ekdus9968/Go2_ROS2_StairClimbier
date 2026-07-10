#!/usr/bin/env python3
"""
Patient Mover Node.

Moves the patient model in Gazebo simulation using SetEntityState service.
Simulates person walking without leg animation (just teleport-like movement).

Modes:
  - waypoint: move between waypoints
  - constant_velocity: move in a direction at constant velocity
  - manual: publish target pose to /patient_target

For particle filter testing.
"""

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Point, Quaternion
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


class PatientMoverNode(Node):
    def __init__(self):
        super().__init__('patient_mover_node')

        # Parameters
        self.declare_parameter('model_name', 'patient')
        self.declare_parameter('update_rate', 10.0)
        self.declare_parameter('mode', 'constant_velocity')

        # Initial position
        self.declare_parameter('init_x', -3.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_z', 1.05)

        # Velocity (m/s) for constant_velocity mode
        self.declare_parameter('vx', 0.2)  # forward slow
        self.declare_parameter('vy', 0.0)
        self.declare_parameter('vz', 0.0)

        # Boundary (patient wraps or stops at boundary)
        self.declare_parameter('bound_x_min', -5.0)
        self.declare_parameter('bound_x_max', 6.0)
        self.declare_parameter('boundary_action', 'reverse')  # 'reverse' or 'stop'

        self.model_name = self.get_parameter('model_name').value
        rate = self.get_parameter('update_rate').value
        self.mode = self.get_parameter('mode').value

        self.current_x = self.get_parameter('init_x').value
        self.current_y = self.get_parameter('init_y').value
        self.current_z = self.get_parameter('init_z').value

        self.vx = self.get_parameter('vx').value
        self.vy = self.get_parameter('vy').value
        self.vz = self.get_parameter('vz').value

        self.bound_x_min = self.get_parameter('bound_x_min').value
        self.bound_x_max = self.get_parameter('bound_x_max').value
        self.boundary_action = self.get_parameter('boundary_action').value

        self.dt = 1.0 / rate

        # Client for SetEntityState service
        self.client = self.create_client(
            SetEntityState, '/gazebo/set_entity_state')

        self.get_logger().info(
            'Waiting for /gazebo/set_entity_state service...')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Still waiting...')
        self.get_logger().info('Service available')

        # Timer
        self.create_timer(self.dt, self.step)

        self.get_logger().info(
            f'Patient Mover started: mode={self.mode}, '
            f'init=({self.current_x:.1f},{self.current_y:.1f},'
            f'{self.current_z:.1f}), vel=({self.vx:.2f},{self.vy:.2f},{self.vz:.2f})')

    # ==========================
    # Main step
    # ==========================
    def step(self):
        if self.mode == 'constant_velocity':
            self.current_x += self.vx * self.dt
            self.current_y += self.vy * self.dt
            self.current_z += self.vz * self.dt

            # Check boundary
            if self.current_x < self.bound_x_min:
                if self.boundary_action == 'reverse':
                    self.vx = abs(self.vx)
                else:
                    self.current_x = self.bound_x_min
                    self.vx = 0
            elif self.current_x > self.bound_x_max:
                if self.boundary_action == 'reverse':
                    self.vx = -abs(self.vx)
                else:
                    self.current_x = self.bound_x_max
                    self.vx = 0

        self.set_pose(self.current_x, self.current_y, self.current_z)

    # ==========================
    # Set pose via service
    # ==========================
    def set_pose(self, x, y, z):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = self.model_name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.w = 1.0
        state.reference_frame = 'world'
        req.state = state

        future = self.client.call_async(req)
        # No wait (async)


def main():
    rclpy.init()
    node = PatientMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()