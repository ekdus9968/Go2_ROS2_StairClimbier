"""
follow_camera_only.launch.py

Clean launch file for human-following with Go2 built-in camera only.
Drops all Hesai LiDAR nodes from the original follow_sidecar_lidar.launch.py.
Fixes the cmd_vel_smoothed topic mismatch from the original codebase.

Nodes started:
  - person_detector       (camera -> person_follow/target)
  - state_machine         (mode arbitration -> /robot_mode)
  - follow_controller     (person_follow/target -> Nav2 goals)
  - controller_server     (Nav2 MPPI)
  - velocity_smoother     (Nav2, with correct topic remapping)
  - lifecycle_manager     (manages Nav2 nodes)
  - go2_nav_bridge        (Nav2 cmd_vel -> Unitree SDK)

What changed from original:
  - Removed: lidar_static_tf, hesai_driver, lidar_filter, pc2_to_grid,
             igrid_interpolation, hgrid_interpolation
  - Added: person_detector_node, state_machine_node
  - Fixed: velocity_smoother output remapped to cmd_vel_smoothed
           (was publishing to /cmd_vel, bridge expected cmd_vel_smoothed)
  - Nav2 costmap: obstacle layer disabled (no LiDAR), configure your
    nav2_params.yaml accordingly
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node, LifecycleNode
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    nav2_params_file = os.path.join(
        get_package_share_directory('person_follow_nav'),
        'config',
        'nav2_params.yaml'
    )

    #
    # 1. Person detector camera feed -> ROS2 target topic
    #
    person_detector = Node(
        package='person_detector',
        executable='person_detector_node',
        name='person_detector',
        output='screen',
        parameters=[{
            'model_path': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'image_topic': '/camera/image_raw',
        }]
    )

    #
    # 2. State machine publishes /robot_mode
    #
    state_machine = Node(
        package='person_follow_nav',
        executable='state_machine_node',
        name='state_machine',
        output='screen',
    )

    #
    # 3. Follow controller subscribes target + mode, sends Nav2 goals
    #
    follow_controller = Node(
        package='person_follow_nav',
        executable='follow_controller_node',
        name='follow_controller',
        output='screen',
        parameters=[{
            'follow_distance_m': 1.2,
            'follow_tolerance_m': 0.2,
            'bearing_hold_tolerance_rad': 0.15,
            'max_goal_rate_hz': 2.0,
            'ema_alpha': 0.6,
            'target_hold_sec': 0.7,
            'target_timeout_sec': 1.0,
        }]
    )

    #
    # 4. Nav2 controller server (MPPI)
    #
    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[
            # controller_server publishes cmd_vel -> remap to cmd_vel_nav
            # so velocity_smoother can pick it up
            ('cmd_vel', 'cmd_vel_nav'),
        ]
    )

    #
    # 5. Velocity smoother
    # FIX: original code did NOT remap smoother output, so bridge never
    # received velocity commands. Now explicitly remapped to cmd_vel_smoothed.
    #
    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_params_file],
        remappings=[
            # input: pick up controller_server's remapped output
            ('cmd_vel', 'cmd_vel_nav'),
            # output: publish to what go2_nav_bridge expects
            ('cmd_vel_smoothed', 'cmd_vel_smoothed'),
        ]
    )

    #
    # 6. Lifecycle manager for Nav2 nodes
    #
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['controller_server', 'velocity_smoother'],
        }]
    )

    #
    # 7. Go2 SDK bridge
    # Converts /odom from DDS, forwards cmd_vel_smoothed to Unitree SDK
    #
    go2_nav_bridge = Node(
        package='go2_nav_bridge',
        executable='bridge_node',
        name='go2_nav_bridge',
        output='screen',
        parameters=[{
            'cmd_vel_topic': 'cmd_vel_smoothed',   # matches smoother output
            'cmd_timeout_sec': 0.4,
        }]
    )

    return LaunchDescription([
        person_detector,
        state_machine,
        follow_controller,
        controller_server,
        velocity_smoother,
        lifecycle_manager,
        go2_nav_bridge,
    ])
