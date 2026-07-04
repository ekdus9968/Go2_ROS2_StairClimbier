"""PHASE 6b: Full follow stack with proper startup timing."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('high_level_planner')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    odom_to_tf = Node(
        package='high_level_planner',
        executable='odom_to_tf_node',
        name='odom_to_tf',
        output='screen',
        parameters=[{
            'odom_topic': '/odom',
            'parent_frame': 'odom',
            'child_frame': 'base_link',
        }],
    )

    base_link_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_base_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base'],
    )

    fake_detector = Node(
        package='person_detector',
        executable='fake_detector_node',
        name='fake_detector',
        output='screen',
    )

    state_machine = Node(
        package='high_level_planner',
        executable='state_machine_node',
        name='state_machine',
        output='screen',
    )

    controller_server = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[nav2_params],
    )

    velocity_smoother = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[nav2_params],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel'),
        ],
    )

    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_follow',
        output='screen',
        parameters=[{
            'autostart': True,
            'node_names': ['controller_server', 'velocity_smoother'],
            'use_sim_time': True,
        }],
    )

    follow_controller = Node(
        package='high_level_planner',
        executable='follow_controller_node',
        name='follow_controller',
        output='screen',
        parameters=[{
            'follow_distance_m': 1.54,
            'follow_tolerance_m': 0.2,
            'max_goal_rate_hz': 2.0,
        }],
    )

    # TF/detection 노드 즉시 시작
    # Nav2 lifecycle 8초 후 (TF 안정화 대기)
    # follow_controller 12초 후 (Nav2 활성화 대기)
    return LaunchDescription([
        odom_to_tf,
        base_link_to_base,
        fake_detector,
        state_machine,
        controller_server,
        velocity_smoother,
        TimerAction(period=8.0, actions=[lifecycle_manager]),
        TimerAction(period=12.0, actions=[follow_controller]),
    ])
