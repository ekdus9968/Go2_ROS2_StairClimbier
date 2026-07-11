"""Launch stair detector node."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('stair_detector_lidar'),
        'config',
        'stair_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='stair_detector_lidar',
            executable='stair_detector_node',
            name='stair_detector_node',
            parameters=[config_file],
            output='screen',
        )
    ])