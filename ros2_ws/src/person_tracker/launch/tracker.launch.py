"""Launch particle filter node."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('person_tracker'),
        'config',
        'tracker_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='person_tracker',
            executable='particle_filter_node',
            name='particle_filter_node',
            parameters=[config_file],
            output='screen',
        )
    ])