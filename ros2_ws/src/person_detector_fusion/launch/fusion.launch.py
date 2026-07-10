"""Launch person fusion node."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('person_detector_fusion'),
        'config',
        'fusion_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='person_detector_fusion',
            executable='person_fusion_node',
            name='person_fusion_node',
            parameters=[config_file],
            output='screen',
        )
    ])