"""Launch LiDAR clustering node with parameters from config file."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('person_detector_lidar'),
        'config',
        'lidar_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='person_detector_lidar',
            executable='lidar_clustering_node',
            name='lidar_clustering_node',
            parameters=[config_file],
            output='screen',
        )
    ])
