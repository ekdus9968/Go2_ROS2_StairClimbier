"""Launch camera keypoint detection node with parameters from config."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
 
 
def generate_launch_description():
    config_file = os.path.join(
        get_package_share_directory('person_detector_camera'),
        'config',
        'camera_params.yaml'
    )
    
    return LaunchDescription([
        Node(
            package='person_detector_camera',
            executable='camera_keypoint_node',
            name='camera_keypoint_node',
            parameters=[config_file],
            output='screen',
        )
    ])
 