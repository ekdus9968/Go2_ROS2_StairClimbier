"""Launch Gazebo with patient stairs world."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # World file path
    pkg_dir = get_package_share_directory('stair_simulation')
    world_file = os.path.join(pkg_dir, 'worlds', 'patient_stairs.world')
    
    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true',
        }.items()
    )
    
    return LaunchDescription([
        gazebo_launch,
    ])