from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        # Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                FindPackageShare('gazebo_ros'),
                '/launch/gazebo.launch.py'
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('stair_simulation'),
                    'worlds',
                    'patient_stairs.world'
                ]),
            }.items()
        ),
        # rl_sar gazebo (already exist : rl_sar/src/rl_sar/launch/gazebo.launch.py)
        # ...
    ])