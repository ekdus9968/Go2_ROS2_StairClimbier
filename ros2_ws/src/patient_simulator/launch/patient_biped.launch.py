"""Patient biped launch with isolated controller_manager."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('patient_simulator')
    xacro_file = os.path.join(pkg_share, 'urdf', 'biped.urdf.xacro')
    
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=None
    )
    
    x_pos = LaunchConfiguration('x', default='1.0')
    y_pos = LaunchConfiguration('y', default='0.0')
    z_pos = LaunchConfiguration('z', default='0.0')
    
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='patient_state_publisher',
        namespace='patient',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
    )
    
    spawn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/patient/robot_description',
            '-entity', 'patient',
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
        ],
        output='screen',
    )
    
    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'joint_state_broadcaster',
            '--controller-manager', '/patient/controller_manager',
        ],
        output='screen',
    )
    
    traj_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=[
            'patient_trajectory_controller',
            '--controller-manager', '/patient/controller_manager',
        ],
        output='screen',
    )
    
    delayed_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn,
            on_exit=[jsb_spawner],
        )
    )
    
    delayed_traj = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb_spawner,
            on_exit=[traj_spawner],
        )
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='1.00'),
        rsp,
        spawn,
        delayed_jsb,
        delayed_traj,
    ])
