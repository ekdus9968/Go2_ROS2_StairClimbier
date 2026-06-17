"""Patient stairs world + Go2 with payload + configurable spawn."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # our world
    world_pkg = get_package_share_directory('stair_simulation')
    world_file = os.path.join(world_pkg, 'worlds', 'patient_stairs.world')
    
    # additional sensor Go2 xacro
    payload_pkg = get_package_share_directory('go2_with_payload')
    xacro_file = os.path.join(payload_pkg, 'urdf', 'go2_with_payload.urdf.xacro')
    
    # Robot description
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )
    
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
    )
    
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch', 'gazebo.launch.py'
            )
        ),
        launch_arguments={'world': world_file}.items(),
    )
    
    # Spawn arguments
    x_pos = LaunchConfiguration('x')
    y_pos = LaunchConfiguration('y')
    z_pos = LaunchConfiguration('z')
    yaw = LaunchConfiguration('yaw')
    
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', '/robot_description',
            '-entity', 'robot_model',
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
            '-Y', yaw,
        ],
        output='screen',
    )
    
    joint_state_broadcaster_node = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )
    
    param_node = Node(
        package='demo_nodes_cpp',
        executable='parameter_blackboard',
        name='param_node',
        parameters=[{
            'robot_name': 'go2',
            'gazebo_model_name': 'go2_gazebo',
        }],
    )
    
    return LaunchDescription([
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.5'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        robot_state_publisher_node,
        gazebo,
        spawn_entity_node,
        joint_state_broadcaster_node,
        param_node,
    ])