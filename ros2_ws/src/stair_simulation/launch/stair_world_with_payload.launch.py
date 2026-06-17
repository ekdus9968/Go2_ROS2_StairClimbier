"""Launch with patient stairs + Go2 with payload."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rname = "go2"
    
    # our world
    world_pkg = get_package_share_directory('stair_simulation')
    world_file = os.path.join(world_pkg, 'worlds', 'patient_stairs.world')
    
    # Go2 + payload xacro
    payload_pkg = get_package_share_directory('go2_with_payload')
    xacro_file = os.path.join(payload_pkg, 'urdf', 'go2_with_payload.urdf.xacro')
    
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )
    
    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )
    
    # Gazebo with our world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"), 
                "launch", "gazebo.launch.py"
            )
        ),
        launch_arguments={"world": world_file}.items(),
    )
    
    # Spawn
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "/robot_description",
            "-entity", "robot_model",
            "-x", "0.0",
            "-y", "0.0",
            "-z", "0.5",
        ],
        output="screen",
    )
    
    # Joint state broadcaster
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable='spawner',
        arguments=["joint_state_broadcaster"],
        output="screen",
    )
    
    # Param node (for rl_sar)
    param_node = Node(
        package="demo_nodes_cpp",
        executable="parameter_blackboard",
        name="param_node",
        parameters=[{
            "robot_name": rname,
            "gazebo_model_name": rname + "_gazebo",
        }],
    )
    
    return LaunchDescription([
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster,
        param_node,
    ])