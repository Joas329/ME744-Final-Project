from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import OpaqueFunction
from launch.substitutions import Command

share = Path(get_package_share_directory("ml_ekf"))

def launch_setup(context, *args, **kwargs):
    desc = ParameterValue(Command(["xacro ", str(share / "description/servo_arm.urdf.xacro")]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": desc}]
    )

    joint_state_publisher_gui = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui"
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(share / "config/display.rviz")]
    )

    nodes = [
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz
    ]

    return nodes

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup)
    ])