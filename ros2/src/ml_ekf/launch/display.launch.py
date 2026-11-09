from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command

share = Path(get_package_share_directory("ml_ekf"))

def generate_launch_description():
    desc = ParameterValue(Command(["xacro ", str(share / "description/servo_arm.urdf.xacro")]), value_type=str)
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": desc}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(share / "config/display.rviz")]),
    ])
