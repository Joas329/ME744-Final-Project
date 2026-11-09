# ros2 launch ml_ekf display.launch.py # RViz + joint slider (no physics)
# ros2 launch ml_ekf display.launch.py gazebo:=true # Gazebo + RViz; Gazebo publishes the joint state

import os
import xacro

from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue

share = Path(get_package_share_directory("ml_ekf"))

def launch_setup(context, *args, **kwargs):
    gazebo = LaunchConfiguration("gazebo").perform(context) == "true"
    world = LaunchConfiguration("world").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)
    servo = LaunchConfiguration("servo").perform(context)
    command = LaunchConfiguration("command").perform(context)
    sim_time = {"use_sim_time": gazebo}  # in Gazebo every node must follow the simulation clock

    # sim:=true adds the fixed joint to the world and the Gazebo plugins (servo_arm.gazebo.xacro)
    desc = ParameterValue(Command(["xacro ", str(share / "description/servo_arm.urdf.xacro"), f" sim:={'true' if gazebo else 'false'}"]), value_type=str)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": desc, **sim_time}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", str(share / "config/display.rviz")],
        parameters=[sim_time],
    )

    if not gazebo:
        # without physics, the slider is the only source of /joint_states
        joint_state_publisher_gui = Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui")
        return [robot_state_publisher, joint_state_publisher_gui, rviz]

    # ************ #
    # ** Gazebo ** #
    # ************ #
    out = Path.home() / ".ros" / "ml_ekf_world.sdf"
    out.write_text(xacro.process_file(world).toxml())
    world = str(out)

    res = str(share.parent)
    env = [SetEnvironmentVariable(v, os.pathsep.join(filter(None, [res, os.environ.get(v, "")])))
           for v in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH")]

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(Path(get_package_share_directory("ros_gz_sim")) / "launch/gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {world}"}.items(),
    )

    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=["-world", world_name, "-name", "servo_arm", "-topic", "robot_description"],
        output="screen",
    )

    # Gazebo <-> ROS topics. Direction: "[" = Gazebo to ROS, "]" = ROS to Gazebo
    joint_state = f"/world/{world_name}/model/servo_arm/joint_state"
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
            f"{joint_state}@sensor_msgs/msg/JointState[ignition.msgs.Model",
            "/model/servo_arm/joint/servo_joint/cmd_force@std_msgs/msg/Float64]ignition.msgs.Double",
            "/rig/camera/image@sensor_msgs/msg/Image[ignition.msgs.Image",
        ],
        remappings=[
            (joint_state, "/joint_states"), # robot_state_publisher listens here
            ("/model/servo_arm/joint/servo_joint/cmd_force", "/servo/torque"), # publish a torque [N m] here
        ],
        parameters=[sim_time],
        output="screen",
    )

    # servo: target angle -> motor torque. "true" = the hidden real-servo electronics, "nominal" = ideal PD
    nodes = []

    true_servo = Node(package="ml_ekf", executable="true_servo", parameters=[sim_time], output="screen")
    nominal_servo = Node(package="ml_ekf", executable="nominal_servo", parameters=[sim_time], output="screen")

    if servo in ("true", "nominal"):
        if servo == "true":
            nodes.append(true_servo)
        elif servo == "nominal":
            nodes.append(nominal_servo)

    if command != "none":
        nodes.append(Node(package="ml_ekf", executable="servo_command", parameters=[{"mode": command, **sim_time}], output="screen"))

    # ****************** #
    # ** Angle sensor ** #
    # ****************** #
    angle_sensor = Node(package="ml_ekf", executable="angle_sensor", parameters=[sim_time], output="screen")

    # ******************* #
    # ** Data Recorder ** #
    # ******************* #
    data_recorder = Node(package="ml_ekf", executable="recorder", parameters=[sim_time], output="screen")

    return [*env, gz_sim, robot_state_publisher, spawn, bridge, rviz, *nodes, angle_sensor, data_recorder]

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gazebo", default_value="false", description="true: simulate in Gazebo"),
        DeclareLaunchArgument("world", default_value=str(share / "description/rig.sdf.xacro"), description="world file"),
        DeclareLaunchArgument("world_name", default_value="rig", description="the <world name> inside that file"),
        DeclareLaunchArgument("servo", default_value="true", description="true | nominal | none"),
        DeclareLaunchArgument("command", default_value="none", description="steps | sine | chirp | none"),

        OpaqueFunction(function=launch_setup),
    ])
