#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    bringup_share = get_package_share_directory(
        "humanoid_race_bringup"
    )

    h1_description_share = get_package_share_directory(
        "h1_description"
    )

    h1_urdf_path = os.path.join(
        h1_description_share,
        "urdf",
        "h1.urdf",
    )
    
    with open(
        h1_urdf_path,
        "r",
        encoding="utf-8",
    ) as urdf_file:
        robot_description = urdf_file.read()

    use_waypoint_controller = LaunchConfiguration(
        "use_waypoint_controller"
    )

    declare_use_waypoint_controller = DeclareLaunchArgument(
        "use_waypoint_controller",
        default_value="true",
        description=(
            "Start the custom waypoint VFH controller"
        ),
    )

    h1_policy_script = PathJoinSubstitution(
        [
            FindPackageShare("h1_sim_bringup"),
            "scripts",
            "start_h1_policy.sh",
        ]
    )

    h1_policy = ExecuteProcess(
        cmd=[h1_policy_script],
        output="screen",
        shell=False,
    )

    cmd_vel_udp_bridge = Node(
        package="h1_locomotion_bridge",
        executable="cmd_vel_udp_bridge.py",
        name="h1_cmd_vel_udp_bridge",
        output="screen",
    )

    keyboard_teleop = Node(
        package="h1_locomotion_bridge",
        executable="keyboard_teleop.py",
        name="h1_keyboard_teleop",
        output="screen",
        prefix="xterm -e",
    )

    rviz_config = os.path.join(
        get_package_share_directory(
            "humanoid_race_bringup"
        ),
        "config",
        "h1_mppi_obstacle.rviz",
    )


    lidar_publisher_node = Node(
        package="h1_locomotion_bridge",
        executable="lidar_udp_publisher.py",
        name="h1_lidar_udp_publisher",
        output="screen",
    )

    lidar_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_static_transform",
        arguments=[
            "--x", "0.12",
            "--y", "0.0",
            "--z", "0.30",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", "pelvis",
            "--child-frame-id", "lidar_link",
        ],
        output="screen",
    )

    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="h1_rviz",
    #     arguments=[
    #         "-d",
    #         rviz_config,
    #     ],
    #     output="screen",
    # )
    
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="h1_rviz",
        arguments=[
            "-d",
            rviz_config,
        ],
        output="screen",
    )


    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": False,
            }
        ],
    )

    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "rate": 30,
            }
        ],
    )

    odom_publisher_node = Node(
        package="h1_locomotion_bridge",
        executable="odom_udp_publisher.py",
        name="h1_odom_udp_publisher",
        output="screen",
    )

    waypoint_vfh_controller = Node(
        package="h1_locomotion_bridge",
        executable="waypoint_vfh_controller.py",
        name="waypoint_vfh_controller",
        output="screen",
        condition=IfCondition(
            use_waypoint_controller
        ),
    )

    camera_bridge_node = Node(
        package="h1_locomotion_bridge",
        executable="camera_udp_publisher.py",
        name="camera_bridge_node",
        output="screen",
    )
    
    return LaunchDescription(
        [
            declare_use_waypoint_controller,

            h1_policy,
            cmd_vel_udp_bridge,

            # Keep disabled during autonomous operation.
            # keyboard_teleop,

            lidar_publisher_node,
            odom_publisher_node,
            camera_bridge_node,

            waypoint_vfh_controller,

            lidar_static_tf_node,
            robot_state_publisher_node,
            joint_state_publisher_node,
            rviz_node,
        ]
    )