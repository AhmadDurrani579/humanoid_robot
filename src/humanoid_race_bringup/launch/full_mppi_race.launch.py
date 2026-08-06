#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    bringup_share = get_package_share_directory(
        "humanoid_race_bringup"
    )

    race_system_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share,
                "launch",
                "race_system.launch.py",
            )
        ),
        launch_arguments={
            "use_waypoint_controller": "false",
        }.items(),
    )

    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[
            os.path.join(
                bringup_share,
                "config",
                "ekf.yaml",
            )
        ],
    )

    planner_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share,
                "launch",
                "nav2_planner_only.launch.py",
            )
        )
    )

    mppi_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share,
                "launch",
                "nav2_mppi_controller.launch.py",
            )
        )
    )

    bt_navigator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                bringup_share,
                "launch",
                "nav2_bt_navigator.launch.py",
            )
        )
    )
    
    # udp_bridge = Node(
    #     package="h1_locomotion_bridge",
    #     executable="cmd_vel_udp_bridge.py",
    #     name="cmd_vel_udp_bridge",
    #     output="screen",
    # )

    return LaunchDescription([
        race_system_launch,

        TimerAction(
            period=3.0,
            actions=[ekf_node],
        ),

        TimerAction(
            period=6.0,
            actions=[planner_launch],
        ),

        TimerAction(
            period=9.0,
            actions=[mppi_launch],
        ),
        
        TimerAction(
            period=12.0,
            actions=[bt_navigator_launch],
        ),

        # TimerAction(
        #     period=12.0,
        #     actions=[udp_bridge],
        # ),
    ])