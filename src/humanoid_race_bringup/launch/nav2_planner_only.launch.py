#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    bringup_share = get_package_share_directory(
        "humanoid_race_bringup"
    )

    nav2_bringup_share = get_package_share_directory(
        "nav2_bringup"
    )

    params_file = os.path.join(
        bringup_share,
        "config",
        "nav2_params.yaml",
    )

    map_file = (
        "/home/loq/humanoid_race_ws/"
        "maps/h1_rectangle_map.yaml"
    )

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_bringup_share,
                "launch",
                "localization_launch.py",
            )
        ),
        launch_arguments={
            "map": map_file,
            "params_file": params_file,
            "use_sim_time": "False",
            "autostart": "True",
            "use_composition": "False",
        }.items(),
    )

    planner_server = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[params_file],
    )

    planner_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_planner",
        output="screen",
        parameters=[
            {
                "autostart": True,
                "node_names": [
                    "planner_server",
                ],
            }
        ],
    )

    return LaunchDescription(
        [
            localization,
            planner_server,
            planner_lifecycle_manager,
        ]
    )
