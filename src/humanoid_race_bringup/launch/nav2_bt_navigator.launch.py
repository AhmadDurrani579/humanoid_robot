#!/usr/bin/env python3

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory(
            "humanoid_race_bringup"
        )
    )

    params_file = (
        package_share
        / "config"
        / "nav2_params.yaml"
    )

    behavior_server = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[str(params_file)],
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[str(params_file)],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_bt_navigation",
        output="screen",
        parameters=[
            {
                "use_sim_time": False,
                "autostart": True,
                "node_names": [
                    "behavior_server",
                    "bt_navigator",
                ],
            }
        ],
    )

    return LaunchDescription(
        [
            behavior_server,
            bt_navigator,
            lifecycle_manager,
        ]
    )