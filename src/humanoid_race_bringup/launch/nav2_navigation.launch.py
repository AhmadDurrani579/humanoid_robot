#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


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

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                nav2_bringup_share,
                "launch",
                "bringup_launch.py",
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

    return LaunchDescription([nav2])
