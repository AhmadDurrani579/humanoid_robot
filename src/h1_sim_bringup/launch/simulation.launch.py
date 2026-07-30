from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    policy_script = PathJoinSubstitution(
        [
            FindPackageShare("h1_sim_bringup"),
            "scripts",
            "start_h1_policy.sh",
        ]
    )

    start_h1_policy = ExecuteProcess(
        cmd=[policy_script],
        output="screen",
        shell=False,
    )

    return LaunchDescription(
        [
            start_h1_policy,
        ]
    )