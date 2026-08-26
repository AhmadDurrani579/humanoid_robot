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
from ament_index_python.packages import get_package_prefix


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
    
    
    enable_reconstruction = LaunchConfiguration(
        "enable_reconstruction"
    )


    declare_use_waypoint_controller = DeclareLaunchArgument(
        "use_waypoint_controller",
        default_value="true",
        description=(
            "Start the custom waypoint VFH controller"
        ),
    )
    
    declare_enable_reconstruction = DeclareLaunchArgument(
        "enable_reconstruction",
        default_value="true",
        description=(
            "Start persistent RGB-D 3D reconstruction"
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

    yolo_node = ExecuteProcess(
        cmd=[
            "/home/loq/yolo_ros2_env/bin/python",
            os.path.join(
                get_package_prefix("h1_locomotion_bridge"),
                "lib",
                "h1_locomotion_bridge",
                "yolo_detector_node.py",
            ),
        ],
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

    # MuJoCo:
    # pelvis -> camera_mount
    # pos="0.12 0 0.48"
    # euler="0 0.2094 0"
    camera_mount_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_mount_static_transform",
        arguments=[
            "--x", "0.12",
            "--y", "0.0",
            "--z", "0.48",
            "--roll", "0.0",
            "--pitch", "0.2094",
            "--yaw", "0.0",
            "--frame-id", "pelvis",
            "--child-frame-id", "camera_mount",
        ],
        output="screen",
    )

    # camera_mount -> ROS depth optical frame
    #
    # Camera position inside camera_mount:
    # pos="0.050 0 0"
    #
    # ROS optical-frame convention:
    # X = right
    # Y = down
    # Z = forward
    camera_depth_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_depth_static_transform",
        arguments=[
            "--x", "0.05",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "-1.57079632679",
            "--pitch", "0.0",
            "--yaw", "-1.57079632679",
            "--frame-id", "camera_mount",
            "--child-frame-id", "camera_depth_optical_frame",
        ],
        output="screen",
    )
    
    # ---------------------------------------------------------
    # RGB-D NAVIGATION PERCEPTION PIPELINE
    # ---------------------------------------------------------

    depth_nav_resampler_node = Node(
        package="h1_locomotion_bridge",
        executable="depth_nav_resampler.py",
        name="depth_nav_resampler",
        output="screen",
        parameters=[
            {
                "factor": 2,
                "input_depth_topic": "/camera/depth/image_raw",
                "input_info_topic": "/camera/depth/camera_info",
                "output_depth_topic": "/camera/depth/image_nav",
                "output_info_topic": "/camera/depth/camera_info_nav",

                # Navigation-only corrected camera frame.
                "output_frame": "camera_depth_nav_optical_frame",
            }
        ],
    )
    
    semantic_depth_fusion_node = Node(
        package="h1_locomotion_bridge",
        executable="semantic_depth_fusion_node.py",
        name="semantic_depth_fusion_node",
        output="screen",
    )
    
    nav_depth_sensor_model_node = Node(
        package="h1_locomotion_bridge",
        executable="depth_sensor_model.py",
        name="nav_depth_sensor_model",
        output="screen",
        parameters=[
            {
                "input_topic": "/camera/depth/image_nav",
                "output_topic": "/camera/depth/image_nav_realistic",
            }
        ],
    )

    realistic_pointcloud_node = Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        name="realistic_pointcloud_node",
        output="screen",
        remappings=[
            (
                "image_rect",
                "/camera/depth/image_nav_realistic",
            ),
            (
                "/camera/depth/camera_info",
                "/camera/depth/camera_info_nav",
            ),
            (
                "points",
                "/camera/depth/points_realistic",
            ),
        ],
    )
    
        # ---------------------------------------------------------
    # NAVIGATION-ONLY CAMERA TF
    # ---------------------------------------------------------
    #
    # EKF/Nav2 uses pelvis Z = 0 because two_d_mode is enabled.
    #
    # Actual MuJoCo standing pelvis height measured from /odom:
    #   1.0300202369689941 m
    #
    # Physical camera mount above pelvis:
    #   0.48 m
    #
    # Navigation camera mount height:
    #   1.0300202369689941 + 0.48
    #   = 1.5100202369689941 m
    #
    # This does NOT modify the original physical camera TF.
    #

    camera_nav_mount_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_nav_mount_static_transform",
        arguments=[
            "--x", "0.12",
            "--y", "0.0",
            "--z", "1.5100202369689941",
            "--roll", "0.0",
            "--pitch", "0.2094",
            "--yaw", "0.0",
            "--frame-id", "pelvis",
            "--child-frame-id", "camera_nav_mount",
        ],
        output="screen",
    )

    camera_depth_nav_static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_depth_nav_static_transform",
        arguments=[
            "--x", "0.05",
            "--y", "0.0",
            "--z", "0.0",
            "--roll", "-1.57079632679",
            "--pitch", "0.0",
            "--yaw", "-1.57079632679",
            "--frame-id", "camera_nav_mount",
            "--child-frame-id", "camera_depth_nav_optical_frame",
        ],
        output="screen",
    )
    
    
    # ---------------------------------------------------------
    # RGB-D 3D RECONSTRUCTION
    # ---------------------------------------------------------

    rtab_rgb_relay_node = Node(
        package="h1_locomotion_bridge",
        executable="rtab_rgb_relay.py",
        name="rtab_rgb_relay",
        output="screen",
        condition=IfCondition(
            enable_reconstruction
        ),
    )


    rtab_point_cloud_xyzrgb_node = Node(
        package="rtabmap_util",
        executable="point_cloud_xyzrgb",
        name="rtab_point_cloud_xyzrgb",
        output="screen",
        parameters=[
            {
                "approx_sync": True,
                "decimation": 1,

                # Current tested value.
                # Small voxel reduction to reduce duplicate points.
                "voxel_size": 0.02,
            }
        ],
        remappings=[
            (
                "rgb/image",
                "/camera/color/image_rtab",
            ),
            (
                "depth/image",
                "/camera/depth/image_nav_realistic",
            ),
            (
                "rgb/camera_info",
                "/camera/depth/camera_info_nav",
            ),
            (
                "cloud",
                "/cloud",
            ),
        ],
        condition=IfCondition(
            enable_reconstruction
        ),
    )


    rtab_point_cloud_assembler_node = Node(
        package="rtabmap_util",
        executable="point_cloud_assembler",
        name="rtab_point_cloud_assembler",
        output="screen",
        parameters=[
            {
                # Keep newest 200 RGB-D clouds.
                "max_clouds": 200,

                # Do not reset the whole assembled cloud
                # whenever max_clouds is reached.
                "circular_buffer": True,

                # Align incoming clouds using OUR TF tree.
                "fixed_frame_id": "world",

                # Publish final cloud directly in world.
                "frame_id": "world",

                # We already voxel-filter the live cloud above.
                "voxel_size": 0.0,
            }
        ],
        remappings=[
            (
                "cloud",
                "/cloud",
            ),
            (
                "assembled_cloud",
                "/reconstruction/cloud_assembled",
            ),
        ],
        condition=IfCondition(
            enable_reconstruction
        ),
    )
        
    return LaunchDescription(
        [
            declare_use_waypoint_controller,
            declare_enable_reconstruction,

            h1_policy,
            cmd_vel_udp_bridge,

            # Keep disabled during autonomous operation.
            # keyboard_teleop,

            lidar_publisher_node,
            odom_publisher_node,
            camera_bridge_node,

            waypoint_vfh_controller,

            lidar_static_tf_node,
            
            camera_mount_static_tf_node,
            camera_depth_static_tf_node,
            
            # Navigation-only corrected camera TF.
            camera_nav_mount_static_tf_node,
            camera_depth_nav_static_tf_node,
            
            # RGB-D navigation perception.

            depth_nav_resampler_node,
            nav_depth_sensor_model_node,
            realistic_pointcloud_node,
            
            # Persistent RGB-D 3D reconstruction.
            rtab_rgb_relay_node,
            rtab_point_cloud_xyzrgb_node,
            rtab_point_cloud_assembler_node,


            # Yolo detector node.
            # Semantic perception.

            yolo_node,
            semantic_depth_fusion_node,
            
            robot_state_publisher_node,
            joint_state_publisher_node,
            rviz_node,
        ]
    )