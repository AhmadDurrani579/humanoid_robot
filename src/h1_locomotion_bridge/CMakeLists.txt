cmake_minimum_required(VERSION 3.8)
project(h1_locomotion_bridge)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

install(
  PROGRAMS
    scripts/keyboard_teleop.py
    scripts/cmd_vel_udp_bridge.py
    scripts/lidar_udp_publisher.py
    scripts/odom_udp_publisher.py
    scripts/vfh_obstacle_avoidance.py
    scripts/straight_waypoint_follower.py
    scripts/waypoint_vfh_controller.py
    scripts/nav2_vfh_path_controller.py
    scripts/race_mission_node.py
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
