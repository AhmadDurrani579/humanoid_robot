cmake_minimum_required(VERSION 3.8)
project(h1_rgbd_reconstruction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ----------------------------------------------------------
# Dependencies
# ----------------------------------------------------------

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(message_filters REQUIRED)

find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(RTABMap REQUIRED)
find_package(OpenCV REQUIRED)


find_package(PCL REQUIRED COMPONENTS common filters)


# ----------------------------------------------------------
# RGB-D reconstruction node
# ----------------------------------------------------------

add_executable(
  rgbd_reconstruction_node
  src/rgbd_reconstruction_node.cpp
  src/rtabmap_reconstruction_backend.cpp
)

ament_target_dependencies(
  rgbd_reconstruction_node
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  pcl_conversions
  pcl_ros
  tf2
  tf2_ros
)

target_include_directories(
  rgbd_reconstruction_node
  PRIVATE
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  rgbd_reconstruction_node
  ${PCL_LIBRARIES}
  rtabmap::core
  ${OpenCV_LIBRARIES}
)

install(
  TARGETS
  rgbd_reconstruction_node
  DESTINATION lib/${PROJECT_NAME}
)


# ----------------------------------------------------------
# Tests
# ----------------------------------------------------------

if(BUILD_TESTING)

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(
    test_sync_logic
    test/test_sync_logic.cpp
  )

  ament_add_gtest(
    test_projection
    test/test_projection.cpp
  )
  
  ament_add_gtest(
    test_rgb_mapping
    test/test_rgb_mapping.cpp
  )

  ament_add_gtest(
    test_voxel_accumulation
    test/test_voxel_accumulation.cpp
  )

  target_include_directories(
    test_voxel_accumulation
    PRIVATE
    ${PCL_INCLUDE_DIRS}
  )

  target_link_libraries(
    test_voxel_accumulation
    ${PCL_LIBRARIES}
  )

  ament_add_gtest(
    test_incremental_voxel_map
    test/test_incremental_voxel_map.cpp
  )

  target_include_directories(
    test_incremental_voxel_map
    PRIVATE
    include
    ${PCL_INCLUDE_DIRS}
  )

  target_link_libraries(
    test_incremental_voxel_map
    ${PCL_LIBRARIES}
  )

  ament_add_gtest(
    test_voxel_frame_count
    test/test_voxel_frame_count.cpp
  )

  target_include_directories(
    test_voxel_frame_count
    PRIVATE
    include
    ${PCL_INCLUDE_DIRS}
  )

  target_link_libraries(
    test_voxel_frame_count
    ${PCL_LIBRARIES}
  )
  
endif()


ament_package()