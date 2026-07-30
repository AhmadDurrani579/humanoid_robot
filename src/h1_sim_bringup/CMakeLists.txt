cmake_minimum_required(VERSION 3.8)
project(h1_sim_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

install(
  DIRECTORY
    launch
    config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/start_h1_policy.sh
    scripts/deploy_h1_keyboard.py
    scripts/deploy_h1_ros.py
  DESTINATION share/${PROJECT_NAME}/scripts
)

ament_package()
