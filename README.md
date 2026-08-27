# Unitree H1 Humanoid Navigation using ROS 2 Jazzy, MuJoCo and Nav2 MPPI

## Overview

This project implements autonomous navigation of a **Unitree H1 humanoid robot** in a MuJoCo simulation environment using **ROS 2 Jazzy** and the **Nav2 navigation framework**.

The system combines classical navigation methods with RGB-D based perception to enable humanoid obstacle-aware navigation.

The main components include:

- Unitree H1 humanoid robot simulation
- MuJoCo physics environment
- ROS 2 Jazzy communication framework
- Nav2 MPPI local controller
- LiDAR-based navigation
- RGB-D depth perception
- 3D point cloud reconstruction
- Nav2 VoxelLayer obstacle representation
- Multi-goal race navigation framework


---

# System Architecture

The complete navigation pipeline:

```
                    MuJoCo Simulation

                           |
                           v

                    Unitree H1 Robot

             -----------------------------
             |                           |
             v                           v

        LiDAR Sensor              RGB-D Camera

             |                           |

          /scan              Depth PointCloud2

             |                           |

     Nav2 ObstacleLayer        Nav2 VoxelLayer

             |                           |

             -------- Local Costmap -------

                           |

                           v

                 Nav2 MPPI Controller

                           |

                           v

                      /cmd_vel

                           |

                           v

                 H1 Locomotion Bridge
```

---

# Features

## Humanoid Navigation

- Autonomous Unitree H1 navigation
- Multi-goal race navigation
- Dynamic obstacle avoidance
- Nav2 based path planning
- MPPI trajectory optimization


## RGB-D Perception

The RGB-D camera provides depth information for environmental understanding.

Capabilities:

- Depth image processing
- PointCloud2 generation
- 3D environment reconstruction
- Obstacle perception


## 3D Reconstruction

The RGB-D pipeline generates a persistent 3D representation of the environment.

Pipeline:

```
RGB-D Camera

      |

      v

Depth Point Cloud

      |

      v

RTAB-Map Point Cloud Processing

      |

      v

3D Environment Reconstruction
```


## Voxel-Based Navigation

RGB-D point clouds are integrated into Nav2 using:

```
nav2_costmap_2d::VoxelLayer
```

The voxel layer provides 3D obstacle information for local navigation.

Pipeline:

```
/camera/depth/points_realistic

             |

             v

        Nav2 VoxelLayer

             |

             v

      3D Voxel Occupancy

             |

             v

       Local Costmap

             |

             v

      MPPI Controller
```

---

# Visualization Results


## MuJoCo Marathon Environment

The Unitree H1 robot is evaluated inside a custom marathon-style navigation environment.

![MuJoCo Marathon Environment](images/04_mujoco_h1_marathon_track.png)


---

## RGB-D Perception

The simulated RGB-D camera generates point cloud information for environmental understanding.

![RGB-D Perception](images/02_nav2_rgbd_obstacle_detection.png)


---

## Obstacle Scenario

A simulated obstacle is introduced to evaluate perception and avoidance behaviour.

![Obstacle Scenario](images/03_mujoco_h1_robot_obstacle_scene.png)


---

## Nav2 VoxelLayer Navigation

The RGB-D point cloud is converted into a voxel representation and integrated into the Nav2 local costmap.

![Voxel Navigation](images/06_rviz_voxel_layer_navigation.png)


---

## Obstacle Avoidance Result

The robot successfully detects and avoids obstacles while following the navigation route.

![Obstacle Avoidance](images/05_mujoco_h1_obstacle_avoidance_test.png)


---

# Software Requirements

## Operating System

Tested on:

```
Ubuntu 24.04 LTS
```


## ROS Framework

```
ROS 2 Jazzy
```


## Main Dependencies

- ROS 2 Jazzy
- Nav2
- RViz2
- MuJoCo
- RTAB-Map utilities
- CycloneDDS
- Python 3


---

# Installation

Create ROS 2 workspace:

```bash
mkdir -p ~/humanoid_race_ws/src

cd ~/humanoid_race_ws/src
```


Clone repository:

```bash
git clone <repository-url>
```


Install dependencies:

```bash
cd ~/humanoid_race_ws

rosdep install \
--from-paths src \
--ignore-src \
-r -y
```


Build workspace:

```bash
colcon build
```


Source workspace:

```bash
source install/setup.bash
```


---

# Running the Simulation

Set ROS domain:

```bash
export ROS_DOMAIN_ID=5
```


Launch the complete H1 navigation system:

```bash
ros2 launch humanoid_race_bringup full_mppi_race.launch.py
```


The launch file starts:

- Unitree H1 simulation
- Sensor bridges
- TF system
- RGB-D pipeline
- Nav2 stack
- MPPI controller
- Voxel-based obstacle navigation


---

# Important ROS Topics


## Sensors

| Topic | Description |
|---|---|
| `/scan` | LiDAR LaserScan data |
| `/camera/depth/points_realistic` | RGB-D generated point cloud |


## Navigation

| Topic | Description |
|---|---|
| `/local_costmap/costmap` | 2D local costmap |
| `/local_costmap/voxel_grid` | 3D voxel occupancy |
| `/cmd_vel` | Velocity commands |


## Reconstruction

| Topic | Description |
|---|---|
| `/reconstruction/cloud_assembled` | Persistent reconstructed point cloud |


---

# Controller

The local navigation controller uses:

```
Nav2 MPPI Controller
```

MPPI provides:

- Sampling based trajectory optimization
- Smooth velocity generation
- Real-time obstacle avoidance
- Local path following


---

# Results

The system demonstrates:

✅ Unitree H1 autonomous navigation  
✅ ROS 2 Jazzy integration  
✅ Nav2 MPPI control  
✅ RGB-D obstacle perception  
✅ 3D voxel obstacle representation  
✅ Dynamic obstacle avoidance  
✅ Persistent RGB-D reconstruction  
✅ Multi-goal navigation


---

# Demo Videos

Coming soon:

## RGB-D 3D Reconstruction

Demonstration of persistent environment reconstruction using RGB-D perception.


## Nav2 MPPI + VoxelLayer Navigation

Demonstration of humanoid obstacle avoidance using RGB-D voxel perception.


---

# Future Work

Possible improvements:

- Real Unitree H1 hardware deployment
- Improved humanoid locomotion integration
- Learning-based navigation approaches
- Large-scale outdoor navigation


---

# License

This project is developed for research and educational purposes.


---

# Author

Ahmad

Robotics | ROS 2 | Navigation | Computer Vision