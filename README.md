# Unitree H1 Humanoid Navigation with ROS 2 Nav2 MPPI and RGB-D Voxel Perception

A ROS 2 Jazzy based humanoid navigation framework for the **Unitree H1 robot** in **MuJoCo simulation**, integrating:

- Nav2 MPPI controller
- LiDAR-based obstacle navigation
- RGB-D perception
- 3D voxel obstacle representation
- Persistent RGB-D 3D reconstruction
- Multi-goal race navigation framework

The project explores autonomous navigation of a humanoid robot by combining classical navigation methods with RGB-D based environmental understanding.

---

# Demonstrations

## RGB-D 3D Reconstruction

Coming soon.

The robot builds a persistent 3D representation of the simulated environment using RGB-D perception.

Pipeline:

```
RGB-D Camera
      |
      v
Depth Point Cloud Generation
      |
      v
RTABMap Point Cloud Processing
      |
      v
Persistent 3D Reconstruction
```

---

## Nav2 MPPI Navigation with RGB-D VoxelLayer

Coming soon.

The humanoid robot performs obstacle-aware navigation using:

```
RGB-D Camera
      |
      v
PointCloud2
      |
      v
Nav2 VoxelLayer
      |
      v
Local Costmap
      |
      v
MPPI Controller
      |
      v
Humanoid Motion
```

---

# System Overview

The complete navigation architecture:

```
                 MuJoCo Simulation
                       |
                       |
                 Unitree H1 Robot
                       |
        --------------------------------
        |                              |
     LiDAR Sensor                 RGB-D Camera
        |                              |
      /scan                  Depth Point Cloud
        |                              |
        |                         VoxelLayer
        |                              |
        ----------- Local Costmap -------
                       |
                       |
                 Nav2 MPPI Controller
                       |
                       |
                    /cmd_vel
                       |
                       |
              H1 Locomotion Bridge
```

---

# Features

## Humanoid Navigation

- Unitree H1 humanoid robot simulation
- ROS 2 Jazzy integration
- MuJoCo based robot environment
- Autonomous waypoint navigation
- Race-style multi-goal navigation

## Nav2 Integration

- MPPI controller
- Local costmap navigation
- Dynamic obstacle avoidance
- LiDAR obstacle layer
- RGB-D VoxelLayer integration

## RGB-D and Semantic Perception

- Depth camera simulation
- PointCloud2 generation
- 3D voxel obstacle representation
- Persistent environment reconstruction
- Semantic object detection and depth fusion

## Sensor Fusion

The system combines:

```
LiDAR
 +
RGB-D Camera
      |
      v
Depth + RGB Processing
      |
      +------ YOLO Semantic Detection
      |
      v
Semantic Depth Fusion
      |
      v
Navigation / Environment Understanding

```

---

# Software Requirements

## Operating System

Tested on:

```
Ubuntu 22.04
```

## ROS

```
ROS 2 Jazzy
```

## Main Dependencies

- Nav2
- RViz2
- MuJoCo
- RTAB-Map utilities
- CycloneDDS
- Python 3

---

# Workspace Setup

Create ROS 2 workspace:

```bash
mkdir -p ~/humanoid_race_ws/src

cd ~/humanoid_race_ws/src
```

Clone repository:

```bash
git clone <repository-url>
```

---

# Install Dependencies

From workspace:

```bash
cd ~/humanoid_race_ws

rosdep install \
--from-paths src \
--ignore-src \
-r -y
```

---

# Build

Build the workspace:

```bash
colcon build
```

Source:

```bash
source install/setup.bash
```

---

# Running the Simulation

Set ROS domain:

```bash
export ROS_DOMAIN_ID=5
```

Launch the complete system:

```bash
ros2 launch humanoid_race_bringup full_mppi_race.launch.py
```

This starts:

- Unitree H1 simulation
- Robot state publisher
- Sensor bridges
- TF system
- Nav2 stack
- MPPI controller
- RGB-D perception pipeline
- 3D reconstruction pipeline

---

# RGB-D Reconstruction Pipeline

The RGB-D reconstruction system generates a persistent 3D environment model.

## Data Flow

```
Camera Depth Data

        |
        v

/camera/depth/points_realistic

        |
        v

RTABMap Point Cloud Processing

        |
        v

/reconstruction/cloud_assembled
```

## Visualization in RViz

Add:

```
/reconstruction/cloud_assembled
```

Display type:

```
PointCloud2
```

Frame:

```
world
```

---

# Navigation Perception Pipeline

## LiDAR Navigation

LiDAR provides 2D obstacle information.

Topic:

```
/scan
```

Pipeline:

```
LaserScan
    |
    v
ObstacleLayer
    |
    v
Local Costmap
```

---

## RGB-D Voxel Navigation

RGB-D provides 3D obstacle information.

Topic:

```
/camera/depth/points_realistic
```

Pipeline:

```
PointCloud2

      |
      v

Nav2 VoxelLayer

      |
      v

3D Occupancy Voxels

      |
      v

Local Costmap

      |
      v

MPPI Controller
```

---

# Important ROS Topics

## Sensors

| Topic | Description |
|---|---|
| `/scan` | LiDAR LaserScan |
| `/camera/depth/points_realistic` | RGB-D generated point cloud |

---

## Navigation

| Topic | Description |
|---|---|
| `/local_costmap/costmap` | 2D navigation costmap |
| `/local_costmap/voxel_grid` | 3D voxel representation |
| `/cmd_vel` | Robot velocity command |

---

## Reconstruction

| Topic | Description |
|---|---|
| `/reconstruction/cloud_assembled` | Persistent 3D reconstructed cloud |

---

# RViz Visualization

Recommended displays:

## Navigation View

```
/scan

/local_costmap/costmap

/local_costmap/voxel_grid

MPPI trajectory
```

---

## Reconstruction View

```
/reconstruction/cloud_assembled
```

---

# Results

The system successfully demonstrates:

✅ Unitree H1 autonomous navigation  
✅ ROS 2 Nav2 MPPI control  
✅ LiDAR obstacle avoidance  
✅ RGB-D obstacle perception  
✅ Nav2 VoxelLayer integration  
✅ Dynamic obstacle marking and clearing  
✅ Persistent RGB-D 3D reconstruction  
✅ Multi-goal navigation framework  

---

# Project Structure

```
humanoid_race_ws/

├── README.md

├── src/

│   ├── h1_locomotion_bridge

│   └── humanoid_race_bringup

├── docs/

│   ├── images/

│   └── videos/

└── maps/

```

---

# Future Work

Potential improvements:

- Deployment on real Unitree H1 hardware
- Learning-based navigation
- Footstep-aware humanoid planning
- Large-scale outdoor navigation
- Real-world RGB-D validation

---

# License

This project is released for research and educational purposes.

---

# Acknowledgements

Built using:

- ROS 2
- Nav2
- MuJoCo
- Unitree Robotics
- RTAB-Map
- Open-source robotics community