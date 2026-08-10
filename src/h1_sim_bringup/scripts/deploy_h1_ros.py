#!/usr/bin/env python3

import argparse
import socket
import struct
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from legged_gym import LEGGED_GYM_ROOT_DIR

UDP_HOST = "127.0.0.1"
UDP_PORT = 15000


LIDAR_UDP_IP = "127.0.0.1"
LIDAR_UDP_PORT = 15001


# This policy needs a small forward command to remain stable.
IDLE_FORWARD_SPEED = 0.0975

# If ROS commands stop arriving, return to the idle gait.
COMMAND_TIMEOUT_SECONDS = 1.0

# Speed ramp limits.
# Linear velocity changes by up to 0.30 m/s every second.
# Turning velocity changes by up to 0.50 rad/s every second.
LINEAR_ACCELERATION = 0.25
ANGULAR_ACCELERATION = 0.25

# Command safety limits
MAX_FORWARD_SPEED = 2.50
MAX_BACKWARD_SPEED = 0.50
MAX_SIDE_SPEED = 0.30
MAX_YAW_SPEED = 0.50
# Print the actual robot speed once per second.
SPEED_PRINT_INTERVAL = 1.0

LIDAR_MIN_ANGLE = -np.pi
LIDAR_MAX_ANGLE = np.pi
LIDAR_NUM_RAYS = 360

LIDAR_MIN_RANGE = 0.10
LIDAR_MAX_RANGE = 30.0

LIDAR_UPDATE_INTERVAL = 0.10

# RGB-D camera UDP output.
CAMERA_UDP_IP = "127.0.0.1"
CAMERA_UDP_PORT = 15003
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15.0
CAMERA_UPDATE_INTERVAL = 1.0 / CAMERA_FPS
CAMERA_CHUNK_SIZE = 60000
CAMERA_PACKET_MAGIC = b"H1CM"
CAMERA_FRAME_TYPE_COLOR = 1
CAMERA_FRAME_TYPE_DEPTH = 2
CAMERA_PACKET_HEADER_FORMAT = "<4sBIII"


# Idle position-hold controller gains.
IDLE_POSITION_KP_X = 0.35
IDLE_POSITION_KP_Y = 0.10
IDLE_YAW_KP = 2.50

# Maximum idle corrections.
MAX_IDLE_X_CORRECTION = 0.08
MAX_IDLE_Y_CORRECTION = 0.03
MAX_IDLE_YAW_CORRECTION = 0.25

# Small constant lateral correction.
IDLE_Y_BIAS = 0.0

ODOM_UDP_IP = "127.0.0.1"
ODOM_UDP_PORT = 15002

ODOM_UPDATE_RATE = 30.0
ODOM_UPDATE_INTERVAL = 1.0 / ODOM_UPDATE_RATE


def get_gravity_orientation(quaternion):
    """Calculate gravity orientation in the robot body frame."""

    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(
        3,
        dtype=np.float32,
    )

    gravity_orientation[0] = 2 * (
        -qz * qx + qw * qy
    )

    gravity_orientation[1] = -2 * (
        qz * qy + qw * qx
    )

    gravity_orientation[2] = 1 - 2 * (
        qw * qw + qz * qz
    )

    return gravity_orientation


def pd_control(
    target_q,
    q,
    kp,
    target_dq,
    dq,
    kd,
):
    """Calculate joint torques using PD control."""

    return (
        (target_q - q) * kp
        + (target_dq - dq) * kd
    )


def load_config(config_file):
    """Load the H1 MuJoCo YAML configuration."""

    config_path = (
        f"{LEGGED_GYM_ROOT_DIR}/deploy/"
        f"deploy_mujoco/configs/{config_file}"
    )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as config_stream:
        config = yaml.load(
            config_stream,
            Loader=yaml.FullLoader,
        )

    config["policy_path"] = config[
        "policy_path"
    ].replace(
        "{LEGGED_GYM_ROOT_DIR}",
        LEGGED_GYM_ROOT_DIR,
    )

    config["xml_path"] = config[
        "xml_path"
    ].replace(
        "{LEGGED_GYM_ROOT_DIR}",
        LEGGED_GYM_ROOT_DIR,
    )

    return config


def create_udp_socket():
    """Create the non-blocking UDP command receiver."""

    udp_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    udp_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    udp_socket.bind(
        (UDP_HOST, UDP_PORT)
    )

    udp_socket.setblocking(False)

    return udp_socket


def receive_latest_command(udp_socket):
    """Read all waiting UDP packets and return the newest command."""

    latest_command = None

    while True:
        try:
            packet, _ = udp_socket.recvfrom(12)

            if len(packet) != 12:
                print(
                    "Ignoring UDP packet with "
                    f"invalid size: {len(packet)} bytes"
                )
                continue

            latest_command = struct.unpack(
                "fff",
                packet,
            )

        except BlockingIOError:
            break

    return latest_command


def send_camera_frame(
    udp_socket,
    frame_type,
    frame_id,
    frame_bytes,
):
    """Split one RGB or depth frame into safe UDP datagrams."""

    total_size = len(frame_bytes)
    chunk_count = (
        total_size + CAMERA_CHUNK_SIZE - 1
    ) // CAMERA_CHUNK_SIZE

    for chunk_index in range(chunk_count):
        start = chunk_index * CAMERA_CHUNK_SIZE
        end = min(
            start + CAMERA_CHUNK_SIZE,
            total_size,
        )

        header = struct.pack(
            CAMERA_PACKET_HEADER_FORMAT,
            CAMERA_PACKET_MAGIC,
            frame_type,
            frame_id,
            chunk_index,
            chunk_count,
        )

        udp_socket.sendto(
            header + frame_bytes[start:end],
            (CAMERA_UDP_IP, CAMERA_UDP_PORT),
        )


def get_yaw_from_quaternion(quaternion):
    qw = float(quaternion[0])
    qx = float(quaternion[1])
    qy = float(quaternion[2])
    qz = float(quaternion[3])

    return np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def wrap_angle(angle):
    return np.arctan2(
        np.sin(angle),
        np.cos(angle),
    )

def move_towards(
    current,
    target,
    maximum_change,
):
    """Move one value gradually towards its target."""

    difference = target - current

    if abs(difference) <= maximum_change:
        return target

    return (
        current
        + np.sign(difference) * maximum_change
    )


def generate_lidar_scan(
    model,
    data,
    lidar_site_id,
    excluded_body_id,
):
    """Generate a horizontal 360-degree 2D LiDAR scan."""

    lidar_position = data.site_xpos[
        lidar_site_id
    ].copy()

    # Keep the 2D scan horizontal by applying yaw only.
    robot_yaw = get_yaw_from_quaternion(
        data.qpos[3:7]
    )

    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)

    lidar_rotation = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ],
        dtype=np.float64,
    )

    angles = np.linspace(
        LIDAR_MIN_ANGLE,
        LIDAR_MAX_ANGLE,
        LIDAR_NUM_RAYS,
        endpoint=False,
    )

    ranges = np.full(
        LIDAR_NUM_RAYS,
        LIDAR_MAX_RANGE,
        dtype=np.float32,
    )

    geom_group = np.array(
        [1, 0, 0, 0, 0, 0],
        dtype=np.uint8,
    )

    for index, angle in enumerate(angles):
        local_direction = np.array(
            [
                np.cos(angle),
                np.sin(angle),
                0.0,
            ],
            dtype=np.float64,
        )

        world_direction = (
            lidar_rotation
            @ local_direction
        )

        hit_geom_id = np.array(
            [-1],
            dtype=np.int32,
        )

        distance = mujoco.mj_ray(
            model,
            data,
            lidar_position,
            world_direction,
            geom_group,
            1,
            excluded_body_id,
            hit_geom_id,
        )

        if (
            LIDAR_MIN_RANGE
            <= distance
            <= LIDAR_MAX_RANGE
        ):
            ranges[index] = float(distance)

    return ranges


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the H1 policy using ROS 2 "
            "UDP velocity commands."
        )
    )

    parser.add_argument(
        "config_file",
        type=str,
        help=(
            "YAML filename inside "
            "deploy_mujoco/configs"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config_file
    )

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    simulation_duration = float(
        config["simulation_duration"]
    )

    simulation_dt = float(
        config["simulation_dt"]
    )

    control_decimation = int(
        config["control_decimation"]
    )

    kps = np.array(
        config["kps"],
        dtype=np.float32,
    )

    kds = np.array(
        config["kds"],
        dtype=np.float32,
    )

    default_angles = np.array(
        config["default_angles"],
        dtype=np.float32,
    )

    ang_vel_scale = float(
        config["ang_vel_scale"]
    )

    dof_pos_scale = float(
        config["dof_pos_scale"]
    )

    dof_vel_scale = float(
        config["dof_vel_scale"]
    )

    action_scale = float(
        config["action_scale"]
    )

    cmd_scale = np.array(
        config["cmd_scale"],
        dtype=np.float32,
    )

    num_actions = int(
        config["num_actions"]
    )

    num_obs = int(
        config["num_obs"]
    )

    # The policy receives current_cmd.
    current_cmd = np.array(
        [
            IDLE_FORWARD_SPEED,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    # ROS commands update target_cmd.
    target_cmd = current_cmd.copy()
    
    # False means the RL walking policy is active.

    stand_mode = True
    previous_stand_mode = None

    idle_reference_x = None
    idle_reference_y = None
    idle_reference_yaw = None

    action = np.zeros(
        num_actions,
        dtype=np.float32,
    )
    target_dof_pos = default_angles.copy()

    obs = np.zeros(
        num_obs,
        dtype=np.float32,
    )

    print(
        f"Loading robot model: {xml_path}"
    )

    model = mujoco.MjModel.from_xml_path(
        xml_path
    )

    print("Geometry groups:")

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        geom_group_id = int(
            model.geom_group[geom_id]
        )

        if geom_name:
            print(
                f"{geom_name}: group={geom_group_id}"
            )

    
    data = mujoco.MjData(model)

    camera_color_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "camera_color",
    )

    camera_depth_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "camera_depth",
    )

    if camera_color_id == -1 or camera_depth_id == -1:
        raise RuntimeError(
            "MuJoCo cameras 'camera_color' and "
            "'camera_depth' were not found."
        )

    camera_renderer = mujoco.Renderer(
        model,
        height=CAMERA_HEIGHT,
        width=CAMERA_WIDTH,
    )

    print(
        "RGB-D cameras ready | "
        f"color_id={camera_color_id} | "
        f"depth_id={camera_depth_id} | "
        f"resolution={CAMERA_WIDTH}x{CAMERA_HEIGHT} | "
        f"rate={CAMERA_FPS:.1f} Hz"
    )
    
    lidar_sit_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "lidar_site",
    )
    if lidar_sit_id == -1:
        raise RuntimeError(
            "MuJoCo site 'lidar_site' was not found."
        )

    # print(
    #     f"LiDAR site found with ID: {lidar_sit_id}"
    # )
    
    pelvis_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    if pelvis_body_id == -1:
        raise RuntimeError(
            "MuJoCo body 'pelvis' was not found."
        )

    print(
        f"Pelvis body found with ID: {pelvis_body_id}"
    )
    
    gyro_sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "pelvis_gyro",
    )

    accelerometer_sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "pelvis_accelerometer",
    )

    orientation_sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        "pelvis_orientation",
    )

    if (
        gyro_sensor_id == -1
        or accelerometer_sensor_id == -1
        or orientation_sensor_id == -1
    ):
        raise RuntimeError(
            "One or more MuJoCo IMU sensors were not found."
        )

    gyro_address = int(
        model.sensor_adr[gyro_sensor_id]
    )

    accelerometer_address = int(
        model.sensor_adr[accelerometer_sensor_id]
    )

    orientation_address = int(
        model.sensor_adr[orientation_sensor_id]
    )

    print(
        "IMU sensors found | "
        f"gyro={gyro_sensor_id}, "
        f"accelerometer={accelerometer_sensor_id}, "
        f"orientation={orientation_sensor_id}"
    )
    
    model.opt.timestep = simulation_dt

    print(
        f"Loading policy: {policy_path}"
    )

    policy = torch.jit.load(
        policy_path,
        map_location="cpu",
    )

    policy.eval()

    udp_socket = create_udp_socket()

    # LiDAR UDP sender socket
    lidar_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    # RGB-D camera UDP sender socket.
    camera_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    camera_socket.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_SNDBUF,
        4 * 1024 * 1024,
    )

    # Odometry UDP sender socket
    odom_socket = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    
    print(
        f"Initial command: {current_cmd}"
    )
    
    print(
        "Listening for ROS commands on "
        f"{UDP_HOST}:{UDP_PORT}"
    )

    counter = 0
    last_lidar_update_time = time.monotonic()
    last_camera_update_time = time.monotonic()
    last_odom_update_time = time.monotonic()
    camera_frame_id = 0
    last_command_time = time.monotonic()
    
    command_timeout_active = False

    # Variables used to measure actual movement.
    last_speed_print_time = time.monotonic()
    previous_position_time = time.monotonic()
    previous_position = data.qpos[:2].copy()

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            simulation_start_time = (
                time.monotonic()
            )
            previous_stand_mode = None
            while (
                viewer.is_running()
                and (
                    time.monotonic()
                    - simulation_start_time
                ) < simulation_duration
            ):
                step_start_time = time.monotonic()

                # -------------------------------------
                # 1. Receive the newest ROS command
                # -------------------------------------
                latest_command = (
                    receive_latest_command(
                        udp_socket
                    )
                )

                if latest_command is not None:
                    (
                        linear_x,
                        linear_y,
                        angular_z,
                    ) = latest_command

                    movement_requested = (
                        abs(linear_x) > 0.01
                        or abs(linear_y) > 0.01
                        or abs(angular_z) > 0.01
                    )

                    stand_mode = not movement_requested

                    if stand_mode != previous_stand_mode:
                        print(
                            "Controller mode: "
                            + (
                                "IDLE GAIT"
                                if stand_mode
                                else "WALK"
                            )
                        )

                        previous_stand_mode = stand_mode

                    if stand_mode:
                        robot_x = float(data.qpos[0])
                        robot_y = float(data.qpos[1])

                        robot_yaw = get_yaw_from_quaternion(
                            data.qpos[3:7]
                        )

                        if idle_reference_x is None:
                            idle_reference_x = robot_x
                            idle_reference_y = robot_y
                            idle_reference_yaw = robot_yaw

                        world_error_x = (
                            idle_reference_x - robot_x
                        )

                        world_error_y = (
                            idle_reference_y - robot_y
                        )

                        cos_yaw = np.cos(robot_yaw)
                        sin_yaw = np.sin(robot_yaw)

                        body_error_x = (
                            cos_yaw * world_error_x
                            + sin_yaw * world_error_y
                        )

                        body_error_y = (
                            -sin_yaw * world_error_x
                            + cos_yaw * world_error_y
                        )

                        yaw_error = wrap_angle(
                            idle_reference_yaw - robot_yaw
                        )

                        x_correction = np.clip(
                            IDLE_POSITION_KP_X * body_error_x,
                            -MAX_IDLE_X_CORRECTION,
                            MAX_IDLE_X_CORRECTION,
                        )

                        y_correction = np.clip(
                            IDLE_POSITION_KP_Y * body_error_y,
                            -MAX_IDLE_Y_CORRECTION,
                            MAX_IDLE_Y_CORRECTION,
                        )

                        yaw_correction = np.clip(
                            IDLE_YAW_KP * yaw_error,
                            -MAX_IDLE_YAW_CORRECTION,
                            MAX_IDLE_YAW_CORRECTION,
                        )

                        target_cmd[:] = [
                            IDLE_FORWARD_SPEED + x_correction,
                            IDLE_Y_BIAS + y_correction,
                            yaw_correction,
                        ]

                    else:
                        idle_reference_x = None
                        idle_reference_y = None
                        idle_reference_yaw = None

                        target_cmd[:] = [
                            np.clip(
                                linear_x,
                                -MAX_BACKWARD_SPEED,
                                MAX_FORWARD_SPEED,
                            ),
                            np.clip(
                                linear_y,
                                -MAX_SIDE_SPEED,
                                MAX_SIDE_SPEED,
                            ),
                            np.clip(
                                angular_z,
                                -MAX_YAW_SPEED,
                                MAX_YAW_SPEED,
                            ),
                        ]
                    last_command_time = time.monotonic()
                    command_timeout_active = False
                # -------------------------------------
                # 2. Check ROS command timeout
                # -------------------------------------
                time_since_last_command = (
                    time.monotonic()
                    - last_command_time
                )

                if (
                    time_since_last_command
                    > COMMAND_TIMEOUT_SECONDS
                    and not command_timeout_active
                ):
                    stand_mode = True

                    idle_reference_x = float(data.qpos[0])
                    idle_reference_y = float(data.qpos[1])
                    idle_reference_yaw = get_yaw_from_quaternion(
                        data.qpos[3:7]
                    )

                    print(
                        "Command timeout: entering "
                        "idle position hold"
                    )

                    command_timeout_active = True          
                        
                # -------------------------------------
                # 3. Apply command speed ramp
                # -------------------------------------
                
                # -------------------------------------
                
                # Continuous idle position and yaw hold
                # -------------------------------------
                if stand_mode:
                    robot_x = float(data.qpos[0])
                    robot_y = float(data.qpos[1])

                    robot_yaw = get_yaw_from_quaternion(
                        data.qpos[3:7]
                    )

                    if idle_reference_x is None:
                        idle_reference_x = robot_x
                        idle_reference_y = robot_y
                        idle_reference_yaw = robot_yaw

                    world_error_x = (
                        idle_reference_x - robot_x
                    )

                    world_error_y = (
                        idle_reference_y - robot_y
                    )

                    cos_yaw = np.cos(robot_yaw)
                    sin_yaw = np.sin(robot_yaw)

                    body_error_x = (
                        cos_yaw * world_error_x
                        + sin_yaw * world_error_y
                    )

                    body_error_y = (
                        -sin_yaw * world_error_x
                        + cos_yaw * world_error_y
                    )

                    yaw_error = wrap_angle(
                        idle_reference_yaw - robot_yaw
                    )

                    x_correction = np.clip(
                        IDLE_POSITION_KP_X * body_error_x,
                        -MAX_IDLE_X_CORRECTION,
                        MAX_IDLE_X_CORRECTION,
                    )

                    y_correction = np.clip(
                        IDLE_POSITION_KP_Y * body_error_y,
                        -MAX_IDLE_Y_CORRECTION,
                        MAX_IDLE_Y_CORRECTION,
                    )

                    yaw_correction = np.clip(
                        IDLE_YAW_KP * yaw_error,
                        -MAX_IDLE_YAW_CORRECTION,
                        MAX_IDLE_YAW_CORRECTION,
                    )

                    target_cmd[:] = [
                        IDLE_FORWARD_SPEED + x_correction,
                        IDLE_Y_BIAS + y_correction,
                        yaw_correction,
                    ]
                
                linear_step = (
                    LINEAR_ACCELERATION
                    * model.opt.timestep
                )

                angular_step = (
                    ANGULAR_ACCELERATION
                    * model.opt.timestep
                )

                current_cmd[0] = move_towards(
                    current_cmd[0],
                    target_cmd[0],
                    linear_step,
                )

                current_cmd[1] = move_towards(
                    current_cmd[1],
                    target_cmd[1],
                    linear_step,
                )

                current_cmd[2] = move_towards(
                    current_cmd[2],
                    target_cmd[2],
                    angular_step,
                )

                # Print ramp status approximately
                # once per simulated second.
                # if counter % 500 == 0:
                #     print(
                #         "Ramp status | "
                #         f"current: "
                #         f"x={current_cmd[0]:.3f}, "
                #         f"y={current_cmd[1]:.3f}, "
                #         f"yaw={current_cmd[2]:.3f} | "
                #         f"target: "
                #         f"x={target_cmd[0]:.3f}, "
                #         f"y={target_cmd[1]:.3f}, "
                #         f"yaw={target_cmd[2]:.3f}"
                #     )

                # -------------------------------------
                # 4. PD control and MuJoCo step
                # -------------------------------------
                target_velocity = np.zeros_like(
                    kds
                )

                tau = pd_control(
                    target_dof_pos,
                    data.qpos[7:],
                    kps,
                    target_velocity,
                    data.qvel[6:],
                    kds,
                )

                data.ctrl[:] = tau

                mujoco.mj_step(
                    model,
                    data,
                )
                # Simulation-only idle position lock.
                if (
                    stand_mode
                    and idle_reference_x is not None
                    and idle_reference_y is not None
                ):
                    data.qpos[0] = idle_reference_x
                    data.qpos[1] = idle_reference_y

                    data.qvel[0] = 0.0
                    data.qvel[1] = 0.0

                    mujoco.mj_forward(
                        model,
                        data,
                    )

                # -------------------------------------
                current_time = time.monotonic()   
                
                # -------------------------------------
                # Publish MuJoCo odometry through UDP
                # -------------------------------------
                if (
                    current_time - last_odom_update_time
                    >= ODOM_UPDATE_INTERVAL
                ):
                    position_x = float(data.qpos[0])
                    position_y = float(data.qpos[1])
                    position_z = float(data.qpos[2])

                    # MuJoCo free-joint quaternion order:
                    # w, x, y, z
                    quaternion_w = float(data.qpos[3])
                    quaternion_x = float(data.qpos[4])
                    quaternion_y = float(data.qpos[5])
                    quaternion_z = float(data.qpos[6])

                    world_velocity_x = float(data.qvel[0])
                    world_velocity_y = float(data.qvel[1])
                    world_velocity_z = float(data.qvel[2])

                    angular_velocity_x = float(data.qvel[3])
                    angular_velocity_y = float(data.qvel[4])
                    angular_velocity_z = float(data.qvel[5])

                    robot_yaw = get_yaw_from_quaternion(
                        data.qpos[3:7]
                    )

                    cos_yaw = np.cos(robot_yaw)
                    sin_yaw = np.sin(robot_yaw)

                    # Convert world-frame planar velocity
                    # into pelvis/body-frame velocity.
                    body_velocity_x = (
                        cos_yaw * world_velocity_x
                        + sin_yaw * world_velocity_y
                    )

                    body_velocity_y = (
                        -sin_yaw * world_velocity_x
                        + cos_yaw * world_velocity_y
                    )

                    gyro = data.sensordata[
                        gyro_address:gyro_address + 3
                    ]

                    accelerometer = data.sensordata[
                        accelerometer_address:
                        accelerometer_address + 3
                    ]

                    imu_orientation = data.sensordata[
                        orientation_address:
                        orientation_address + 4
                    ]

                    # MuJoCo framequat order is w, x, y, z.
                    imu_quaternion_w = float(
                        imu_orientation[0]
                    )
                    imu_quaternion_x = float(
                        imu_orientation[1]
                    )
                    imu_quaternion_y = float(
                        imu_orientation[2]
                    )
                    imu_quaternion_z = float(
                        imu_orientation[3]
                    )

                    odom_packet = struct.pack(
                        "23f",

                        # Odometry position: 3
                        position_x,
                        position_y,
                        position_z,

                        # Odometry orientation: 4
                        quaternion_x,
                        quaternion_y,
                        quaternion_z,
                        quaternion_w,

                        # Odometry linear velocity: 3
                        body_velocity_x,
                        body_velocity_y,
                        world_velocity_z,

                        # Odometry angular velocity: 3
                        angular_velocity_x,
                        angular_velocity_y,
                        angular_velocity_z,

                        # IMU gyro: 3
                        float(gyro[0]),
                        float(gyro[1]),
                        float(gyro[2]),

                        # IMU accelerometer: 3
                        float(accelerometer[0]),
                        float(accelerometer[1]),
                        float(accelerometer[2]),

                        # IMU orientation: 4
                        imu_quaternion_x,
                        imu_quaternion_y,
                        imu_quaternion_z,
                        imu_quaternion_w,
                    )
                    
                    odom_socket.sendto(
                        odom_packet,
                        (
                            ODOM_UDP_IP,
                            ODOM_UDP_PORT,
                        ),
                    )

                    last_odom_update_time = current_time

                if (
                    current_time
                    - last_camera_update_time
                    >= CAMERA_UPDATE_INTERVAL
                ):
                    camera_renderer.update_scene(
                        data,
                        camera="camera_color",
                    )

                    color_image = np.ascontiguousarray(
                        camera_renderer.render(),
                        dtype=np.uint8,
                    )

                    camera_renderer.update_scene(
                        data,
                        camera="camera_depth",
                    )
                    camera_renderer.enable_depth_rendering()

                    depth_image = np.ascontiguousarray(
                        camera_renderer.render(),
                        dtype=np.float32,
                    )

                    camera_renderer.disable_depth_rendering()

                    send_camera_frame(
                        camera_socket,
                        CAMERA_FRAME_TYPE_COLOR,
                        camera_frame_id,
                        color_image.tobytes(order="C"),
                    )

                    send_camera_frame(
                        camera_socket,
                        CAMERA_FRAME_TYPE_DEPTH,
                        camera_frame_id,
                        depth_image.tobytes(order="C"),
                    )

                    camera_frame_id += 1
                    last_camera_update_time = current_time

                if (
                    current_time
                    - last_lidar_update_time
                    >= LIDAR_UPDATE_INTERVAL
                ):
                    lidar_ranges = generate_lidar_scan(
                        model,
                        data,
                        lidar_sit_id,
                        pelvis_body_id,
                    )
                    
                    # Print robot and LiDAR world positions
                    robot_x = float(data.qpos[0])
                    robot_y = float(data.qpos[1])

                    lidar_x = float(
                        data.site_xpos[lidar_sit_id][0]
                    )

                    lidar_y = float(
                        data.site_xpos[lidar_sit_id][1]
                    )

                    # print(
                    #     f"Position | "
                    #     f"robot=({robot_x:.2f}, {robot_y:.2f}), "
                    #     f"lidar=({lidar_x:.2f}, {lidar_y:.2f})"
                    # )

                    
                    lidar_packet = struct.pack(
                        f"{LIDAR_NUM_RAYS}f",
                        *lidar_ranges,
                    )

                    lidar_socket.sendto(
                        lidar_packet,
                        (
                            LIDAR_UDP_IP,
                            LIDAR_UDP_PORT,
                        ),
                    )
                    
                    front_distance = float(
                        np.min(lidar_ranges[165:196])
                    )

                    left_distance = float(
                        np.min(lidar_ranges[265:276])
                    )

                    right_distance = float(
                        np.min(lidar_ranges[85:96])
                    )

                    back_indices = np.concatenate(
                        (
                            lidar_ranges[0:6],
                            lidar_ranges[355:360],
                        )
                    )

                    back_distance = float(
                        np.min(back_indices)
                    )
                    # print(
                    #     "LiDAR scan | "
                    #     f"front={front_distance:.2f} m, "
                    #     f"left={left_distance:.2f} m, "
                    #     f"right={right_distance:.2f} m, "
                    #     f"back={back_distance:.2f} m"
                    # )                  
                    last_lidar_update_time = current_time
                    
                    
                counter += 1

                # -------------------------------------
                # 5. Measure actual robot speed
                # -------------------------------------
                current_time = time.monotonic()

                if (
                    current_time
                    - last_speed_print_time
                    >= SPEED_PRINT_INTERVAL
                ):
                    current_position = (
                        data.qpos[:2].copy()
                    )

                    elapsed_time = (
                        current_time
                        - previous_position_time
                    )

                    position_change = (
                        current_position
                        - previous_position
                    )

                    if elapsed_time > 0:
                        actual_velocity_x = float(
                            position_change[0]
                            / elapsed_time
                        )

                        actual_velocity_y = float(
                            position_change[1]
                            / elapsed_time
                        )
                    else:
                        actual_velocity_x = 0.0
                        actual_velocity_y = 0.0

                    actual_yaw_rate = float(data.qvel[5])
                    
                    actual_speed = float(
                        np.hypot(
                            actual_velocity_x,
                            actual_velocity_y,
                        )
                    )

                    actual_speed_kmh = (
                        actual_speed * 3.6
                    )

                    # print(
                    #     "Speed measurement | "
                    #     f"commanded: "
                    #     f"x={current_cmd[0]:.2f} m/s, "
                    #     f"y={current_cmd[1]:.2f} m/s, "
                    #     f"yaw={current_cmd[2]:.2f} rad/s | "
                    #     f"actual: "
                    #     f"x={actual_velocity_x:.2f} m/s, "
                    #     f"y={actual_velocity_y:.2f} m/s, "
                    #     f"yaw={actual_yaw_rate:.2f} rad/s, "
                    #     f"total={actual_speed:.2f} m/s "
                    #     f"({actual_speed_kmh:.2f} km/h)"
                    # )                    
                    previous_position = (
                        current_position
                    )

                    previous_position_time = (
                        current_time
                    )

                    last_speed_print_time = (
                        current_time
                    )

                # -------------------------------------
                # 6. Create policy observation
                # -------------------------------------
                if (
                    counter
                    % control_decimation
                    == 0
                ):
                    joint_positions = (
                        data.qpos[7:].copy()
                    )

                    joint_velocities = (
                        data.qvel[6:].copy()
                    )

                    quaternion = (
                        data.qpos[3:7].copy()
                    )

                    angular_velocity = (
                        data.qvel[3:6].copy()
                    )

                    scaled_joint_positions = (
                        (
                            joint_positions
                            - default_angles
                        )
                        * dof_pos_scale
                    )

                    scaled_joint_velocities = (
                        joint_velocities
                        * dof_vel_scale
                    )

                    gravity_orientation = (
                        get_gravity_orientation(
                            quaternion
                        )
                    )

                    scaled_angular_velocity = (
                        angular_velocity
                        * ang_vel_scale
                    )

                    gait_period = 0.8

                    elapsed_simulation_time = (
                        counter
                        * simulation_dt
                    )

                    phase = (
                        (
                            elapsed_simulation_time
                            % gait_period
                        )
                        / gait_period
                    )

                    sin_phase = np.sin(
                        2 * np.pi * phase
                    )

                    cos_phase = np.cos(
                        2 * np.pi * phase
                    )

                    obs[:3] = (
                        scaled_angular_velocity
                    )

                    obs[3:6] = (
                        gravity_orientation
                    )

                    obs[6:9] = (
                        current_cmd
                        * cmd_scale
                    )

                    obs[
                        9 :
                        9 + num_actions
                    ] = scaled_joint_positions

                    obs[
                        9 + num_actions :
                        9 + 2 * num_actions
                    ] = scaled_joint_velocities

                    obs[
                        9 + 2 * num_actions :
                        9 + 3 * num_actions
                    ] = action

                    obs[
                        9 + 3 * num_actions :
                        9 + 3 * num_actions + 2
                    ] = np.array(
                        [
                            sin_phase,
                            cos_phase,
                        ],
                        dtype=np.float32,
                    )

                    obs_tensor = (
                        torch.from_numpy(obs)
                        .unsqueeze(0)
                    )

                    # Always run the RL policy because it performs
                    # the active balancing of the humanoid.
                    with torch.no_grad():
                        action = (
                            policy(obs_tensor)
                            .cpu()
                            .numpy()
                            .squeeze()
                            .astype(np.float32)
                        )

                    target_dof_pos = (
                        action
                        * action_scale
                        + default_angles
                    )                        
                        
                viewer.sync()

                remaining_step_time = (
                    model.opt.timestep
                    - (
                        time.monotonic()
                        - step_start_time
                    )
                )

                if remaining_step_time > 0:
                    time.sleep(
                        remaining_step_time
                    )

    except KeyboardInterrupt:
        print("Simulation stopped.")

    finally:
        udp_socket.close()
        lidar_socket.close()
        camera_socket.close()
        odom_socket.close()
        camera_renderer.close()
        print("UDP sockets and camera renderer closed.")


if __name__ == "__main__":
    main()
