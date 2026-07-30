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

# This pretrained policy needs a small forward command
# to maintain a stable locomotion gait.
IDLE_FORWARD_SPEED = 0.1

# Return to idle if no ROS command is received for this duration.
COMMAND_TIMEOUT_SECONDS = 1.0


def get_gravity_orientation(quaternion):
    """Calculate the gravity direction in the robot body frame."""

    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3, dtype=np.float32)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculate joint torques from target positions and velocities."""

    return (
        (target_q - q) * kp
        + (target_dq - dq) * kd
    )


def load_config(config_file):
    """Load the Unitree MuJoCo configuration."""

    config_path = (
        f"{LEGGED_GYM_ROOT_DIR}/deploy/"
        f"deploy_mujoco/configs/{config_file}"
    )

    with open(config_path, "r", encoding="utf-8") as config_stream:
        config = yaml.load(
            config_stream,
            Loader=yaml.FullLoader,
        )

    config["policy_path"] = config["policy_path"].replace(
        "{LEGGED_GYM_ROOT_DIR}",
        LEGGED_GYM_ROOT_DIR,
    )

    config["xml_path"] = config["xml_path"].replace(
        "{LEGGED_GYM_ROOT_DIR}",
        LEGGED_GYM_ROOT_DIR,
    )

    return config


def create_udp_socket():
    """Create a non-blocking UDP receiver for ROS velocity commands."""

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
    """
    Receive the newest available UDP command.

    Returns:
        tuple[float, float, float] | None
    """

    latest_command = None

    while True:
        try:
            packet, _ = udp_socket.recvfrom(12)

            if len(packet) != 12:
                print(
                    f"Ignoring invalid UDP packet "
                    f"with size {len(packet)} bytes"
                )
                continue

            latest_command = struct.unpack(
                "fff",
                packet,
            )

        except BlockingIOError:
            break

    return latest_command


def main():
    parser = argparse.ArgumentParser(
        description="Run the Unitree H1 policy using ROS 2 UDP commands."
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Configuration filename inside deploy_mujoco/configs",
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

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

    cmd = np.array(
        [IDLE_FORWARD_SPEED, 0.0, 0.0],
        dtype=np.float32,
    )

    action = np.zeros(
        num_actions,
        dtype=np.float32,
    )

    target_dof_pos = default_angles.copy()

    obs = np.zeros(
        num_obs,
        dtype=np.float32,
    )

    print(f"Loading robot XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)
    model.opt.timestep = simulation_dt

    print(f"Loading policy: {policy_path}")
    policy = torch.jit.load(
        policy_path,
        map_location="cpu",
    )

    policy.eval()

    udp_socket = create_udp_socket()

    print(f"Initial command: {cmd}")
    print(
        f"Listening for ROS commands on "
        f"{UDP_HOST}:{UDP_PORT}"
    )

    counter = 0
    last_command_time = time.monotonic()
    command_timeout_active = False

    try:
        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            simulation_start_time = time.monotonic()

            while (
                viewer.is_running()
                and time.monotonic() - simulation_start_time
                < simulation_duration
            ):
                step_start_time = time.monotonic()

                latest_command = receive_latest_command(
                    udp_socket
                )

                if latest_command is not None:
                    linear_x, linear_y, angular_z = (
                        latest_command
                    )

                    cmd[:] = [
                        linear_x,
                        linear_y,
                        angular_z,
                    ]

                    last_command_time = time.monotonic()
                    command_timeout_active = False

                    print(
                        "ROS command received: "
                        f"x={cmd[0]:.2f}, "
                        f"y={cmd[1]:.2f}, "
                        f"yaw={cmd[2]:.2f}"
                    )

                time_since_last_command = (
                    time.monotonic()
                    - last_command_time
                )

                if (
                    time_since_last_command
                    > COMMAND_TIMEOUT_SECONDS
                ):
                    if not command_timeout_active:
                        cmd[:] = [
                            IDLE_FORWARD_SPEED,
                            0.0,
                            0.0,
                        ]

                        print(
                            "Command timeout: returning "
                            f"to idle gait {cmd}"
                        )

                        command_timeout_active = True

                target_velocity = np.zeros_like(kds)

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

                counter += 1

                if counter % control_decimation == 0:
                    joint_positions = data.qpos[7:].copy()
                    joint_velocities = data.qvel[6:].copy()

                    quaternion = data.qpos[3:7].copy()
                    angular_velocity = data.qvel[3:6].copy()

                    scaled_joint_positions = (
                        joint_positions - default_angles
                    ) * dof_pos_scale

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
                        counter * simulation_dt
                    )

                    phase = (
                        elapsed_simulation_time
                        % gait_period
                    ) / gait_period

                    sin_phase = np.sin(
                        2 * np.pi * phase
                    )

                    cos_phase = np.cos(
                        2 * np.pi * phase
                    )

                    obs[:3] = scaled_angular_velocity
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd * cmd_scale

                    obs[
                        9 : 9 + num_actions
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
                        [sin_phase, cos_phase],
                        dtype=np.float32,
                    )

                    obs_tensor = torch.from_numpy(
                        obs
                    ).unsqueeze(0)

                    with torch.no_grad():
                        action = (
                            policy(obs_tensor)
                            .cpu()
                            .numpy()
                            .squeeze()
                            .astype(np.float32)
                        )

                    target_dof_pos = (
                        action * action_scale
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
        print("Simulation stopped by user.")

    finally:
        udp_socket.close()
        print("UDP socket closed.")


if __name__ == "__main__":
    main()