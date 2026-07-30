#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class VfhConfig:
    # Dynamic speed limits
    min_forward_speed: float = 0.15
    max_forward_speed: float = 2.00

    # Acceleration is slower than braking.
    acceleration_rate: float = 0.20
    deceleration_rate: float = 2.50

    max_yaw_rate: float = 0.80

    # Obstacle distances
    stop_distance: float = 1.20
    safe_distance: float = 6.00

    # VFH geometry
    sector_size_deg: float = 5.0
    front_fov_deg: float = 180.0
    robot_radius: float = 0.40
    safety_margin: float = 0.20

    # Road centre recovery
    centre_line_y: float = 0.0
    centre_lookahead: float = 3.0
    max_recovery_angle_deg: float = 35.0

    # Direction selection
    goal_weight: float = 2.0
    smooth_weight: float = 1.2
    density_weight: float = 0.8

    # Safety
    scan_timeout: float = 0.50
    odom_timeout: float = 0.50


@dataclass(frozen=True)
class VfhResult:
    linear_x: float
    angular_z: float
    steering_angle: float
    front_clearance: float
    emergency_stop: bool


def normalise_angle(angle_rad: float) -> float:
    return (
        angle_rad + math.pi
    ) % (
        2.0 * math.pi
    ) - math.pi


def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    sin_yaw = 2.0 * (
        w * z + x * y
    )

    cos_yaw = 1.0 - 2.0 * (
        y * y + z * z
    )

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def sanitise_scan(
    ranges: Sequence[float],
    range_min: float,
    range_max: float,
) -> np.ndarray:
    values = np.asarray(
        ranges,
        dtype=np.float64,
    ).copy()

    invalid = (
        ~np.isfinite(values)
        | (values < range_min)
        | (values > range_max)
    )

    values[invalid] = range_max

    return values


def extract_front_scan(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    front_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(
        ranges.size,
        dtype=np.float64,
    )

    angles = (
        angle_min
        + indices * angle_increment
    )

    half_fov = math.radians(
        front_fov_deg / 2.0
    )

    mask = (
        (angles >= -half_fov)
        & (angles <= half_fov)
    )

    return (
        angles[mask],
        ranges[mask],
    )


def build_sector_histogram(
    angles: np.ndarray,
    ranges: np.ndarray,
    config: VfhConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    half_fov_deg = (
        config.front_fov_deg / 2.0
    )

    edges_deg = np.arange(
        -half_fov_deg,
        half_fov_deg
        + config.sector_size_deg,
        config.sector_size_deg,
        dtype=np.float64,
    )

    centres_deg = (
        edges_deg[:-1]
        + edges_deg[1:]
    ) / 2.0

    centres = np.radians(
        centres_deg
    )

    density = np.zeros(
        centres.size,
        dtype=np.float64,
    )

    minimum_ranges = np.full(
        centres.size,
        np.inf,
        dtype=np.float64,
    )

    angles_deg = np.degrees(
        angles
    )

    for angle_deg, distance in zip(
        angles_deg,
        ranges,
    ):
        sector_index = int(
            math.floor(
                (
                    angle_deg
                    + half_fov_deg
                )
                / config.sector_size_deg
            )
        )

        sector_index = int(
            np.clip(
                sector_index,
                0,
                centres.size - 1,
            )
        )

        minimum_ranges[sector_index] = min(
            minimum_ranges[sector_index],
            float(distance),
        )

        if distance < config.safe_distance:
            obstacle_density = (
                config.safe_distance
                - distance
            ) / config.safe_distance

            density[sector_index] = max(
                density[sector_index],
                obstacle_density,
            )

    minimum_ranges[
        ~np.isfinite(minimum_ranges)
    ] = config.safe_distance * 10.0

    return (
        centres,
        density,
        minimum_ranges,
    )


def inflate_blocked_sectors(
    centres: np.ndarray,
    minimum_ranges: np.ndarray,
    config: VfhConfig,
) -> np.ndarray:
    blocked = np.zeros(
        centres.size,
        dtype=bool,
    )

    effective_radius = (
        config.robot_radius
        + config.safety_margin
    )

    for index, distance in enumerate(
        minimum_ranges
    ):
        if distance >= config.safe_distance:
            continue

        safe_distance = max(
            float(distance),
            effective_radius + 0.01,
        )

        ratio = np.clip(
            effective_radius / safe_distance,
            0.0,
            1.0,
        )

        inflation_angle = math.asin(
            ratio
        )

        angular_difference = np.abs(
            np.array(
                [
                    normalise_angle(
                        candidate
                        - centres[index]
                    )
                    for candidate in centres
                ]
            )
        )

        blocked |= (
            angular_difference
            <= inflation_angle
        )

    return blocked


def compute_desired_heading(
    robot_y: float,
    robot_yaw: float,
    config: VfhConfig,
) -> float:
    centre_error_y = (
        config.centre_line_y
        - robot_y
    )

    desired_world_heading = math.atan2(
        centre_error_y,
        config.centre_lookahead,
    )

    desired_relative_heading = (
        desired_world_heading
        - robot_yaw
    )

    max_recovery_angle = math.radians(
        config.max_recovery_angle_deg
    )

    return float(
        np.clip(
            normalise_angle(
                desired_relative_heading
            ),
            -max_recovery_angle,
            max_recovery_angle,
        )
    )


def get_sector_clearance(
    angles: np.ndarray,
    ranges: np.ndarray,
    centre_angle: float,
    half_width_deg: float,
    default_range: float,
) -> float:
    half_width = math.radians(
        half_width_deg
    )

    difference = np.abs(
        np.array(
            [
                normalise_angle(
                    angle - centre_angle
                )
                for angle in angles
            ]
        )
    )

    selected = ranges[
        difference <= half_width
    ]

    if selected.size == 0:
        return default_range

    return float(
        np.min(selected)
    )


def select_steering_angle(
    centres: np.ndarray,
    density: np.ndarray,
    blocked: np.ndarray,
    desired_heading: float,
    previous_heading: float,
) -> float | None:
    free_indices = np.flatnonzero(
        ~blocked
    )

    if free_indices.size == 0:
        return None

    best_angle = None
    best_cost = float("inf")

    for index in free_indices:
        candidate = float(
            centres[index]
        )

        goal_cost = abs(
            normalise_angle(
                candidate
                - desired_heading
            )
        )

        smooth_cost = abs(
            normalise_angle(
                candidate
                - previous_heading
            )
        )

        obstacle_cost = float(
            density[index]
        )

        total_cost = (
            2.0 * goal_cost
            + 1.2 * smooth_cost
            + 0.8 * obstacle_cost
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_angle = candidate

    return best_angle


def compute_velocity_command(
    front_angles: np.ndarray,
    front_ranges: np.ndarray,
    robot_y: float,
    robot_yaw: float,
    previous_heading: float,
    config: VfhConfig,
) -> VfhResult:
    (
        centres,
        density,
        minimum_ranges,
    ) = build_sector_histogram(
        front_angles,
        front_ranges,
        config,
    )

    blocked = inflate_blocked_sectors(
        centres,
        minimum_ranges,
        config,
    )

    front_clearance = get_sector_clearance(
        front_angles,
        front_ranges,
        centre_angle=0.0,
        half_width_deg=15.0,
        default_range=config.safe_distance * 10.0,
    )

    left_clearance = get_sector_clearance(
        front_angles,
        front_ranges,
        centre_angle=math.radians(45.0),
        half_width_deg=25.0,
        default_range=config.safe_distance * 10.0,
    )

    right_clearance = get_sector_clearance(
        front_angles,
        front_ranges,
        centre_angle=math.radians(-45.0),
        half_width_deg=25.0,
        default_range=config.safe_distance * 10.0,
    )

    desired_heading = compute_desired_heading(
        robot_y,
        robot_yaw,
        config,
    )

    steering_angle = select_steering_angle(
        centres,
        density,
        blocked,
        desired_heading,
        previous_heading,
    )

    emergency_stop = (
        front_clearance
        <= config.stop_distance
    )

    if steering_angle is None:
        # No free VFH sector. Rotate towards
        # whichever front side has more clearance.
        steering_angle = (
            math.radians(70.0)
            if left_clearance >= right_clearance
            else math.radians(-70.0)
        )

        emergency_stop = True

    steering_ratio = float(
    np.clip(
        abs(steering_angle)
        / math.radians(90.0),
        0.0,
        1.0,
    )
)

    if front_clearance <= config.stop_distance:
        distance_factor = 0.0

    elif front_clearance >= config.safe_distance:
        distance_factor = 1.0

    else:
        distance_factor = (
            front_clearance
            - config.stop_distance
        ) / (
            config.safe_distance
            - config.stop_distance
        )

    # Strongly reduce speed when the selected VFH direction
    # requires a large steering angle.
    turn_factor = float(
        np.clip(
            1.0 - 1.50 * steering_ratio,
            0.20,
            1.0,
        )
    )

    steering_degrees = abs(
        math.degrees(
            steering_angle
        )
    )

    if steering_degrees < 5.0:
        steering_speed_limit = 2.00

    elif steering_degrees < 15.0:
        steering_speed_limit = 1.20

    elif steering_degrees < 30.0:
        steering_speed_limit = 0.70

    else:
        steering_speed_limit = 0.40

    target_speed = min(
        config.max_forward_speed
        * distance_factor
        * turn_factor,
        steering_speed_limit,
    )


    if emergency_stop:
        target_speed = 0.0

    angular_z = np.clip(
        steering_angle,
        -config.max_yaw_rate,
        config.max_yaw_rate,
    )

    return VfhResult(
        linear_x=float(target_speed),
        angular_z=float(angular_z),
        steering_angle=float(
            steering_angle
        ),
        front_clearance=float(
            front_clearance
        ),
        emergency_stop=bool(
            emergency_stop
        ),
    )


def ramp_speed(
    current_speed: float,
    target_speed: float,
    acceleration_rate: float,
    deceleration_rate: float,
    delta_time: float,
) -> float:
    """Move current speed gradually towards the requested target speed."""
    if target_speed > current_speed:
        maximum_change = acceleration_rate * delta_time
    else:
        maximum_change = deceleration_rate * delta_time

    difference = target_speed - current_speed

    change = float(
        np.clip(
            difference,
            -maximum_change,
            maximum_change,
        )
    )

    return current_speed + change


class VfhObstacleAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__(
            "vfh_obstacle_avoidance"
        )

        self.config = VfhConfig()

        self.scan_message: LaserScan | None = None
        self.robot_y: float | None = None
        self.robot_yaw: float | None = None

        self.last_scan_time: float | None = None
        self.last_odom_time: float | None = None

        self.previous_heading = 0.0

        self.current_linear_speed = 0.0
        self.last_control_time = time.monotonic()

        self.command_publisher = (
            self.create_publisher(
                Twist,
                "/cmd_vel",
                10,
            )
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.control_timer = self.create_timer(
            0.05,
            self.control_callback,
        )

        self.last_log_time = 0.0

        self.get_logger().info(
            "VFH obstacle avoidance started | "
            f"max speed={self.config.max_forward_speed:.2f} m/s, "
            f"max yaw={self.config.max_yaw_rate:.2f} rad/s, "
            f"safe distance={self.config.safe_distance:.2f} m, "
            f"stop distance={self.config.stop_distance:.2f} m, "
            "centre line y=0.0"
        )
        
    def scan_callback(
        self,
        message: LaserScan,
    ) -> None:
        self.scan_message = message
        self.last_scan_time = time.monotonic()

    def odom_callback(
        self,
        message: Odometry,
    ) -> None:
        self.robot_y = float(
            message.pose.pose.position.y
        )

        orientation = (
            message.pose.pose.orientation
        )

        self.robot_yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )

        self.last_odom_time = time.monotonic()

    def publish_stop(self) -> None:
        self.current_linear_speed = 0.0
        self.last_control_time = time.monotonic()

        self.command_publisher.publish(
            Twist()
        )

    def control_callback(self) -> None:
        now = time.monotonic()

        if (
            self.scan_message is None
            or self.robot_y is None
            or self.robot_yaw is None
            or self.last_scan_time is None
            or self.last_odom_time is None
        ):
            self.publish_stop()
            return

        scan_age = (
            now - self.last_scan_time
        )

        odom_age = (
            now - self.last_odom_time
        )

        if (
            scan_age > self.config.scan_timeout
            or odom_age > self.config.odom_timeout
        ):
            self.publish_stop()

            self.get_logger().warning(
                "Stopping: stale scan or odometry",
                throttle_duration_sec=2.0,
            )
            return

        scan = self.scan_message

        clean_ranges = sanitise_scan(
            scan.ranges,
            scan.range_min,
            scan.range_max,
        )

        (
            front_angles,
            front_ranges,
        ) = extract_front_scan(
            clean_ranges,
            scan.angle_min,
            scan.angle_increment,
            self.config.front_fov_deg,
        )

        result = compute_velocity_command(
            front_angles,
            front_ranges,
            self.robot_y,
            self.robot_yaw,
            self.previous_heading,
            self.config,
        )

        # Smooth steering-angle memory to reduce
        # left/right oscillation.
        self.previous_heading = (
            0.70 * self.previous_heading
            + 0.30 * result.steering_angle
        )

        delta_time = max(
            0.001,
            now - self.last_control_time,
        )

        self.last_control_time = now

        self.current_linear_speed = ramp_speed(
            current_speed=self.current_linear_speed,
            target_speed=result.linear_x,
            acceleration_rate=self.config.acceleration_rate,
            deceleration_rate=self.config.deceleration_rate,
            delta_time=delta_time,
        )

        # Emergency stopping must not wait for the normal deceleration ramp.
        if result.emergency_stop:
            self.current_linear_speed = 0.0

        command = Twist()

        command.linear.x = float(
            self.current_linear_speed
        )
        command.angular.z = result.angular_z

        self.command_publisher.publish(
            command
        )

        if now - self.last_log_time >= 0.50:
            self.get_logger().info(
                "VFH | "
                f"front={result.front_clearance:.2f} m, "
                f"y={self.robot_y:.2f} m, "
                f"steer={math.degrees(result.steering_angle):.1f} deg, "
                f"target_x={result.linear_x:.2f}, "
                f"cmd_x={self.current_linear_speed:.2f}, "
                f"cmd_yaw={result.angular_z:.2f}, "
                f"emergency={result.emergency_stop}"
            )

            self.last_log_time = now

    def destroy_node(self) -> bool:
        self.publish_stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = VfhObstacleAvoidanceNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()