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
class ControllerConfig:
    waypoint_tolerance: float = 0.70
    final_waypoint_tolerance: float = 0.70

    clear_path_speed: float = 0.75
    avoidance_speed_limit: float = 0.45
    corner_speed_limit: float = 0.35
    minimum_avoidance_speed: float = 0.15

    acceleration_rate: float = 0.30
    deceleration_rate: float = 0.90
    max_yaw_rate: float = 0.80

    stop_distance: float = 1.20
    safe_distance: float = 4.50
    self_hit_distance: float = 0.25

    sector_size_deg: float = 5.0
    front_fov_deg: float = 180.0
    robot_radius: float = 0.40
    safety_margin: float = 0.20

    goal_weight: float = 3.0
    smooth_weight: float = 1.5
    density_weight: float = 0.8

    scan_timeout: float = 0.50
    odom_timeout: float = 0.50


@dataclass(frozen=True)
class PlannerResult:
    target_speed: float
    angular_z: float
    steering_angle: float
    front_clearance: float
    emergency_stop: bool
    avoiding: bool


def normalise_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * ((w * z) + (x * y))
    cos_yaw = 1.0 - 2.0 * ((y * y) + (z * z))
    return math.atan2(sin_yaw, cos_yaw)


def sanitise_scan(
    ranges: Sequence[float],
    range_min: float,
    range_max: float,
    self_hit_distance: float,
) -> np.ndarray:
    values = np.asarray(ranges, dtype=np.float64).copy()

    invalid = (
        ~np.isfinite(values)
        | (values < range_min)
        | (values > range_max)
        | (values < self_hit_distance)
    )

    values[invalid] = range_max
    return values


def extract_front_scan(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    front_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(ranges.size, dtype=np.float64)
    angles = angle_min + indices * angle_increment
    half_fov = math.radians(front_fov_deg / 2.0)
    mask = (angles >= -half_fov) & (angles <= half_fov)
    return angles[mask], ranges[mask]


def build_sector_histogram(
    angles: np.ndarray,
    ranges: np.ndarray,
    config: ControllerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_fov_deg = config.front_fov_deg / 2.0

    edges_deg = np.arange(
        -half_fov_deg,
        half_fov_deg + config.sector_size_deg,
        config.sector_size_deg,
        dtype=np.float64,
    )

    centres = np.radians((edges_deg[:-1] + edges_deg[1:]) / 2.0)
    density = np.zeros(centres.size, dtype=np.float64)
    minimum_ranges = np.full(centres.size, np.inf, dtype=np.float64)

    for angle_deg, distance in zip(np.degrees(angles), ranges):
        sector_index = int(
            math.floor((angle_deg + half_fov_deg) / config.sector_size_deg)
        )
        sector_index = int(np.clip(sector_index, 0, centres.size - 1))

        minimum_ranges[sector_index] = min(
            minimum_ranges[sector_index], float(distance)
        )

        if distance < config.safe_distance:
            obstacle_density = (
                config.safe_distance - float(distance)
            ) / config.safe_distance
            density[sector_index] = max(
                density[sector_index], obstacle_density
            )

    minimum_ranges[~np.isfinite(minimum_ranges)] = config.safe_distance * 10.0
    return centres, density, minimum_ranges


def inflate_blocked_sectors(
    centres: np.ndarray,
    minimum_ranges: np.ndarray,
    config: ControllerConfig,
) -> np.ndarray:
    blocked = np.zeros(centres.size, dtype=bool)
    effective_radius = config.robot_radius + config.safety_margin

    for index, distance in enumerate(minimum_ranges):
        if distance >= config.safe_distance:
            continue

        safe_distance = max(float(distance), effective_radius + 0.01)
        inflation_angle = math.asin(
            float(np.clip(effective_radius / safe_distance, 0.0, 1.0))
        )

        angular_difference = np.abs(
            np.array(
                [
                    normalise_angle(candidate - centres[index])
                    for candidate in centres
                ],
                dtype=np.float64,
            )
        )
        blocked |= angular_difference <= inflation_angle

    return blocked


def get_clearance(
    angles: np.ndarray,
    ranges: np.ndarray,
    centre_angle: float,
    half_width_deg: float,
    default_range: float,
) -> float:
    half_width = math.radians(half_width_deg)
    difference = np.abs(
        np.array(
            [normalise_angle(angle - centre_angle) for angle in angles],
            dtype=np.float64,
        )
    )

    selected = ranges[difference <= half_width]
    if selected.size == 0:
        return default_range
    return float(np.min(selected))


def select_steering_angle(
    centres: np.ndarray,
    density: np.ndarray,
    blocked: np.ndarray,
    desired_relative_heading: float,
    previous_heading: float,
    config: ControllerConfig,
) -> float | None:
    free_indices = np.flatnonzero(~blocked)
    if free_indices.size == 0:
        return None

    best_angle: float | None = None
    best_cost = float("inf")

    for index in free_indices:
        candidate = float(centres[index])
        goal_cost = abs(normalise_angle(candidate - desired_relative_heading))
        smooth_cost = abs(normalise_angle(candidate - previous_heading))
        obstacle_cost = float(density[index])

        total_cost = (
            config.goal_weight * goal_cost
            + config.smooth_weight * smooth_cost
            + config.density_weight * obstacle_cost
        )

        if total_cost < best_cost:
            best_cost = total_cost
            best_angle = candidate

    return best_angle


def ramp_speed(
    current_speed: float,
    target_speed: float,
    acceleration_rate: float,
    deceleration_rate: float,
    delta_time: float,
) -> float:
    rate = acceleration_rate if target_speed > current_speed else deceleration_rate
    maximum_change = rate * delta_time
    difference = target_speed - current_speed

    return current_speed + float(
        np.clip(difference, -maximum_change, maximum_change)
    )


class WaypointVfhController(Node):
    def __init__(self) -> None:
        super().__init__("waypoint_vfh_controller")

        self.config = ControllerConfig()

        self.waypoints = [
            (3.0, 0.0), (9.0, 0.0), (15.0, 0.0), (21.0, 0.0), (26.0, 0.0),
            (27.5, 0.4), (28.7, 1.2), (29.5, 2.2), (30.0, 3.5),
            (30.0, 6.0), (30.0, 9.0),
            (29.7, 10.2), (29.0, 11.0), (28.0, 11.7), (26.5, 12.0),
            (22.0, 12.0), (18.0, 12.0), (12.0, 12.0), (6.0, 12.0), (3.0, 12.0),
            (1.8, 11.7), (0.9, 11.0), (0.3, 10.0), (0.0, 8.5),
            (0.0, 6.0), (0.0, 3.0),
            (0.3, 1.8), (1.0, 0.9), (2.0, 0.3), (3.5, 0.0),
            (9.0, 0.0),
        ]

        self.current_waypoint_index = 0
        self.finished = False

        self.scan_message: LaserScan | None = None
        self.robot_x: float | None = None
        self.robot_y: float | None = None
        self.robot_yaw: float | None = None

        self.last_scan_time: float | None = None
        self.last_odom_time: float | None = None

        self.previous_heading = 0.0
        self.current_linear_speed = 0.0
        self.last_control_time = time.monotonic()
        self.last_log_time = 0.0

        self.command_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
        )
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.get_logger().info(
            "Waypoint + VFH controller started | "
            f"{len(self.waypoints)} waypoints | "
            f"clear speed={self.config.clear_path_speed:.2f} m/s | "
            f"avoidance limit={self.config.avoidance_speed_limit:.2f} m/s"
        )

    def scan_callback(self, message: LaserScan) -> None:
        self.scan_message = message
        self.last_scan_time = time.monotonic()

    def odom_callback(self, message: Odometry) -> None:
        self.robot_x = float(message.pose.pose.position.x)
        self.robot_y = float(message.pose.pose.position.y)

        orientation = message.pose.pose.orientation
        self.robot_yaw = quaternion_to_yaw(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        self.last_odom_time = time.monotonic()

    def publish_stop(self) -> None:
        self.current_linear_speed = 0.0
        self.last_control_time = time.monotonic()
        self.command_publisher.publish(Twist())

    def update_waypoint(self) -> bool:
        assert self.robot_x is not None
        assert self.robot_y is not None

        while self.current_waypoint_index < len(self.waypoints):
            waypoint_x, waypoint_y = self.waypoints[self.current_waypoint_index]
            distance = math.hypot(
                waypoint_x - self.robot_x,
                waypoint_y - self.robot_y,
            )

            tolerance = (
                self.config.final_waypoint_tolerance
                if self.current_waypoint_index == len(self.waypoints) - 1
                else self.config.waypoint_tolerance
            )

            if distance >= tolerance:
                return True

            self.get_logger().info(
                f"Waypoint {self.current_waypoint_index} reached | "
                f"robot=({self.robot_x:.2f}, {self.robot_y:.2f})"
            )
            self.current_waypoint_index += 1
            self.previous_heading = 0.0

        self.finished = True
        self.publish_stop()
        self.get_logger().info("All waypoints reached. Robot stopped.")
        return False

    def waypoint_relative_heading(self) -> tuple[float, float]:
        assert self.robot_x is not None
        assert self.robot_y is not None
        assert self.robot_yaw is not None

        waypoint_x, waypoint_y = self.waypoints[self.current_waypoint_index]
        dx = waypoint_x - self.robot_x
        dy = waypoint_y - self.robot_y

        distance = math.hypot(dx, dy)
        target_world_heading = math.atan2(dy, dx)
        desired_relative_heading = normalise_angle(
            target_world_heading - self.robot_yaw
        )

        desired_relative_heading = float(
            np.clip(
                desired_relative_heading,
                -math.radians(87.0),
                math.radians(87.0),
            )
        )
        return desired_relative_heading, distance

    def compute_plan(
        self,
        front_angles: np.ndarray,
        front_ranges: np.ndarray,
        desired_heading: float,
        waypoint_distance: float,
    ) -> PlannerResult:
        centres, density, minimum_ranges = build_sector_histogram(
            front_angles, front_ranges, self.config
        )

        blocked = inflate_blocked_sectors(
            centres, minimum_ranges, self.config
        )

        front_clearance = get_clearance(
            front_angles,
            front_ranges,
            centre_angle=0.0,
            half_width_deg=15.0,
            default_range=self.config.safe_distance * 10.0,
        )

        left_clearance = get_clearance(
            front_angles,
            front_ranges,
            centre_angle=math.radians(45.0),
            half_width_deg=25.0,
            default_range=self.config.safe_distance * 10.0,
        )

        right_clearance = get_clearance(
            front_angles,
            front_ranges,
            centre_angle=math.radians(-45.0),
            half_width_deg=25.0,
            default_range=self.config.safe_distance * 10.0,
        )

        steering_angle = select_steering_angle(
            centres,
            density,
            blocked,
            desired_heading,
            self.previous_heading,
            self.config,
        )

        emergency_stop = front_clearance <= self.config.stop_distance

        if steering_angle is None:
            steering_angle = (
                math.radians(70.0)
                if left_clearance >= right_clearance
                else math.radians(-70.0)
            )
            emergency_stop = True

        avoiding = (
            front_clearance < self.config.safe_distance
            or abs(normalise_angle(steering_angle - desired_heading))
            > math.radians(10.0)
        )

        heading_error = abs(desired_heading)
        steering_ratio = float(
            np.clip(
                abs(steering_angle) / math.radians(90.0),
                0.0,
                1.0,
            )
        )

        if front_clearance <= self.config.stop_distance:
            distance_factor = 0.0
        elif front_clearance >= self.config.safe_distance:
            distance_factor = 1.0
        else:
            distance_factor = (
                front_clearance - self.config.stop_distance
            ) / (
                self.config.safe_distance - self.config.stop_distance
            )

        if heading_error > math.radians(40.0):
            target_speed = 0.0
        elif heading_error > math.radians(25.0):
            target_speed = 0.15
        elif heading_error > math.radians(12.0):
            target_speed = 0.30
        else:
            target_speed = self.config.clear_path_speed

        target_speed *= distance_factor
        target_speed *= 1.0 - 0.65 * steering_ratio

        if avoiding and target_speed > 0.0:
            target_speed = min(
                target_speed,
                self.config.avoidance_speed_limit,
            )
            target_speed = max(
                target_speed,
                self.config.minimum_avoidance_speed,
            )

        if waypoint_distance < 1.2:
            target_speed = min(target_speed, 0.35)

        if self.current_waypoint_index in {
            5, 6, 7, 8,
            11, 12, 13, 14,
            20, 21, 22, 23,
            26, 27, 28, 29,
        }:
            target_speed = min(
                target_speed,
                self.config.corner_speed_limit,
            )

        if emergency_stop:
            target_speed = 0.0

        angular_z = float(
            np.clip(
                steering_angle,
                -self.config.max_yaw_rate,
                self.config.max_yaw_rate,
            )
        )

        return PlannerResult(
            target_speed=float(max(0.0, target_speed)),
            angular_z=angular_z,
            steering_angle=float(steering_angle),
            front_clearance=float(front_clearance),
            emergency_stop=bool(emergency_stop),
            avoiding=bool(avoiding),
        )

    def control_callback(self) -> None:
        now = time.monotonic()

        if self.finished:
            self.publish_stop()
            return

        if (
            self.scan_message is None
            or self.robot_x is None
            or self.robot_y is None
            or self.robot_yaw is None
            or self.last_scan_time is None
            or self.last_odom_time is None
        ):
            self.publish_stop()
            return

        if (
            now - self.last_scan_time > self.config.scan_timeout
            or now - self.last_odom_time > self.config.odom_timeout
        ):
            self.publish_stop()
            self.get_logger().warning(
                "Stopping: stale scan or odometry",
                throttle_duration_sec=2.0,
            )
            return

        if not self.update_waypoint():
            return

        desired_heading, waypoint_distance = self.waypoint_relative_heading()
        scan = self.scan_message

        clean_ranges = sanitise_scan(
            scan.ranges,
            scan.range_min,
            scan.range_max,
            self.config.self_hit_distance,
        )

        front_angles, front_ranges = extract_front_scan(
            clean_ranges,
            scan.angle_min,
            scan.angle_increment,
            self.config.front_fov_deg,
        )

        result = self.compute_plan(
            front_angles,
            front_ranges,
            desired_heading,
            waypoint_distance,
        )

        self.previous_heading = (
            0.70 * self.previous_heading
            + 0.30 * result.steering_angle
        )

        delta_time = max(0.001, now - self.last_control_time)
        self.last_control_time = now

        self.current_linear_speed = ramp_speed(
            self.current_linear_speed,
            result.target_speed,
            self.config.acceleration_rate,
            self.config.deceleration_rate,
            delta_time,
        )

        if result.emergency_stop:
            self.current_linear_speed = 0.0

        command = Twist()
        command.linear.x = float(self.current_linear_speed)
        command.angular.z = result.angular_z
        self.command_publisher.publish(command)

        if now - self.last_log_time >= 0.50:
            waypoint_x, waypoint_y = self.waypoints[
                self.current_waypoint_index
            ]

            self.get_logger().info(
                "WP-VFH | "
                f"wp={self.current_waypoint_index}, "
                f"robot=({self.robot_x:.2f}, {self.robot_y:.2f}), "
                f"target=({waypoint_x:.2f}, {waypoint_y:.2f}), "
                f"distance={waypoint_distance:.2f}, "
                f"goal={math.degrees(desired_heading):.1f} deg, "
                f"steer={math.degrees(result.steering_angle):.1f} deg, "
                f"front={result.front_clearance:.2f} m, "
                f"cmd_x={self.current_linear_speed:.2f}, "
                f"cmd_yaw={result.angular_z:.2f}, "
                f"avoiding={result.avoiding}, "
                f"emergency={result.emergency_stop}"
            )
            self.last_log_time = now

    def destroy_node(self) -> bool:
        self.publish_stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaypointVfhController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
