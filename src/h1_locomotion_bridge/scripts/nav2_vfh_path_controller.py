#!/usr/bin/env python3

from __future__ import annotations

import math
import time

import numpy as np
import rclpy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener

# Reuse the proven VFH helper functions from the fixed-waypoint controller.
from waypoint_vfh_controller import (
    ControllerConfig,
    PlannerResult,
    build_sector_histogram,
    extract_front_scan,
    get_clearance,
    inflate_blocked_sectors,
    normalise_angle,
    ramp_speed,
    sanitise_scan,
    select_steering_angle,
)


class Nav2VfhPathController(Node):
    """
    Follow a Nav2 global path using the existing H1 VFH steering logic.

    Inputs:
      /plan                 nav_msgs/Path
      /scan                 sensor_msgs/LaserScan
      /odometry/filtered    nav_msgs/Odometry
      TF map -> pelvis

    Output:
      /cmd_vel              geometry_msgs/Twist

    Important:
      Run this only with the planner-only Nav2 launch.
      Do not run MPPI or waypoint_vfh_controller at the same time.
    """

    def __init__(self) -> None:
        super().__init__("nav2_vfh_path_controller")

        self.config = ControllerConfig()

        # Path-following parameters.
        self.lookahead_distance = 1.20
        self.goal_tolerance = 0.25
        self.path_timeout = 5.0
        self.minimum_forward_speed = 0.30
        self.maximum_forward_speed = 0.70
        self.maximum_yaw_rate = 0.80

        self.global_frame = "map"
        self.robot_frame = "pelvis"

        self.path: Path | None = None
        self.scan_message: LaserScan | None = None

        self.last_path_time: float | None = None
        self.last_scan_time: float | None = None
        self.last_odom_time: float | None = None

        self.current_linear_speed = 0.0
        self.previous_heading = 0.0
        self.last_control_time = time.monotonic()
        self.last_log_time = 0.0

        self.goal_reached = False
        self.last_target_index = 0
        self.path_progress_index = 0
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.command_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.create_subscription(
            Path,
            "/plan",
            self.path_callback,
            10,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self.odom_callback,
            10,
        )

        self.control_timer = self.create_timer(
            0.05,
            self.control_callback,
        )

        self.get_logger().info(
            "Nav2 VFH path controller started | "
            f"path=/plan | "
            f"lookahead={self.lookahead_distance:.2f} m | "
            f"goal tolerance={self.goal_tolerance:.2f} m"
        )

    def path_callback(self, message: Path) -> None:
        if not message.poses:
            self.get_logger().warning(
                "Received an empty Nav2 path."
            )
            self.path = None
            self.publish_stop()
            return

        if message.header.frame_id:
            self.global_frame = message.header.frame_id

        self.path = message
        self.last_path_time = time.monotonic()
        self.goal_reached = False
        self.last_target_index = 0
        self.path_progress_index = 0
        self.previous_heading = 0.0
        
        final_pose = message.poses[-1].pose.position

        self.get_logger().info(
            f"Received Nav2 path with {len(message.poses)} poses | "
            f"frame={self.global_frame} | "
            f"goal=({final_pose.x:.2f}, {final_pose.y:.2f})"
        )

    def scan_callback(self, message: LaserScan) -> None:
        self.scan_message = message
        self.last_scan_time = time.monotonic()

    def odom_callback(self, message: Odometry) -> None:
        # EKF odometry is used as a freshness/health check.
        # Robot position in the path frame is read from TF map -> pelvis.
        del message
        self.last_odom_time = time.monotonic()

    def publish_stop(self) -> None:
        self.current_linear_speed = 0.0
        self.last_control_time = time.monotonic()
        self.command_publisher.publish(Twist())

    def get_robot_pose(self) -> tuple[float, float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.10),
            )
        except TransformException as error:
            self.get_logger().warning(
                f"Waiting for TF "
                f"{self.global_frame} -> {self.robot_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        sin_yaw = 2.0 * (
            (rotation.w * rotation.z)
            + (rotation.x * rotation.y)
        )

        cos_yaw = 1.0 - 2.0 * (
            (rotation.y * rotation.y)
            + (rotation.z * rotation.z)
        )

        yaw = math.atan2(sin_yaw, cos_yaw)

        return (
            float(translation.x),
            float(translation.y),
            float(yaw),
        )

    def find_nearest_path_index(
        self,
        robot_x: float,
        robot_y: float,
    ) -> int:
        assert self.path is not None

        poses = self.path.poses

        start_index = max(
            0,
            self.path_progress_index - 20,
        )

        end_index = min(
            len(poses),
            self.path_progress_index + 100,
        )

        nearest_index = start_index
        nearest_distance = float("inf")

        for index in range(start_index, end_index):
            position = poses[index].pose.position

            distance = math.hypot(
                float(position.x) - robot_x,
                float(position.y) - robot_y,
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index

        self.path_progress_index = max(
            self.path_progress_index,
            nearest_index,
        )

        return self.path_progress_index    
        
    def select_lookahead_index(
        self,
        nearest_index: int,
    ) -> int:
        assert self.path is not None

        poses = self.path.poses

        if nearest_index >= len(poses) - 1:
            return len(poses) - 1

        accumulated_distance = 0.0

        for index in range(nearest_index, len(poses) - 1):
            current = poses[index].pose.position
            next_pose = poses[index + 1].pose.position

            accumulated_distance += math.hypot(
                float(next_pose.x) - float(current.x),
                float(next_pose.y) - float(current.y),
            )

            if accumulated_distance >= self.lookahead_distance:
                return index + 1

        return len(poses) - 1

    def calculate_path_heading(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_index: int,
    ) -> tuple[float, float]:
        assert self.path is not None

        target = self.path.poses[target_index].pose.position

        delta_x = float(target.x) - robot_x
        delta_y = float(target.y) - robot_y

        target_distance = math.hypot(
            delta_x,
            delta_y,
        )

        target_world_heading = math.atan2(
            delta_y,
            delta_x,
        )

        relative_heading = normalise_angle(
            target_world_heading - robot_yaw
        )

        relative_heading = float(
            np.clip(
                relative_heading,
                -math.radians(87.0),
                math.radians(87.0),
            )
        )

        return relative_heading, target_distance

    def distance_to_goal(
        self,
        robot_x: float,
        robot_y: float,
    ) -> float:
        assert self.path is not None

        goal = self.path.poses[-1].pose.position

        return math.hypot(
            float(goal.x) - robot_x,
            float(goal.y) - robot_y,
        )

    def compute_vfh_plan(
        self,
        front_angles: np.ndarray,
        front_ranges: np.ndarray,
        desired_heading: float,
        goal_distance: float,
    ) -> PlannerResult:
        centres, density, minimum_ranges = build_sector_histogram(
            front_angles,
            front_ranges,
            self.config,
        )

        blocked = inflate_blocked_sectors(
            centres,
            minimum_ranges,
            self.config,
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

        emergency_stop = (
            front_clearance <= self.config.stop_distance
        )

        if steering_angle is None:
            steering_angle = (
                math.radians(70.0)
                if left_clearance >= right_clearance
                else math.radians(-70.0)
            )
            emergency_stop = True

        avoiding = (
            front_clearance < self.config.safe_distance
            or abs(
                normalise_angle(
                    steering_angle - desired_heading
                )
            ) > math.radians(10.0)
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
            clearance_factor = 0.0

        elif front_clearance >= self.config.safe_distance:
            clearance_factor = 1.0

        else:
            clearance_factor = (
                front_clearance - self.config.stop_distance
            ) / (
                self.config.safe_distance
                - self.config.stop_distance
            )

        # Use stable H1 walking speeds.
        if heading_error > math.radians(55.0):
            target_speed = 0.0

        elif heading_error > math.radians(35.0):
            target_speed = 0.30

        elif heading_error > math.radians(18.0):
            target_speed = 0.40

        else:
            target_speed = self.maximum_forward_speed

        target_speed *= clearance_factor
        target_speed *= 1.0 - (0.55 * steering_ratio)

        if avoiding and target_speed > 0.0:
            target_speed = min(
                target_speed,
                self.config.avoidance_speed_limit,
            )

        # Slow down near the final goal.
        if goal_distance < 1.50:
            target_speed = min(target_speed, 0.45)

        if goal_distance < 0.70:
            target_speed = min(target_speed, 0.30)

        # Do not send unstable tiny forward commands to the RL policy.
        if 0.0 < target_speed < self.minimum_forward_speed:
            target_speed = self.minimum_forward_speed

        if emergency_stop:
            target_speed = 0.0

        # Use continuous path-heading correction when the route is clear.
        # VFH sector centres are spaced by 5 degrees, which otherwise causes
        # angular.z to oscillate between approximately +/-0.0436 rad/s.
        if not avoiding and front_clearance >= self.config.safe_distance:
            heading_gain = 1.5

            angular_z = float(
                np.clip(
                    heading_gain * desired_heading,
                    -self.maximum_yaw_rate,
                    self.maximum_yaw_rate,
                )
            )

            # Ignore extremely small heading noise.
            if abs(angular_z) < 0.01:
                angular_z = 0.0

        else:
            angular_z = float(
                np.clip(
                    steering_angle,
                    -self.maximum_yaw_rate,
                    self.maximum_yaw_rate,
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

    def inputs_are_fresh(self, now: float) -> bool:
        if (
            self.path is None
            or self.scan_message is None
            or self.last_path_time is None
            or self.last_scan_time is None
            or self.last_odom_time is None
        ):
            return False

        if now - self.last_path_time > self.path_timeout:
            # The global plan may not be continuously republished.
            # Keep the latest non-empty path, so only warn here.
            self.get_logger().warning(
                "Using the latest stored Nav2 path.",
                throttle_duration_sec=5.0,
            )

        if (
            now - self.last_scan_time > self.config.scan_timeout
            or now - self.last_odom_time > self.config.odom_timeout
        ):
            return False

        return True

    def control_callback(self) -> None:
        now = time.monotonic()

        if self.goal_reached:
            self.publish_stop()
            return

        if not self.inputs_are_fresh(now):
            self.publish_stop()
            return

        robot_pose = self.get_robot_pose()

        if robot_pose is None:
            self.publish_stop()
            return

        robot_x, robot_y, robot_yaw = robot_pose

        assert self.path is not None
        assert self.scan_message is not None

        goal_distance = self.distance_to_goal(
            robot_x,
            robot_y,
        )

        if goal_distance <= self.goal_tolerance:
            self.goal_reached = True
            self.publish_stop()

            self.get_logger().info(
                f"Nav2 path goal reached | "
                f"robot=({robot_x:.2f}, {robot_y:.2f}) | "
                f"remaining={goal_distance:.2f} m"
            )
            return

        nearest_index = self.find_nearest_path_index(
            robot_x,
            robot_y,
        )

        target_index = self.select_lookahead_index(
            nearest_index,
        )

        self.last_target_index = target_index

        desired_heading, target_distance = (
            self.calculate_path_heading(
                robot_x,
                robot_y,
                robot_yaw,
                target_index,
            )
        )

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

        result = self.compute_vfh_plan(
            front_angles,
            front_ranges,
            desired_heading,
            goal_distance,
        )

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
            self.current_linear_speed,
            result.target_speed,
            self.config.acceleration_rate,
            self.config.deceleration_rate,
            delta_time,
        )

        if (
            result.emergency_stop
            or result.target_speed <= 0.0
        ):
            self.current_linear_speed = 0.0

        command = Twist()
        command.linear.x = float(
            self.current_linear_speed
        )
        command.angular.z = float(
            result.angular_z
        )

        self.command_publisher.publish(command)

        if now - self.last_log_time >= 1.0:
            self.get_logger().info(
                "PATH-FOLLOW | "
                f"robot=({robot_x:.2f}, {robot_y:.2f}) | "
                f"nearest={nearest_index} | "
                f"target={target_index}/{len(self.path.poses) - 1} | "
                f"target_distance={target_distance:.2f} m | "
                f"goal_distance={goal_distance:.2f} m | "
                f"heading={math.degrees(desired_heading):.1f} deg | "
                f"cmd_x={self.current_linear_speed:.2f} | "
                f"cmd_yaw={result.angular_z:.2f} | "
                f"front={result.front_clearance:.2f} m"
            )

            self.last_log_time = now

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.publish_stop()

        return super().destroy_node()

def main(args=None) -> None:
    rclpy.init(args=args)

    node = Nav2VfhPathController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.publish_stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()