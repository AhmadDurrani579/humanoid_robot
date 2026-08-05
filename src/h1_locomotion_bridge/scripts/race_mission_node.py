#!/usr/bin/env python3

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class RouteGoal:
    x: float
    y: float
    yaw_degrees: float
    name: str
    tolerance: float


class RaceMissionNode(Node):
    """
    Automatically completes one full race lap.

    Intermediate goals are pass-through waypoints:
      - Complete when inside their tolerance.
      - Or complete after the robot passes near them and starts moving away.

    The final goal remains strict:
      - It must be reached within its configured tolerance.
    """

    def __init__(self) -> None:
        super().__init__("race_mission_node")

        # Frames and Nav2 planner.
        self.global_frame = "map"
        self.robot_frame = "pelvis"
        self.planner_id = "GridBased"

        # Mission settings.
        self.goal_timeout = 120.0
        self.between_goal_delay = 0.10
        self.maximum_planning_retries = 3

        # Pass-through waypoint detection settings.
        self.minimum_waypoint_progress = 0.60
        self.distance_increase_threshold = 0.20
        self.distance_increase_required_count = 5
        self.minimum_goal_follow_time = 1.0
        
        
        # Tracking for the current waypoint.
        self.closest_goal_distance = float("inf")
        self.distance_increase_count = 0
        self.initial_goal_distance = float("inf")
        
        # Same route that worked during manual tests.
        self.route: list[RouteGoal] = [
            RouteGoal(
                x=27.0,
                y=0.0,
                yaw_degrees=0.0,
                name="First straight",
                tolerance=0.30,
            ),
            RouteGoal(
                x=28.7,
                y=1.2,
                yaw_degrees=45.0,
                name="First corner entry",
                tolerance=0.65,
            ),
            RouteGoal(
                x=29.5,
                y=2.2,
                yaw_degrees=70.0,
                name="First corner middle",
                tolerance=0.80,
            ),
            RouteGoal(
                x=30.0,
                y=4.0,
                yaw_degrees=90.0,
                name="First corner exit",
                tolerance=0.50,
            ),
            RouteGoal(
                x=30.0,
                y=10.0,
                yaw_degrees=90.0,
                name="Right straight",
                tolerance=0.35,
            ),
            RouteGoal(
                x=27.0,
                y=13.0,
                yaw_degrees=180.0,
                name="Top-right corner",
                tolerance=0.55,
            ),
            RouteGoal(
                x=3.0,
                y=13.0,
                yaw_degrees=180.0,
                name="Top straight",
                tolerance=0.35,
            ),
            RouteGoal(
                x=0.0,
                y=10.0,
                yaw_degrees=-90.0,
                name="Top-left corner",
                tolerance=0.55,
            ),
            RouteGoal(
                x=0.0,
                y=3.0,
                yaw_degrees=-90.0,
                name="Left straight",
                tolerance=0.35,
            ),
            RouteGoal(
                x=3.0,
                y=0.0,
                yaw_degrees=0.0,
                name="Return to start",
                tolerance=0.25,
            ),
        ]

        self.action_client = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose",
        )

        self.path_clear_publisher = self.create_publisher(
            Path,
            "/plan",
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.current_goal_index = 0
        self.current_goal_start_time: float | None = None
        self.mission_start_time: float | None = None

        self.planning_in_progress = False
        self.following_current_goal = False
        self.mission_finished = False
        self.mission_failed = False

        self.planning_retry_count = 0
        self.next_goal_send_time = time.monotonic() + 1.0
        self.last_status_log_time = 0.0
        self.last_server_warning_time = 0.0

        self.control_timer = self.create_timer(
            0.10,
            self.control_callback,
        )

        self.get_logger().info(
            "Race mission node started | "
            f"goals={len(self.route)} | "
            f"timeout={self.goal_timeout:.0f} s | "
            f"pass-through distance="
        )

    def clear_path(self) -> None:
        """Publish an empty path to stop the path controller."""

        empty_path = Path()
        empty_path.header.frame_id = self.global_frame
        empty_path.header.stamp = self.get_clock().now().to_msg()

        self.path_clear_publisher.publish(empty_path)

    def get_robot_position(self) -> tuple[float, float] | None:
        """Read the robot position from map -> pelvis TF."""

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
                f"{self.global_frame} -> {self.robot_frame}: "
                f"{error}",
                throttle_duration_sec=2.0,
            )
            return None

        translation = transform.transform.translation

        return (
            float(translation.x),
            float(translation.y),
        )

    @staticmethod
    def quaternion_from_yaw(
        yaw_degrees: float,
    ) -> tuple[float, float]:
        """Convert yaw in degrees to quaternion z and w."""

        yaw_radians = math.radians(yaw_degrees)

        quaternion_z = math.sin(yaw_radians / 2.0)
        quaternion_w = math.cos(yaw_radians / 2.0)

        return quaternion_z, quaternion_w

    def create_planner_goal(
        self,
        route_goal: RouteGoal,
    ) -> ComputePathToPose.Goal:
        """Create a Nav2 ComputePathToPose action goal."""

        action_goal = ComputePathToPose.Goal()

        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()

        pose.pose.position.x = route_goal.x
        pose.pose.position.y = route_goal.y
        pose.pose.position.z = 0.0

        quaternion_z, quaternion_w = self.quaternion_from_yaw(
            route_goal.yaw_degrees
        )

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = quaternion_z
        pose.pose.orientation.w = quaternion_w

        action_goal.goal = pose
        action_goal.planner_id = self.planner_id
        action_goal.use_start = False

        return action_goal

    def send_current_goal(self) -> None:
        """Send the current route goal to Nav2."""

        if self.current_goal_index >= len(self.route):
            self.finish_mission()
            return

        if self.planning_in_progress:
            return

        route_goal = self.route[self.current_goal_index]

        # Reset pass-through tracking for the new goal.
        self.closest_goal_distance = float("inf")
        self.distance_increase_count = 0
        self.initial_goal_distance = float("inf")

        self.planning_in_progress = True
        self.following_current_goal = False

        self.get_logger().info(
            f"Sending goal "
            f"{self.current_goal_index + 1}/{len(self.route)} | "
            f"{route_goal.name} | "
            f"position=({route_goal.x:.2f}, "
            f"{route_goal.y:.2f}) | "
            f"yaw={route_goal.yaw_degrees:.1f} deg | "
            f"tolerance={route_goal.tolerance:.2f} m"
        )

        action_goal = self.create_planner_goal(route_goal)

        future = self.action_client.send_goal_async(action_goal)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future) -> None:
        """Handle Nav2 goal acceptance."""

        try:
            goal_handle = future.result()

        except Exception as error:
            self.planning_in_progress = False
            self.handle_planning_failure(
                f"Failed to send planner goal: {error}"
            )
            return

        if not goal_handle.accepted:
            self.planning_in_progress = False
            self.handle_planning_failure(
                "Planner rejected the goal."
            )
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.planning_result_callback
        )

    def planning_result_callback(self, future) -> None:
        """Handle the planned path result."""

        self.planning_in_progress = False

        try:
            wrapped_result = future.result()
            result = wrapped_result.result

        except Exception as error:
            self.handle_planning_failure(
                f"Planner result failed: {error}"
            )
            return

        if result.error_code != ComputePathToPose.Result.NONE:
            error_message = result.error_msg

            if not error_message:
                error_message = "No additional error message."

            self.handle_planning_failure(
                f"Planner error code={result.error_code} | "
                f"{error_message}"
            )
            return

        if not result.path.poses:
            self.handle_planning_failure(
                "Planner returned an empty path."
            )
            return

        self.planning_retry_count = 0
        self.following_current_goal = True
        self.current_goal_start_time = time.monotonic()

        if self.mission_start_time is None:
            self.mission_start_time = self.current_goal_start_time

        self.get_logger().info(
            f"Path ready for goal "
            f"{self.current_goal_index + 1} | "
            f"poses={len(result.path.poses)}"
        )

    def handle_planning_failure(
        self,
        reason: str,
    ) -> None:
        """Retry planning or fail the mission."""

        self.following_current_goal = False
        self.clear_path()

        self.planning_retry_count += 1

        if (
            self.planning_retry_count
            > self.maximum_planning_retries
        ):
            self.fail_mission(
                f"Planning failed after "
                f"{self.maximum_planning_retries} retries | "
                f"{reason}"
            )
            return

        self.get_logger().warning(
            f"{reason} | "
            f"retry "
            f"{self.planning_retry_count}/"
            f"{self.maximum_planning_retries}"
        )

        self.next_goal_send_time = time.monotonic() + 2.0

    def calculate_goal_distance(
        self,
        robot_x: float,
        robot_y: float,
    ) -> float:
        """Calculate Euclidean distance to the current goal."""

        route_goal = self.route[self.current_goal_index]

        return math.hypot(
            route_goal.x - robot_x,
            route_goal.y - robot_y,
        )

    def is_final_goal(self) -> bool:
        """Return True when following the last route goal."""

        return self.current_goal_index == len(self.route) - 1

    def update_pass_through_detection(
        self,
        distance: float,
    ) -> bool:
        """
        Detect that an intermediate waypoint has been passed.

        The robot must first make meaningful progress toward the waypoint.
        After reaching its closest point, the distance must then increase
        continuously before the next waypoint is selected.
        """

        if not math.isfinite(self.initial_goal_distance):
            self.initial_goal_distance = distance

        if distance < self.closest_goal_distance:
            self.closest_goal_distance = distance
            self.distance_increase_count = 0
            return False

        progress_made = (
            self.initial_goal_distance
            - self.closest_goal_distance
        )

        moved_away_from_closest = (
            distance
            > self.closest_goal_distance
            + self.distance_increase_threshold
        )

        if moved_away_from_closest:
            self.distance_increase_count += 1
        else:
            self.distance_increase_count = 0

        sufficient_progress = (
            progress_made
            >= self.minimum_waypoint_progress
        )

        return (
            sufficient_progress
            and self.distance_increase_count
            >= self.distance_increase_required_count
        )
        
    def complete_current_goal(
        self,
        robot_x: float,
        robot_y: float,
        completion_distance: float,
        completion_reason: str,
    ) -> None:
        """Complete the current goal and schedule the next one."""

        now = time.monotonic()

        goal_duration = 0.0

        if self.current_goal_start_time is not None:
            goal_duration = (
                now - self.current_goal_start_time
            )

        route_goal = self.route[self.current_goal_index]

        self.get_logger().info(
            f"GOAL COMPLETE "
            f"{self.current_goal_index + 1}/"
            f"{len(self.route)} | "
            f"{route_goal.name} | "
            f"robot=({robot_x:.2f}, {robot_y:.2f}) | "
            f"distance={completion_distance:.2f} m | "
            f"reason={completion_reason} | "
            f"time={goal_duration:.2f} s"
        )

        # Do not clear /plan here.
        # The next Nav2 path replaces the current path.
        self.following_current_goal = False
        self.current_goal_start_time = None
        self.current_goal_index += 1

        if self.current_goal_index >= len(self.route):
            self.finish_mission()
            return

        self.next_goal_send_time = (
            now + self.between_goal_delay
        )

    def finish_mission(self) -> None:
        """Finish the lap successfully."""

        if self.mission_finished:
            return

        self.mission_finished = True
        self.following_current_goal = False
        self.clear_path()

        total_time = 0.0

        if self.mission_start_time is not None:
            total_time = (
                time.monotonic() - self.mission_start_time
            )

        self.get_logger().info(
            "========================================"
        )
        self.get_logger().info(
            "RACE MISSION COMPLETE"
        )
        self.get_logger().info(
            f"Goals completed: "
            f"{len(self.route)}/{len(self.route)}"
        )
        self.get_logger().info(
            f"Total lap time: {total_time:.2f} seconds"
        )
        self.get_logger().info(
            "Mission result: SUCCESS"
        )
        self.get_logger().info(
            "========================================"
        )

    def fail_mission(self, reason: str) -> None:
        """Stop the controller and fail the mission."""

        if self.mission_failed:
            return

        self.mission_failed = True
        self.following_current_goal = False
        self.planning_in_progress = False
        self.clear_path()

        self.get_logger().error(
            "========================================"
        )
        self.get_logger().error(
            "RACE MISSION FAILED"
        )
        self.get_logger().error(
            f"Completed goals: "
            f"{self.current_goal_index}/{len(self.route)}"
        )
        self.get_logger().error(reason)
        self.get_logger().error(
            "========================================"
        )

    def control_callback(self) -> None:
        """Main mission control loop."""

        if self.mission_finished or self.mission_failed:
            return

        now = time.monotonic()

        if not self.action_client.server_is_ready():
            if now - self.last_server_warning_time >= 2.0:
                self.get_logger().warning(
                    "Waiting for "
                    "/compute_path_to_pose action server."
                )
                self.last_server_warning_time = now

            return

        if (
            not self.planning_in_progress
            and not self.following_current_goal
        ):
            if now >= self.next_goal_send_time:
                self.send_current_goal()

            return

        if not self.following_current_goal:
            return

        robot_position = self.get_robot_position()

        if robot_position is None:
            return

        robot_x, robot_y = robot_position
        route_goal = self.route[self.current_goal_index]

        distance = self.calculate_goal_distance(
            robot_x,
            robot_y,
        )

        elapsed_time = 0.0

        if self.current_goal_start_time is not None:
            elapsed_time = (
                now - self.current_goal_start_time
            )

        # Normal distance-based completion.
        if distance <= route_goal.tolerance:
            self.complete_current_goal(
                robot_x=robot_x,
                robot_y=robot_y,
                completion_distance=distance,
                completion_reason="inside tolerance",
            )
            return

        # Intermediate goals can be completed after being passed.
        # The final goal remains strict.
        if (
            not self.is_final_goal()
            and elapsed_time >= self.minimum_goal_follow_time
        ):
            waypoint_passed = (
                self.update_pass_through_detection(distance)
            )

            if waypoint_passed:
                self.get_logger().info(
                    f"WAYPOINT PASSED | "
                    f"goal={self.current_goal_index + 1} | "
                    f"closest="
                    f"{self.closest_goal_distance:.2f} m | "
                    f"current={distance:.2f} m"
                )

                self.complete_current_goal(
                    robot_x=robot_x,
                    robot_y=robot_y,
                    completion_distance=(
                        self.closest_goal_distance
                    ),
                    completion_reason="passed waypoint",
                )
                return

        # Timeout protection.
        if elapsed_time > self.goal_timeout:
            self.fail_mission(
                f"Goal timeout | "
                f"goal={self.current_goal_index + 1} | "
                f"{route_goal.name} | "
                f"remaining={distance:.2f} m | "
                f"closest={self.closest_goal_distance:.2f} m | "
                f"tolerance={route_goal.tolerance:.2f} m"
            )
            return

        # Status output every two seconds.
        if now - self.last_status_log_time >= 2.0:
            closest_text = "not recorded"

            if math.isfinite(self.closest_goal_distance):
                closest_text = (
                    f"{self.closest_goal_distance:.2f} m"
                )

            goal_type = (
                "final"
                if self.is_final_goal()
                else "pass-through"
            )

            self.get_logger().info(
                f"MISSION STATUS | "
                f"goal="
                f"{self.current_goal_index + 1}/"
                f"{len(self.route)} | "
                f"{route_goal.name} | "
                f"type={goal_type} | "
                f"robot=({robot_x:.2f}, {robot_y:.2f}) | "
                f"remaining={distance:.2f} m | "
                f"closest={closest_text} | "
                f"tolerance={route_goal.tolerance:.2f} m"
            )

            self.last_status_log_time = now

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.clear_path()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = RaceMissionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.clear_path()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()