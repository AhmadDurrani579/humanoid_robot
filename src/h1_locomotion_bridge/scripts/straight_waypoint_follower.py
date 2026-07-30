#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class StraightWaypointFollower(Node):
    def __init__(self) -> None:
        super().__init__("straight_waypoint_follower")
        self.log_counter = 0
        
        self.waypoints = [
            (3.0, 0.0),
            (9.0, 0.0),
            (15.0, 0.0),
            (21.0, 0.0),
            (26.0, 0.0),

            # First corner
            (27.5, 0.4),
            (28.7, 1.2),
            (29.5, 2.2),
            (30.0, 3.5),

            # Right side
            (30.0, 6.0),
            (30.0, 9.0),

            # Second corner
            (29.7, 10.2),
            (29.0, 11.0),
            (28.0, 11.7),
            (26.5, 12.0),

            # Top straight
            (22.0, 12.0),
            (18.0, 12.0),
            (12.0, 12.0),
            (6.0, 12.0),
            (3.0, 12.0),

            # Third corner
            (1.8, 11.7),
            (0.9, 11.0),
            (0.3, 10.0),
            (0.0, 8.5),

            # Left side
            (0.0, 6.0),
            (0.0, 3.0),

            # Fourth corner
            (0.3, 1.8),
            (1.0, 0.9),
            (2.0, 0.3),
            (3.5, 0.0),

            # Finish
            (9.0, 0.0),
        ]        
        self.current_waypoint_index = 0

        self.waypoint_tolerance = 0.7
        self.max_linear_speed = 0.75
        self.max_angular_speed = 0.65
        self.heading_gain = 1.8
        
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        self.finished = False

        self.get_logger().info(
            "Straight waypoint follower started"
        )

    def odom_callback(self, msg: Odometry) -> None:
        if self.finished:
            self.publish_stop()
            return

        robot_x = msg.pose.pose.position.x
        robot_y = msg.pose.pose.position.y
        robot_yaw = self.quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

        if self.current_waypoint_index >= len(self.waypoints):
            self.finished = True
            self.publish_stop()
            self.get_logger().info(
                "All waypoints reached. Robot stopped."
            )
            return

        waypoint_x, waypoint_y = self.waypoints[
            self.current_waypoint_index
        ]

        dx = waypoint_x - robot_x
        dy = waypoint_y - robot_y

        distance = math.hypot(dx, dy)

        if distance < self.waypoint_tolerance:
            self.get_logger().info(
                f"Waypoint {self.current_waypoint_index} reached "
                f"at x={robot_x:.2f}, y={robot_y:.2f}"
            )

            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.waypoints):
                self.finished = True
                self.publish_stop()
                self.get_logger().info(
                    "Final waypoint reached. Robot stopped."
                )
                return

            waypoint_x, waypoint_y = self.waypoints[
                self.current_waypoint_index
            ]

            dx = waypoint_x - robot_x
            dy = waypoint_y - robot_y
            distance = math.hypot(dx, dy)

        target_heading = math.atan2(dy, dx)

        heading_error = self.normalize_angle(
            target_heading - robot_yaw
        )

        angular_speed = self.heading_gain * heading_error

        angular_speed = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular_speed),
        )

        abs_heading_error = abs(heading_error)

        # Large direction error: turn without moving forward
        if abs_heading_error > math.radians(40.0):
            linear_speed = 0.0

        # Strong turn
        elif abs_heading_error > math.radians(25.0):
            linear_speed = 0.15

        # Moderate turn
        elif abs_heading_error > math.radians(12.0):
            linear_speed = 0.30

        # Straight or almost straight
        else:
            linear_speed = self.max_linear_speed

        # Slow down when approaching a waypoint
        if distance < 1.2:
            linear_speed = min(linear_speed, 0.35)

        # Corner waypoints start from index 5
        if self.current_waypoint_index >= 5:
            linear_speed = min(linear_speed, 0.35)            
        
        
        command = Twist()
        command.linear.x = linear_speed
        command.angular.z = angular_speed

        self.cmd_vel_publisher.publish(command)
        self.log_counter += 1

        if self.log_counter % 10 == 0:
            self.get_logger().info(
                f"Target WP {self.current_waypoint_index}: "
                f"robot=({robot_x:.2f}, {robot_y:.2f}), "
                f"target=({waypoint_x:.2f}, {waypoint_y:.2f}), "
                f"distance={distance:.2f}, "
                f"heading_error={math.degrees(heading_error):.1f} deg, "
                f"speed={linear_speed:.2f}"
            )
            
    def publish_stop(self) -> None:
        stop_command = Twist()
        stop_command.linear.x = 0.0
        stop_command.angular.z = 0.0
        self.cmd_vel_publisher.publish(stop_command)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    @staticmethod
    def quaternion_to_yaw(
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> float:
        sin_yaw = 2.0 * ((w * z) + (x * y))
        cos_yaw = 1.0 - (2.0 * ((y * y) + (z * z)))

        return math.atan2(sin_yaw, cos_yaw)


def main(args=None) -> None:
    rclpy.init(args=args)

    node = StraightWaypointFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()