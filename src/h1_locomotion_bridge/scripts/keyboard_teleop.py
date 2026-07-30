#!/usr/bin/env python3

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP_MESSAGE = """
H1 Keyboard Control
-------------------
W : Forward
S : Backward
A : Move left
D : Move right
Q : Turn left
E : Turn right
X : Idle gait
Ctrl+C : Exit
"""


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("h1_keyboard_teleop")

        self.publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.forward_speed = 0.8
        self.backward_speed = 0.25
        self.side_speed = 0.2
        self.turn_speed = 0.4
        self.idle_forward_speed = 0.0        
        
        self.terminal_settings = termios.tcgetattr(
            sys.stdin
        )

        # Store the command currently selected by the keyboard.
        self.current_command = self.create_command(
            linear_x=self.idle_forward_speed
        )

        # Publish the selected command continuously at 10 Hz.
        self.publish_timer = self.create_timer(
            0.1,
            self.publish_current_command,
        )

        self.get_logger().info(HELP_MESSAGE)

    def create_command(
        self,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
    ):
        command = Twist()

        command.linear.x = float(linear_x)
        command.linear.y = float(linear_y)
        command.angular.z = float(angular_z)

        return command

    def publish_current_command(self):
        """Publish the selected command continuously."""

        self.publisher.publish(
            self.current_command
        )

    def get_key(self):
        """Read one keyboard key without pressing Enter."""

        tty.setraw(sys.stdin.fileno())

        readable, _, _ = select.select(
            [sys.stdin],
            [],
            [],
            0.05,
        )

        key = (
            sys.stdin.read(1)
            if readable
            else ""
        )

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            self.terminal_settings,
        )

        return key.lower()

    def update_command(self, key):
        """Update the current command from a keyboard key."""

        if key == "w":
            self.current_command = self.create_command(
                linear_x=self.forward_speed
            )
            description = "Forward selected"

        elif key == "s":
            self.current_command = self.create_command(
                linear_x=-self.backward_speed
            )
            description = "Backward selected"

        elif key == "a":
            self.current_command = self.create_command(
                linear_y=self.side_speed
            )
            description = "Left selected"

        elif key == "d":
            self.current_command = self.create_command(
                linear_y=-self.side_speed
            )
            description = "Right selected"

        elif key == "q":
            self.current_command = self.create_command(
                linear_x=0.0,
                angular_z=self.turn_speed,
            )
            description = "Turn left selected"

        elif key == "e":
            self.current_command = self.create_command(
                linear_x=0.0,
                angular_z=-self.turn_speed,
            )
            description = "Turn right selected"

        elif key == "x" or key == " ":
            self.current_command = self.create_command(
                linear_x=0.0,
                linear_y=0.0,
                angular_z=0.0,
            )

            description = "Idle gait selected"

        else:
            return

        self.get_logger().info(
            f"{description}: "
            f"x={self.current_command.linear.x:.2f}, "
            f"y={self.current_command.linear.y:.2f}, "
            f"yaw={self.current_command.angular.z:.2f}"
        )

    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key()

                if key == "\x03":
                    break

                if key:
                    self.update_command(key)

                # Process the 10 Hz publisher timer.
                rclpy.spin_once(
                    self,
                    timeout_sec=0.0,
                )

        finally:
            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                self.terminal_settings,
            )

            self.current_command = self.create_command(
                linear_x=self.idle_forward_speed
            )

            self.publisher.publish(
                self.current_command
            )


def main(args=None):
    rclpy.init(args=args)

    node = KeyboardTeleop()

    try:
        node.run()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()