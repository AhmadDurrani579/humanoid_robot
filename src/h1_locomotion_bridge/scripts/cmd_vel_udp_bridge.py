#!/usr/bin/env python3

import socket
import struct

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelUdpBridge(Node):
    def __init__(self):
        super().__init__("h1_cmd_vel_udp_bridge")

        self.udp_host = "127.0.0.1"
        self.udp_port = 15000

        self.udp_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.subscription = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        self.get_logger().info(
            f"Forwarding /cmd_vel to "
            f"{self.udp_host}:{self.udp_port}"
        )

    def cmd_vel_callback(self, message):
        linear_x = float(message.linear.x)
        linear_y = float(message.linear.y)
        angular_z = float(message.angular.z)

        packet = struct.pack(
            "fff",
            linear_x,
            linear_y,
            angular_z,
        )

        self.udp_socket.sendto(
            packet,
            (self.udp_host, self.udp_port),
        )

        self.get_logger().info(
            f"Sent command: "
            f"x={linear_x:.2f}, "
            f"y={linear_y:.2f}, "
            f"yaw={angular_z:.2f}"
        )

    def destroy_node(self):
        self.udp_socket.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CmdVelUdpBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()