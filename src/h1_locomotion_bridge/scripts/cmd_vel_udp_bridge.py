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

        minimum_forward_speed = 0.30

        if 0.0 < linear_x < minimum_forward_speed:
            linear_x = minimum_forward_speed
            
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