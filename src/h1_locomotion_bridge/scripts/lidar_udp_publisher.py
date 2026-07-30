#!/usr/bin/env python3

import socket
import struct
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


UDP_IP = "127.0.0.1"
UDP_PORT = 15001

LIDAR_NUM_RAYS = 360
LIDAR_MIN_RANGE = 0.10
LIDAR_MAX_RANGE = 30.0


class LidarUdpPublisher(Node):

    def __init__(self):
        super().__init__("h1_lidar_udp_publisher")

        self.publisher = self.create_publisher(
            LaserScan,
            "/scan",
            10,
        )

        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.sock.bind(
            (
                UDP_IP,
                UDP_PORT,
            )
        )

        self.sock.setblocking(False)

        self.timer = self.create_timer(
            0.01,
            self.receive_scan,
        )

        self.get_logger().info(
            f"Listening for LiDAR UDP data on "
            f"{UDP_IP}:{UDP_PORT}"
        )

    def receive_scan(self):
        try:
            data, _ = self.sock.recvfrom(4096)
        except BlockingIOError:
            return
        except OSError as error:
            self.get_logger().error(
                f"UDP receive error: {error}"
            )
            return

        expected_size = LIDAR_NUM_RAYS * 4

        if len(data) != expected_size:
            self.get_logger().warning(
                f"Invalid LiDAR packet size: {len(data)} bytes. "
                f"Expected {expected_size} bytes."
            )
            return

        ranges = struct.unpack(
            f"{LIDAR_NUM_RAYS}f",
            data,
        )

        scan = LaserScan()

        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "lidar_link"

        scan.angle_min = -np.pi
        scan.angle_max = np.pi
        scan.angle_increment = (
            2.0 * np.pi / LIDAR_NUM_RAYS
        )

        scan.time_increment = 0.0
        scan.scan_time = 0.1

        scan.range_min = LIDAR_MIN_RANGE
        scan.range_max = LIDAR_MAX_RANGE

        scan.ranges = list(ranges)

        self.publisher.publish(scan)

    def destroy_node(self):
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = LidarUdpPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()