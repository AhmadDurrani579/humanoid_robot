#!/usr/bin/env python3

import socket
import struct

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


UDP_IP = "127.0.0.1"
UDP_PORT = 15002

PACKET_FLOAT_COUNT = 13
PACKET_SIZE = struct.calcsize(
    f"{PACKET_FLOAT_COUNT}f"
)


class OdomUdpPublisher(Node):
    def __init__(self) -> None:
        super().__init__(
            "h1_odom_udp_publisher"
        )

        self.odom_publisher = (
            self.create_publisher(
                Odometry,
                "/odom",
                10,
            )
        )

        self.tf_broadcaster = (
            TransformBroadcaster(self)
        )

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        self.socket.bind(
            (
                UDP_IP,
                UDP_PORT,
            )
        )

        self.socket.setblocking(False)

        self.timer = self.create_timer(
            0.01,
            self.receive_odometry,
        )

        self.get_logger().info(
            "Listening for MuJoCo odometry on "
            f"{UDP_IP}:{UDP_PORT}"
        )

    def receive_odometry(self) -> None:
        latest_packet = None

        while True:
            try:
                packet, _ = self.socket.recvfrom(
                    4096
                )
                latest_packet = packet

            except BlockingIOError:
                break

            except OSError as error:
                self.get_logger().error(
                    f"UDP receive error: {error}"
                )
                return

        if latest_packet is None:
            return

        if len(latest_packet) != PACKET_SIZE:
            self.get_logger().warning(
                "Invalid odometry packet size: "
                f"{len(latest_packet)} bytes"
            )
            return

        (
            position_x,
            position_y,
            position_z,
            quaternion_x,
            quaternion_y,
            quaternion_z,
            quaternion_w,
            linear_velocity_x,
            linear_velocity_y,
            linear_velocity_z,
            angular_velocity_x,
            angular_velocity_y,
            angular_velocity_z,
        ) = struct.unpack(
            "13f",
            latest_packet,
        )

        timestamp = self.get_clock().now().to_msg()

        odom = Odometry()

        odom.header.stamp = timestamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "pelvis"

        odom.pose.pose.position.x = float(
            position_x
        )
        odom.pose.pose.position.y = float(
            position_y
        )
        odom.pose.pose.position.z = float(
            position_z
        )

        odom.pose.pose.orientation.x = float(
            quaternion_x
        )
        odom.pose.pose.orientation.y = float(
            quaternion_y
        )
        odom.pose.pose.orientation.z = float(
            quaternion_z
        )
        odom.pose.pose.orientation.w = float(
            quaternion_w
        )

        odom.twist.twist.linear.x = float(
            linear_velocity_x
        )
        odom.twist.twist.linear.y = float(
            linear_velocity_y
        )
        odom.twist.twist.linear.z = float(
            linear_velocity_z
        )

        odom.twist.twist.angular.x = float(
            angular_velocity_x
        )
        odom.twist.twist.angular.y = float(
            angular_velocity_y
        )
        odom.twist.twist.angular.z = float(
            angular_velocity_z
        )

        self.odom_publisher.publish(odom)

        transform = TransformStamped()

        transform.header.stamp = timestamp
        transform.header.frame_id = "world"
        transform.child_frame_id = "pelvis"

        transform.transform.translation.x = float(
            position_x
        )
        transform.transform.translation.y = float(
            position_y
        )
        transform.transform.translation.z = float(
            position_z
        )

        transform.transform.rotation.x = float(
            quaternion_x
        )
        transform.transform.rotation.y = float(
            quaternion_y
        )
        transform.transform.rotation.z = float(
            quaternion_z
        )
        transform.transform.rotation.w = float(
            quaternion_w
        )

        self.tf_broadcaster.sendTransform(
            transform
        )

    def destroy_node(self) -> bool:
        self.socket.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = OdomUdpPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()