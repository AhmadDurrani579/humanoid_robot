#!/usr/bin/env python3

import socket
import struct
import threading
from typing import Optional, Tuple
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


CAMERA_UDP_IP = "127.0.0.1"
CAMERA_UDP_PORT = 15003

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_FPS = 15.0

COLOR_FRAME_ID = "camera_color_optical_frame"
DEPTH_FRAME_ID = "camera_depth_optical_frame"

COLOR_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * 3
DEPTH_BYTES = IMAGE_WIDTH * IMAGE_HEIGHT * 4

PACKET_HEADER_FORMAT = "<4sBIII"
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)

MAGIC = b"H1CM"
FRAME_TYPE_COLOR = 1
FRAME_TYPE_DEPTH = 2


class FrameAssembler:
    def __init__(self) -> None:
        self.frames = {}
        self.completed_color = 0
        self.completed_depth = 0
        self.last_report_time = time.monotonic()

    def add_chunk(
        self,
        frame_type: int,
        frame_id: int,
        chunk_index: int,
        chunk_count: int,
        payload: bytes,
    ) -> Optional[bytes]:
        key = (frame_type, frame_id)

        if key not in self.frames:
            self.frames[key] = {
                "chunk_count": chunk_count,
                "chunks": {},
            }

        frame = self.frames[key]

        if frame["chunk_count"] != chunk_count:
            del self.frames[key]
            return None

        frame["chunks"][chunk_index] = payload

        if len(frame["chunks"]) != chunk_count:
            return None

        complete_frame = b"".join(
            frame["chunks"][index]
            for index in range(chunk_count)
        )

        if frame_type == FRAME_TYPE_COLOR:
            self.completed_color += 1
        elif frame_type == FRAME_TYPE_DEPTH:
            self.completed_depth += 1

        now = time.monotonic()

        if now - self.last_report_time >= 5.0:
            print(
                "UDP completed frames | "
                f"color={self.completed_color} | "
                f"depth={self.completed_depth}"
            )

            self.completed_color = 0
            self.completed_depth = 0
            self.last_report_time = now
            
        del self.frames[key]

        old_keys = [
            existing_key
            for existing_key in self.frames
            if existing_key[1] < frame_id - 10
        ]

        for old_key in old_keys:
            del self.frames[old_key]

        return complete_frame


class CameraBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_bridge_node")

        self.color_publisher = self.create_publisher(
            Image,
            "/camera/color/image_raw",
            10,
        )

        self.color_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/color/camera_info",
            10,
        )

        self.depth_publisher = self.create_publisher(
            Image,
            "/camera/depth/image_raw",
            10,
        )

        self.depth_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/depth/camera_info",
            10,
        )

        self.frame_assembler = FrameAssembler()

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
            4 * 1024 * 1024,
        )

        self.socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )
        
        actual_rcvbuf = self.socket.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_RCVBUF,
        )

        self.get_logger().info(
            f"Camera UDP receive buffer: {actual_rcvbuf} bytes"
        )

        self.socket.bind(
            (CAMERA_UDP_IP, CAMERA_UDP_PORT)
        )

        self.socket.settimeout(0.2)

        self.running = True

        self.receiver_thread = threading.Thread(
            target=self.receive_loop,
            daemon=True,
        )

        self.receiver_thread.start()

        self.get_logger().info(
            "Camera bridge listening on "
            f"{CAMERA_UDP_IP}:{CAMERA_UDP_PORT}"
        )

    def create_camera_info(
        self,
        frame_id: str,
    ) -> CameraInfo:
        message = CameraInfo()

        message.header.frame_id = frame_id
        message.width = IMAGE_WIDTH
        message.height = IMAGE_HEIGHT

        # Must match MuJoCo camera:
        # <camera ... fovy="58" />
        vertical_fov_degrees = 58.0
        vertical_fov_radians = np.deg2rad(
            vertical_fov_degrees
        )

        focal_length_y = (
            IMAGE_HEIGHT
            / (
                2.0
                * np.tan(
                    vertical_fov_radians / 2.0
                )
            )
        )

        # MuJoCo perspective camera uses square pixels here.
        focal_length_x = focal_length_y

        center_x = IMAGE_WIDTH / 2.0
        center_y = IMAGE_HEIGHT / 2.0

        message.distortion_model = "plumb_bob"

        message.d = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        message.k = [
            focal_length_x,
            0.0,
            center_x,
            0.0,
            focal_length_y,
            center_y,
            0.0,
            0.0,
            1.0,
        ]

        message.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        message.p = [
            focal_length_x,
            0.0,
            center_x,
            0.0,
            0.0,
            focal_length_y,
            center_y,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        return message
    
    
    def publish_color(self, frame: bytes) -> None:
        if len(frame) != COLOR_BYTES:
            self.get_logger().warning(
                "Ignoring invalid RGB frame size: "
                f"{len(frame)}"
            )
            return

        stamp = self.get_clock().now().to_msg()

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = COLOR_FRAME_ID

        image.height = IMAGE_HEIGHT
        image.width = IMAGE_WIDTH
        image.encoding = "rgb8"
        image.is_bigendian = False
        image.step = IMAGE_WIDTH * 3
        image.data = frame

        info = self.create_camera_info(
            COLOR_FRAME_ID
        )
        info.header.stamp = stamp

        self.color_publisher.publish(image)
        self.color_info_publisher.publish(info)

    def publish_depth(self, frame: bytes) -> None:
        if len(frame) != DEPTH_BYTES:
            self.get_logger().warning(
                "Ignoring invalid depth frame size: "
                f"{len(frame)}"
            )
            return

        stamp = self.get_clock().now().to_msg()

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = DEPTH_FRAME_ID

        image.height = IMAGE_HEIGHT
        image.width = IMAGE_WIDTH
        image.encoding = "32FC1"
        image.is_bigendian = False
        image.step = IMAGE_WIDTH * 4
        image.data = frame

        info = self.create_camera_info(
            DEPTH_FRAME_ID
        )
        info.header.stamp = stamp

        self.depth_publisher.publish(image)
        self.depth_info_publisher.publish(info)

    def receive_loop(self) -> None:
        while self.running:
            try:
                packet, _ = self.socket.recvfrom(65535)

            except socket.timeout:
                continue

            except OSError:
                break

            if len(packet) < PACKET_HEADER_SIZE:
                continue

            (
                magic,
                frame_type,
                frame_id,
                chunk_index,
                chunk_count,
            ) = struct.unpack(
                PACKET_HEADER_FORMAT,
                packet[:PACKET_HEADER_SIZE],
            )

            if magic != MAGIC:
                continue

            payload = packet[PACKET_HEADER_SIZE:]

            complete_frame = self.frame_assembler.add_chunk(
                frame_type,
                frame_id,
                chunk_index,
                chunk_count,
                payload,
            )

            if complete_frame is None:
                continue

            if frame_type == FRAME_TYPE_COLOR:
                self.publish_color(complete_frame)

            elif frame_type == FRAME_TYPE_DEPTH:
                self.publish_depth(complete_frame)

    def destroy_node(self) -> bool:
        self.running = False

        try:
            self.socket.close()
        except OSError:
            pass

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = CameraBridgeNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
