#!/usr/bin/env python3

import copy

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs.msg import CameraInfo, Image


def downsample_depth(depth, factor=2):
    """
    Downsample a depth image using pixel decimation.

    Example:
        640 x 480 with factor 2
        becomes
        320 x 240

    We deliberately use decimation rather than interpolation so that
    depth values are not artificially blended between foreground and
    background objects.
    """

    if factor < 1:
        raise ValueError("factor must be >= 1")

    depth = np.asarray(
        depth,
        dtype=np.float32,
    )

    result = depth[
        ::factor,
        ::factor,
    ]

    return np.ascontiguousarray(
        result,
        dtype=np.float32,
    )


def scale_camera_intrinsics(k, p, factor=2):
    """
    Scale CameraInfo K and P matrices for a smaller image.

    If image width and height are divided by factor, pixel-space camera
    parameters must be divided by the same factor.
    """

    if factor < 1:
        raise ValueError("factor must be >= 1")

    new_k = list(k)
    new_p = list(p)

    scale = float(factor)

    # K:
    #
    # [ fx  s  cx ]
    # [  0 fy  cy ]
    # [  0  0   1 ]
    #
    # Scale the first two rows because they are expressed in pixels.
    new_k[0] /= scale
    new_k[1] /= scale
    new_k[2] /= scale

    new_k[3] /= scale
    new_k[4] /= scale
    new_k[5] /= scale

    # P:
    #
    # [ fx'  0  cx' Tx ]
    # [  0  fy' cy' Ty ]
    # [  0   0   1   0 ]
    #
    # Again scale the first two rows.
    for index in range(0, 4):
        new_p[index] /= scale

    for index in range(4, 8):
        new_p[index] /= scale

    return new_k, new_p

def resolve_output_frame(input_frame, output_frame):
    """
    Use a dedicated navigation frame when configured.

    If output_frame is empty, preserve the original camera frame.
    """
    if output_frame:
        return output_frame

    return input_frame


class DepthNavResampler(Node):

    def __init__(self):
        super().__init__("depth_nav_resampler")

        self.declare_parameter(
            "factor",
            2,
        )

        self.declare_parameter(
            "input_depth_topic",
            "/camera/depth/image_raw",
        )

        self.declare_parameter(
            "input_info_topic",
            "/camera/depth/camera_info",
        )

        self.declare_parameter(
            "output_depth_topic",
            "/camera/depth/image_nav",
        )

        self.declare_parameter(
            "output_info_topic",
            "/camera/depth/camera_info_nav",
        )

        self.declare_parameter(
            "output_frame",
            "",
        )
        
        self.factor = int(
            self.get_parameter("factor").value
        )

        if self.factor < 1:
            raise ValueError(
                "Downsample factor must be >= 1"
            )

        self.input_depth_topic = (
            self.get_parameter(
                "input_depth_topic"
            ).value
        )

        self.input_info_topic = (
            self.get_parameter(
                "input_info_topic"
            ).value
        )

        self.output_depth_topic = (
            self.get_parameter(
                "output_depth_topic"
            ).value
        )

        self.output_info_topic = (
            self.get_parameter(
                "output_info_topic"
            ).value
        )
        
        self.output_frame = (
            self.get_parameter(
                "output_frame"
            ).value
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.depth_publisher = (
            self.create_publisher(
                Image,
                self.output_depth_topic,
                qos,
            )
        )

        self.info_publisher = (
            self.create_publisher(
                CameraInfo,
                self.output_info_topic,
                qos,
            )
        )

        self.depth_subscription = (
            self.create_subscription(
                Image,
                self.input_depth_topic,
                self.depth_callback,
                qos,
            )
        )

        self.info_subscription = (
            self.create_subscription(
                CameraInfo,
                self.input_info_topic,
                self.camera_info_callback,
                qos,
            )
        )

        self.get_logger().info(
            "Depth navigation resampler started | "
            f"factor={self.factor} | "
            f"depth={self.input_depth_topic} "
            f"-> {self.output_depth_topic} | "
            f"info={self.input_info_topic} "
            f"-> {self.output_info_topic} | "
            f"output_frame={self.output_frame}"
        )
        
    def depth_callback(self, msg):

        if msg.encoding != "32FC1":
            self.get_logger().error(
                "Expected 32FC1 depth image, "
                f"received '{msg.encoding}'"
            )
            return

        expected_pixels = (
            msg.height * msg.width
        )

        depth = np.frombuffer(
            msg.data,
            dtype=np.float32,
        )

        if depth.size != expected_pixels:
            self.get_logger().error(
                "Depth image size mismatch: "
                f"expected {expected_pixels} pixels, "
                f"received {depth.size}"
            )
            return

        depth = depth.reshape(
            msg.height,
            msg.width,
        )

        small_depth = downsample_depth(
            depth,
            factor=self.factor,
        )

        output = Image()

        # Preserve capture time and optical frame.
        output.header = msg.header
        output.header.frame_id = resolve_output_frame(
            input_frame=msg.header.frame_id,
            output_frame=self.output_frame,
        )
        
        output.height = small_depth.shape[0]
        output.width = small_depth.shape[1]

        output.encoding = "32FC1"
        output.is_bigendian = False

        output.step = (
            output.width
            * np.dtype(np.float32).itemsize
        )

        output.data = small_depth.tobytes(
            order="C"
        )

        self.depth_publisher.publish(
            output
        )

    
    def camera_info_callback(self, msg):

        output = copy.deepcopy(msg)
        
        output.header.frame_id = resolve_output_frame(
            input_frame=msg.header.frame_id,
            output_frame=self.output_frame,
        )
        
        output.width = (
            msg.width // self.factor
        )

        output.height = (
            msg.height // self.factor
        )

        new_k, new_p = (
            scale_camera_intrinsics(
                msg.k,
                msg.p,
                factor=self.factor,
            )
        )

        output.k = new_k
        output.p = new_p

        # If ROI is being used, scale it as well.
        if output.roi.x_offset:
            output.roi.x_offset //= self.factor

        if output.roi.y_offset:
            output.roi.y_offset //= self.factor

        if output.roi.width:
            output.roi.width //= self.factor

        if output.roi.height:
            output.roi.height //= self.factor

        # The output image itself has already been resampled,
        # therefore CameraInfo describes that output image directly.
        output.binning_x = 0
        output.binning_y = 0

        self.info_publisher.publish(
            output
        )


def main(args=None):

    rclpy.init(args=args)

    node = DepthNavResampler()

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
