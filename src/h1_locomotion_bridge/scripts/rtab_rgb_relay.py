#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image


class RtabRgbRelay(Node):

    def __init__(self):
        super().__init__("rtab_rgb_relay")

        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.callback,
            10,
        )

        self.publisher = self.create_publisher(
            Image,
            "/camera/color/image_rtab",
            10,
        )

        self.get_logger().info(
            "RTAB RGB relay started | "
            "640x480 -> 320x240 | "
            "frame=camera_depth_nav_optical_frame"
        )


    def callback(self, msg):

        # ----------------------------------------------------
        # Validate expected RGB input
        # ----------------------------------------------------

        if msg.encoding != "rgb8":
            self.get_logger().error(
                f"Expected rgb8, received {msg.encoding}"
            )
            return


        if msg.width != 640 or msg.height != 480:
            self.get_logger().error(
                "Unexpected RGB size: "
                f"{msg.width}x{msg.height}"
            )
            return


        # ----------------------------------------------------
        # ROS Image -> NumPy RGB image
        # ----------------------------------------------------

        rgb = np.frombuffer(
            msg.data,
            dtype=np.uint8
        ).reshape(
            msg.height,
            msg.width,
            3
        )


        # ----------------------------------------------------
        # Downsample RGB to match navigation depth:
        #
        # RGB                  640x480
        # nav depth            320x240
        # nav CameraInfo       320x240
        #
        # Keep RGB channel ordering unchanged.
        # ----------------------------------------------------

        rgb_small = cv2.resize(
            rgb,
            (320, 240),
            interpolation=cv2.INTER_AREA
        )


        # ----------------------------------------------------
        # Publish RTAB-specific RGB image
        # ----------------------------------------------------

        output = Image()

        output.header.stamp = msg.header.stamp

        output.header.frame_id = (
            "camera_depth_nav_optical_frame"
        )

        output.height = 240
        output.width = 320

        output.encoding = "rgb8"

        output.is_bigendian = False

        output.step = 320 * 3

        output.data = (
            rgb_small.tobytes()
        )


        self.publisher.publish(
            output
        )


def main(args=None):

    rclpy.init(args=args)

    node = RtabRgbRelay()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()