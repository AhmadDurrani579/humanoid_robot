#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from sensor_msgs.msg import Image


def apply_depth_model(
    depth,
    min_range,
    max_range,
    noise_base,
    noise_quadratic,
    dropout_probability,
    quantization,
    seed=None,
    rng=None,
):
    """
    Apply a simple realistic RGB-D depth sensor model.

    The clean input depth is preserved separately. This function creates
    a degraded copy with:
      - minimum/maximum usable range
      - distance-dependent Gaussian noise
      - random missing pixels
      - depth quantisation

    Depth values are in metres.
    """

    result = np.asarray(depth, dtype=np.float32).copy()

    if rng is None:
        rng = np.random.default_rng(seed)

    # Invalid source values remain invalid.
    valid = np.isfinite(result)

    # D455-like useful working range used for our simulation.
    valid &= result >= min_range
    valid &= result <= max_range

    result[~valid] = np.nan

    if np.any(valid):
        valid_depth = result[valid]

        # Noise increases with distance.
        #
        # sigma(z) = base + quadratic * z^2
        sigma = (
            noise_base
            + noise_quadratic * np.square(valid_depth)
        )

        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=valid_depth.shape,
        ).astype(np.float32)

        result[valid] = valid_depth + noise

    # Remove measurements that noise pushed outside the usable range.
    valid_after_noise = np.isfinite(result)
    valid_after_noise &= result >= min_range
    valid_after_noise &= result <= max_range
    result[~valid_after_noise] = np.nan

    # Random missing depth pixels.
    if dropout_probability > 0.0:
        dropout_mask = (
            rng.random(result.shape)
            < dropout_probability
        )

        dropout_mask &= np.isfinite(result)
        result[dropout_mask] = np.nan

    # Quantise valid measurements.
    if quantization > 0.0:
        valid_quantize = np.isfinite(result)

        result[valid_quantize] = (
            np.round(
                result[valid_quantize] / quantization
            )
            * quantization
        )

    return np.asarray(result, dtype=np.float32)


class DepthSensorModelNode(Node):

    def __init__(self):
        super().__init__("depth_sensor_model")

        self.declare_parameter(
            "input_topic",
            "/camera/depth/image_raw",
        )

        self.declare_parameter(
            "output_topic",
            "/camera/depth/image_realistic",
        )

        # Initial D455-like nominal stress-test parameters.
        # These are simulation parameters, not claimed as an exact
        # manufacturer-calibrated physical noise model.
        self.declare_parameter("min_range", 0.60)
        self.declare_parameter("max_range", 6.00)

        self.declare_parameter(
            "noise_base",
            0.002,
        )

        self.declare_parameter(
            "noise_quadratic",
            0.002,
        )

        self.declare_parameter(
            "dropout_probability",
            0.02,
        )

        self.declare_parameter(
            "quantization",
            0.001,
        )

        self.declare_parameter(
            "random_seed",
            42,
        )

        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.min_range = (
            self.get_parameter("min_range").value
        )

        self.max_range = (
            self.get_parameter("max_range").value
        )

        self.noise_base = (
            self.get_parameter("noise_base").value
        )

        self.noise_quadratic = (
            self.get_parameter("noise_quadratic").value
        )

        self.dropout_probability = (
            self.get_parameter(
                "dropout_probability"
            ).value
        )

        self.quantization = (
            self.get_parameter("quantization").value
        )

        random_seed = int(
            self.get_parameter("random_seed").value
        )

        # Persistent generator:
        # noise changes frame-to-frame but is reproducible across runs.
        self.rng = np.random.default_rng(random_seed)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            qos,
        )

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.depth_callback,
            qos,
        )

        self.frame_count = 0

        self.get_logger().info(
            "Realistic RGB-D depth sensor model started | "
            f"input={self.input_topic} | "
            f"output={self.output_topic} | "
            f"range={self.min_range:.2f}-"
            f"{self.max_range:.2f} m | "
            f"dropout={self.dropout_probability * 100:.1f}%"
        )

    def depth_callback(self, msg):
        if msg.encoding != "32FC1":
            self.get_logger().error(
                "Expected depth encoding 32FC1, "
                f"received '{msg.encoding}'"
            )
            return

        expected_pixels = msg.height * msg.width

        depth = np.frombuffer(
            msg.data,
            dtype=np.float32,
        )

        if depth.size != expected_pixels:
            self.get_logger().error(
                "Depth image size mismatch: "
                f"expected {expected_pixels} float32 values, "
                f"received {depth.size}"
            )
            return

        depth = depth.reshape(
            (msg.height, msg.width)
        )

        realistic_depth = apply_depth_model(
            depth=depth,
            min_range=self.min_range,
            max_range=self.max_range,
            noise_base=self.noise_base,
            noise_quadratic=self.noise_quadratic,
            dropout_probability=(
                self.dropout_probability
            ),
            quantization=self.quantization,
            rng=self.rng,
        )

        output = Image()

        # Keep exactly the same capture timestamp and optical frame.
        # This is important for CameraInfo synchronisation and TF.
        output.header = msg.header

        output.height = msg.height
        output.width = msg.width
        output.encoding = "32FC1"
        output.is_bigendian = False
        output.step = msg.width * 4
        output.data = realistic_depth.tobytes(
            order="C"
        )

        self.publisher.publish(output)

        self.frame_count += 1


def main(args=None):
    rclpy.init(args=args)

    node = DepthSensorModelNode()

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