#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "depth_sensor_model.py"
)

spec = importlib.util.spec_from_file_location(
    "depth_sensor_model",
    SCRIPT_PATH,
)

depth_sensor_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(depth_sensor_model)


def test_invalidates_depth_outside_sensor_range():
    depth = np.array(
        [[0.30, 1.00, 2.50, 7.00]],
        dtype=np.float32,
    )

    result = depth_sensor_model.apply_depth_model(
        depth,
        min_range=0.6,
        max_range=6.0,
        noise_base=0.0,
        noise_quadratic=0.0,
        dropout_probability=0.0,
        quantization=0.001,
        seed=1,
    )

    assert np.isnan(result[0, 0])
    assert np.isclose(result[0, 1], 1.0)
    assert np.isclose(result[0, 2], 2.5)
    assert np.isnan(result[0, 3])


def test_quantizes_valid_depth():
    depth = np.array(
        [[2.4574]],
        dtype=np.float32,
    )

    result = depth_sensor_model.apply_depth_model(
        depth,
        min_range=0.6,
        max_range=6.0,
        noise_base=0.0,
        noise_quadratic=0.0,
        dropout_probability=0.0,
        quantization=0.001,
        seed=1,
    )

    assert np.isclose(result[0, 0], 2.457, atol=1e-6)


def test_fixed_seed_produces_repeatable_noise():
    depth = np.full(
        (20, 20),
        3.0,
        dtype=np.float32,
    )

    first = depth_sensor_model.apply_depth_model(
        depth,
        min_range=0.6,
        max_range=6.0,
        noise_base=0.002,
        noise_quadratic=0.002,
        dropout_probability=0.02,
        quantization=0.001,
        seed=42,
    )

    second = depth_sensor_model.apply_depth_model(
        depth,
        min_range=0.6,
        max_range=6.0,
        noise_base=0.002,
        noise_quadratic=0.002,
        dropout_probability=0.02,
        quantization=0.001,
        seed=42,
    )

    np.testing.assert_equal(first, second)


def test_output_keeps_shape_and_float32_type():
    depth = np.ones(
        (480, 640),
        dtype=np.float32,
    ) * 2.0

    result = depth_sensor_model.apply_depth_model(
        depth,
        min_range=0.6,
        max_range=6.0,
        noise_base=0.0,
        noise_quadratic=0.0,
        dropout_probability=0.0,
        quantization=0.001,
        seed=1,
    )

    assert result.shape == (480, 640)
    assert result.dtype == np.float32
