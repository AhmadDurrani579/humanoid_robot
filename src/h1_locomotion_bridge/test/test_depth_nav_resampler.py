#!/usr/bin/env python3

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "depth_nav_resampler.py"
)

spec = importlib.util.spec_from_file_location(
    "depth_nav_resampler",
    SCRIPT_PATH,
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_downsamples_depth_by_factor_two():
    depth = np.arange(
        16,
        dtype=np.float32,
    ).reshape(4, 4)

    result = module.downsample_depth(
        depth,
        factor=2,
    )

    expected = np.array(
        [
            [0.0, 2.0],
            [8.0, 10.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_equal(
        result,
        expected,
    )


def test_downsampled_depth_keeps_float32():
    depth = np.ones(
        (480, 640),
        dtype=np.float32,
    )

    result = module.downsample_depth(
        depth,
        factor=2,
    )

    assert result.shape == (240, 320)
    assert result.dtype == np.float32


def test_scales_camera_intrinsics_by_two():
    k = [
        337.20964008990796, 0.0, 320.0,
        0.0, 337.20964008990796, 240.0,
        0.0, 0.0, 1.0,
    ]

    p = [
        337.20964008990796, 0.0, 320.0, 0.0,
        0.0, 337.20964008990796, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    new_k, new_p = module.scale_camera_intrinsics(
        k,
        p,
        factor=2,
    )

    assert np.isclose(
        new_k[0],
        168.60482004495398,
    )
    assert np.isclose(
        new_k[4],
        168.60482004495398,
    )
    assert np.isclose(new_k[2], 160.0)
    assert np.isclose(new_k[5], 120.0)

    assert np.isclose(
        new_p[0],
        168.60482004495398,
    )
    assert np.isclose(
        new_p[5],
        168.60482004495398,
    )
    assert np.isclose(new_p[2], 160.0)
    assert np.isclose(new_p[6], 120.0)


def test_original_arrays_are_not_modified():
    k = [
        337.0, 0.0, 320.0,
        0.0, 337.0, 240.0,
        0.0, 0.0, 1.0,
    ]

    p = [
        337.0, 0.0, 320.0, 0.0,
        0.0, 337.0, 240.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]

    old_k = list(k)
    old_p = list(p)

    module.scale_camera_intrinsics(
        k,
        p,
        factor=2,
    )

    assert k == old_k
    assert p == old_p


def test_navigation_frame_override():
    result = module.resolve_output_frame(
        input_frame="camera_depth_optical_frame",
        output_frame="camera_depth_nav_optical_frame",
    )

    assert result == "camera_depth_nav_optical_frame"


def test_empty_override_preserves_original_frame():
    result = module.resolve_output_frame(
        input_frame="camera_depth_optical_frame",
        output_frame="",
    )

    assert result == "camera_depth_optical_frame"
