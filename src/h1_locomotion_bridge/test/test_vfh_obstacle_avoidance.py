import importlib.util
import math
from pathlib import Path
import sys

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "vfh_obstacle_avoidance.py"
)

spec = importlib.util.spec_from_file_location(
    "vfh",
    SCRIPT_PATH,
)

vfh = importlib.util.module_from_spec(spec)

assert spec.loader is not None

# Register the dynamically loaded module before executing it.
# Dataclasses use sys.modules to resolve the class module.
sys.modules[spec.name] = vfh

spec.loader.exec_module(vfh)

def test_normalise_angle_wraps_to_pi_interval():
    assert math.isclose(
        vfh.normalise_angle(
            3.0 * math.pi
        ),
        -math.pi,
    )

    assert math.isclose(
        vfh.normalise_angle(
            -3.0 * math.pi
        ),
        -math.pi,
    )

    assert math.isclose(
        vfh.normalise_angle(0.25),
        0.25,
    )


def test_sanitise_scan_replaces_invalid_values():
    result = vfh.sanitise_scan(
        [
            float("nan"),
            float("inf"),
            -1.0,
            0.05,
            0.5,
            31.0,
        ],
        range_min=0.10,
        range_max=30.0,
    )

    expected = np.array(
        [
            30.0,
            30.0,
            30.0,
            30.0,
            0.5,
            30.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_extract_front_scan_ignores_rear_rays():
    ranges = np.full(
        360,
        30.0,
    )

    ranges[0] = 0.2
    ranges[180] = 1.0

    angles, front_ranges = (
        vfh.extract_front_scan(
            ranges=ranges,
            angle_min=-math.pi,
            angle_increment=math.radians(1.0),
            front_fov_deg=180.0,
        )
    )

    assert np.min(front_ranges) == 1.0

    assert np.all(
        angles >= -math.pi / 2.0
    )

    assert np.all(
        angles <= math.pi / 2.0
    )


def test_build_sector_histogram_gives_higher_density_to_close_obstacle():
    config = vfh.VfhConfig()

    angles = np.radians(
        np.array(
            [
                -2.0,
                -1.0,
                0.0,
                1.0,
                2.0,
                40.0,
            ]
        )
    )

    ranges = np.array(
        [
            1.0,
            0.8,
            0.6,
            0.8,
            1.0,
            5.0,
        ]
    )

    centres, density = (
        vfh.build_sector_histogram(
            angles,
            ranges,
            config,
        )
    )

    centre_index = int(
        np.argmin(
            np.abs(centres)
        )
    )

    far_index = int(
        np.argmin(
            np.abs(
                centres
                - math.radians(40.0)
            )
        )
    )

    assert (
        density[centre_index]
        > density[far_index]
    )