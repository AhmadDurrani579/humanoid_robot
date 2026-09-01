#!/usr/bin/env python3

from pathlib import Path
import math
import numpy as np

# ============================================================
# Map configuration
# ============================================================

RESOLUTION = 0.05       # metres / cell

X_MIN = -2.0
X_MAX = 27.0

Y_MIN = -6.0
Y_MAX = 8.0

WIDTH = int(round((X_MAX - X_MIN) / RESOLUTION))
HEIGHT = int(round((Y_MAX - Y_MIN) / RESOLUTION))

# Nav2 PGM convention used here:
# 254 = free
#   0 = occupied
#
# Start with EVERYTHING occupied.
# Then carve out only the actual semantic course.
grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


# ============================================================
# Coordinate helpers
# ============================================================

def world_to_cell(x, y):
    """
    Convert MuJoCo/world XY coordinates into PGM image coordinates.
    PGM row 0 is at the top of the image.
    """

    col = int((x - X_MIN) / RESOLUTION)
    row_from_bottom = int((y - Y_MIN) / RESOLUTION)

    row = HEIGHT - 1 - row_from_bottom

    return col, row


def cell_to_world(col, row):
    """
    Return centre of a map cell in world coordinates.
    """

    x = X_MIN + (col + 0.5) * RESOLUTION
    y = Y_MAX - (row + 0.5) * RESOLUTION

    return x, y


# ============================================================
# Free-space helpers
# ============================================================

def mark_free_box(cx, cy, half_x, half_y):
    """Carve an axis-aligned rectangular free-space region."""

    x0 = cx - half_x
    x1 = cx + half_x
    y0 = cy - half_y
    y1 = cy + half_y

    c0, r0 = world_to_cell(x0, y0)
    c1, r1 = world_to_cell(x1, y1)

    c_min = max(0, min(c0, c1))
    c_max = min(WIDTH - 1, max(c0, c1))

    r_min = max(0, min(r0, r1))
    r_max = min(HEIGHT - 1, max(r0, r1))

    grid[r_min:r_max + 1, c_min:c_max + 1] = 254


def mark_free_rotated_box(cx, cy, half_x, half_y, yaw):
    """
    Carve a rotated rectangular free-space region.

    yaw is in radians.
    """

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    # Bounding radius so we only inspect nearby cells.
    radius = math.hypot(half_x, half_y)

    x0 = cx - radius
    x1 = cx + radius
    y0 = cy - radius
    y1 = cy + radius

    c0, r0 = world_to_cell(x0, y0)
    c1, r1 = world_to_cell(x1, y1)

    c_min = max(0, min(c0, c1))
    c_max = min(WIDTH - 1, max(c0, c1))

    r_min = max(0, min(r0, r1))
    r_max = min(HEIGHT - 1, max(r0, r1))

    for row in range(r_min, r_max + 1):
        for col in range(c_min, c_max + 1):

            x, y = cell_to_world(col, row)

            dx = x - cx
            dy = y - cy

            # Transform the world point into the local frame
            # of the rotated MuJoCo box.
            local_x = cos_yaw * dx + sin_yaw * dy
            local_y = -sin_yaw * dx + cos_yaw * dy

            if (
                abs(local_x) <= half_x
                and abs(local_y) <= half_y
            ):
                grid[row, col] = 254


# ============================================================
# Occupied-obstacle helpers
# ============================================================

def mark_box(cx, cy, half_x, half_y):
    """Mark an axis-aligned rectangular obstacle."""

    x0 = cx - half_x
    x1 = cx + half_x
    y0 = cy - half_y
    y1 = cy + half_y

    c0, r0 = world_to_cell(x0, y0)
    c1, r1 = world_to_cell(x1, y1)

    c_min = max(0, min(c0, c1))
    c_max = min(WIDTH - 1, max(c0, c1))

    r_min = max(0, min(r0, r1))
    r_max = min(HEIGHT - 1, max(r0, r1))

    grid[r_min:r_max + 1, c_min:c_max + 1] = 0


def mark_circle(cx, cy, radius):
    """Mark a circular obstacle."""

    c_center, r_center = world_to_cell(cx, cy)

    radius_cells = int(math.ceil(radius / RESOLUTION))

    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):

            distance_sq = (
                (dc * RESOLUTION) ** 2
                + (dr * RESOLUTION) ** 2
            )

            if distance_sq <= radius ** 2:

                col = c_center + dc
                row = r_center - dr

                if (
                    0 <= col < WIDTH
                    and 0 <= row < HEIGHT
                ):
                    grid[row, col] = 0


# ============================================================
# STEP 1: CARVE THE SEMANTIC COURSE
# ============================================================

# ------------------------------------------------------------
# Straight walkway
#
# scene.xml:
# pos = (8, 0)
# size = (8, 2.4)
#
# Therefore:
# x = 0 -> 16
# y = -2.4 -> +2.4
# ------------------------------------------------------------

mark_free_box(
    cx=7.5,
    cy=0.0,
    half_x=8.5,
    half_y=2.4,
)

# ------------------------------------------------------------
# Mild 30-degree bend
#
# scene.xml:
# pos = (17.5, 1.5)
# size = (2.5, 2.4)
# yaw = 30 degrees
# ------------------------------------------------------------

mark_free_rotated_box(
    cx=17.5,
    cy=1.5,
    half_x=2.5,
    half_y=2.4,
    yaw=math.radians(30.0),
)


# ------------------------------------------------------------
# Finish walkway
#
# scene.xml:
# pos = (21, 3.5)
# size = (3.5, 2.4)
# ------------------------------------------------------------

mark_free_box(
    cx=21.0,
    cy=3.5,
    half_x=3.5,
    half_y=2.4,
)


# ============================================================
# STEP 2: ADD STATIC STRUCTURAL OBSTACLES
# ============================================================

# ------------------------------------------------------------
# Straight corridor rails
#
# MuJoCo size values are HALF-SIZES.
# ------------------------------------------------------------

mark_box(
    cx=8.0,
    cy=3.1,
    half_x=8.4,
    half_y=0.10,
)

mark_box(
    cx=8.0,
    cy=-3.1,
    half_x=8.4,
    half_y=0.10,
)


# ------------------------------------------------------------
# Chair 1
# GT = (8.0, 1.5)
# ------------------------------------------------------------

mark_box(
    cx=8.0,
    cy=1.5,
    half_x=0.35,
    half_y=0.35,
)


# ------------------------------------------------------------
# Backpack
# GT = (8.8, -1.4)
# ------------------------------------------------------------

mark_box(
    cx=8.8,
    cy=-1.4,
    half_x=0.22,
    half_y=0.16,
)


# ------------------------------------------------------------
# Table
# GT = (10.5, 1.6)
# ------------------------------------------------------------

mark_box(
    cx=10.5,
    cy=1.6,
    half_x=0.60,
    half_y=0.40,
)


# ------------------------------------------------------------
# Suitcase
# GT = (11.4, -1.5)
# ------------------------------------------------------------

mark_box(
    cx=11.4,
    cy=-1.5,
    half_x=0.28,
    half_y=0.15,
)


# ------------------------------------------------------------
# Navigation cones
# ------------------------------------------------------------

mark_circle(
    cx=12.7,
    cy=1.25,
    radius=0.18,
)

mark_circle(
    cx=13.5,
    cy=-1.25,
    radius=0.18,
)

mark_circle(
    cx=14.3,
    cy=1.25,
    radius=0.18,
)


# ------------------------------------------------------------
# Corridor boxes
# ------------------------------------------------------------

mark_box(
    cx=13.2,
    cy=2.25,
    half_x=0.35,
    half_y=0.35,
)

mark_box(
    cx=14.7,
    cy=-2.15,
    half_x=0.45,
    half_y=0.35,
)


# ------------------------------------------------------------
# Gate posts
#
# body position = (15.8, 0)
# local Y = +/-1.25
#
# Overhead gate beam is intentionally NOT added to the 2D map.
# ------------------------------------------------------------

mark_box(
    cx=15.8,
    cy=1.25,
    half_x=0.12,
    half_y=0.12,
)

mark_box(
    cx=15.8,
    cy=-1.25,
    half_x=0.12,
    half_y=0.12,
)


# ------------------------------------------------------------
# Chair 2
# GT = (18.2, 3.6)
# ------------------------------------------------------------

mark_box(
    cx=18.2,
    cy=3.6,
    half_x=0.35,
    half_y=0.35,
)


# ------------------------------------------------------------
# Sign post
# GT = (19.0, 1.0)
#
# Only ground-level post is represented.
# ------------------------------------------------------------

mark_box(
    cx=19.0,
    cy=1.0,
    half_x=0.05,
    half_y=0.05,
)


# ------------------------------------------------------------
# Cone 4
# ------------------------------------------------------------

mark_circle(
    cx=19.6,
    cy=2.2,
    radius=0.18,
)


# ============================================================
# INTENTIONALLY NOT IN STATIC MAP
# ============================================================
#
# semantic_person_01
# semantic_person_02
# semantic_person_03
# semantic_person_04
# semantic_person_05
#
# These are semantic perception targets.
#
# Also omitted:
#
# - start markers
# - finish markers
# - overhead gate beam
# - finish overhead beam
# - decorative floor geometry
#
# People will later be represented through:
#
# RGB + YOLO
#       +
# Depth
#       ↓
# 3D semantic localisation
#       ↓
# map-frame semantic markers
# ============================================================


# ============================================================
# Save files
# ============================================================

output_dir = Path(__file__).resolve().parent

pgm_path = output_dir / "semantic_world_map.pgm"
yaml_path = output_dir / "semantic_world_map.yaml"


# ------------------------------------------------------------
# Write binary PGM P5
# ------------------------------------------------------------

with open(pgm_path, "wb") as f:
    f.write(b"P5\n")
    f.write(f"{WIDTH} {HEIGHT}\n".encode())
    f.write(b"255\n")
    f.write(grid.tobytes())


# ------------------------------------------------------------
# Write Nav2 YAML
# ------------------------------------------------------------

yaml_text = f"""image: semantic_world_map.pgm
mode: trinary
resolution: {RESOLUTION}
origin: [{X_MIN}, {Y_MIN}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""

yaml_path.write_text(yaml_text)


# ============================================================
# Summary
# ============================================================

free_cells = int(np.count_nonzero(grid == 254))
occupied_cells = int(np.count_nonzero(grid == 0))

print("Semantic world map generated successfully")
print()
print(f"PGM:  {pgm_path}")
print(f"YAML: {yaml_path}")
print()
print(f"Resolution: {RESOLUTION:.3f} m/cell")
print(f"Map size:   {WIDTH} x {HEIGHT} cells")
print(f"World X:    {X_MIN:.1f} -> {X_MAX:.1f} m")
print(f"World Y:    {Y_MIN:.1f} -> {Y_MAX:.1f} m")
print()
print(f"Free cells:     {free_cells}")
print(f"Occupied cells: {occupied_cells}")