#include <gtest/gtest.h>

#include "h1_rgbd_reconstruction/incremental_voxel_map.hpp"


TEST(
    IncrementalVoxelMap,
    CountsOneConfirmationPerKeyframe)
{
    IncrementalVoxelMap map(
        0.04f
    );


    // ========================================================
    // Keyframe 1
    //
    // Three points fall inside the SAME 4 cm voxel.
    // They are point observations, but this voxel should only
    // receive ONE keyframe confirmation.
    // ========================================================

    map.beginFrame();

    map.integrate(
        0.010f,
        0.010f,
        0.010f,
        255,
        0,
        0
    );

    map.integrate(
        0.015f,
        0.012f,
        0.011f,
        255,
        0,
        0
    );

    map.integrate(
        0.020f,
        0.018f,
        0.015f,
        255,
        0,
        0
    );

    map.endFrame();


    // Same three points:
    // point_count should be 3.
    //
    // But only one accepted keyframe saw the voxel:
    // frame_count should be 1.

    const auto stats_after_frame_1 =
        map.getObservationStats();

    EXPECT_EQ(
        stats_after_frame_1.total_voxels,
        1u
    );

    EXPECT_EQ(
        stats_after_frame_1.total_point_observations,
        3u
    );

    EXPECT_EQ(
        stats_after_frame_1.frame_count_1,
        1u
    );


    // ========================================================
    // Keyframe 2
    //
    // Two more points hit the SAME voxel.
    // ========================================================

    map.beginFrame();

    map.integrate(
        0.011f,
        0.011f,
        0.012f,
        0,
        255,
        0
    );

    map.integrate(
        0.019f,
        0.014f,
        0.016f,
        0,
        255,
        0
    );

    map.endFrame();


    const auto stats_after_frame_2 =
        map.getObservationStats();


    // 3 points from frame 1
    // +
    // 2 points from frame 2

    EXPECT_EQ(
        stats_after_frame_2.total_point_observations,
        5u
    );


    // Same voxel has now been confirmed by
    // TWO independent keyframes.

    EXPECT_EQ(
        stats_after_frame_2.frame_count_1,
        0u
    );

    EXPECT_EQ(
        stats_after_frame_2.frame_count_2_to_5,
        1u
    );
}