#include <gtest/gtest.h>

#include "h1_rgbd_reconstruction/incremental_voxel_map.hpp"


TEST(
    IncrementalVoxelMap,
    RepeatedNearbyPointsUseSingleVoxel)
{
    IncrementalVoxelMap map(0.04f);

    map.integrate(
        1.000f, 2.000f, 0.500f,
        100, 120, 140
    );

    map.integrate(
        1.010f, 2.010f, 0.510f,
        120, 140, 160
    );

    EXPECT_EQ(
        map.size(),
        1u
    );
}


TEST(
    IncrementalVoxelMap,
    SeparatePointsCreateSeparateVoxels)
{
    IncrementalVoxelMap map(0.04f);

    map.integrate(
        1.00f, 2.00f, 0.50f,
        100, 120, 140
    );

    map.integrate(
        1.20f, 2.00f, 0.50f,
        100, 120, 140
    );

    EXPECT_EQ(
        map.size(),
        2u
    );
}


TEST(
    IncrementalVoxelMap,
    RepeatedObservationsAreAveraged)
{
    IncrementalVoxelMap map(0.04f);

    map.integrate(
        1.000f, 2.000f, 0.500f,
        100, 120, 140
    );

    map.integrate(
        1.010f, 2.010f, 0.510f,
        120, 140, 160
    );

    const auto voxels =
        map.toPointCloud();

    ASSERT_EQ(
        voxels.points.size(),
        1u
    );

    const auto & p =
        voxels.points.front();

    EXPECT_NEAR(
        p.x,
        1.005f,
        0.001f
    );

    EXPECT_NEAR(
        p.y,
        2.005f,
        0.001f
    );

    EXPECT_NEAR(
        p.z,
        0.505f,
        0.001f
    );

    EXPECT_NEAR(
        static_cast<float>(p.r),
        110.0f,
        1.0f
    );

    EXPECT_NEAR(
        static_cast<float>(p.g),
        130.0f,
        1.0f
    );

    EXPECT_NEAR(
        static_cast<float>(p.b),
        150.0f,
        1.0f
    );
}