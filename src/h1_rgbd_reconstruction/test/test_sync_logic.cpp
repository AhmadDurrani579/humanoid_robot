#include <gtest/gtest.h>
#include <cmath>

bool timestamps_close(
    double rgb_time,
    double depth_time,
    double tolerance_sec)
{
    return std::fabs(rgb_time - depth_time) <= tolerance_sec;
}

TEST(RgbdSyncLogic, AcceptsCloseTimestamps)
{
    const double rgb_time = 10.000;
    const double depth_time = 10.015;

    EXPECT_TRUE(
        timestamps_close(
            rgb_time,
            depth_time,
            0.05));
}

TEST(RgbdSyncLogic, RejectsFarTimestamps)
{
    const double rgb_time = 10.000;
    const double depth_time = 10.200;

    EXPECT_FALSE(
        timestamps_close(
            rgb_time,
            depth_time,
            0.05));
}