#include <gtest/gtest.h>

struct Point3D
{
    double x;
    double y;
    double z;
};

Point3D project_depth_pixel(
    double u,
    double v,
    double depth,
    double fx,
    double fy,
    double cx,
    double cy)
{
    Point3D point;

    point.x = (u - cx) * depth / fx;
    point.y = (v - cy) * depth / fy;
    point.z = depth;

    return point;
}

TEST(RgbdProjection, OpticalCentreProjectsStraightForward)
{
    const auto point = project_depth_pixel(
        160.0,
        120.0,
        2.0,
        216.486,
        216.486,
        160.0,
        120.0);

    EXPECT_NEAR(point.x, 0.0, 1e-6);
    EXPECT_NEAR(point.y, 0.0, 1e-6);
    EXPECT_NEAR(point.z, 2.0, 1e-6);
}

TEST(RgbdProjection, RightPixelProducesPositiveX)
{
    const auto point = project_depth_pixel(
        170.0,
        120.0,
        2.0,
        216.486,
        216.486,
        160.0,
        120.0);

    EXPECT_GT(point.x, 0.0);
    EXPECT_NEAR(point.y, 0.0, 1e-6);
    EXPECT_NEAR(point.z, 2.0, 1e-6);
}