#include <gtest/gtest.h>
#include <utility>


std::pair<int, int> map_depth_pixel_to_rgb(
    int depth_u,
    int depth_v,
    int depth_width,
    int depth_height,
    int rgb_width,
    int rgb_height)
{
    const double scale_x =
        static_cast<double>(rgb_width) /
        static_cast<double>(depth_width);

    const double scale_y =
        static_cast<double>(rgb_height) /
        static_cast<double>(depth_height);

    const int rgb_u =
        static_cast<int>(depth_u * scale_x);

    const int rgb_v =
        static_cast<int>(depth_v * scale_y);

    return {
        rgb_u,
        rgb_v
    };
}


TEST(
    RgbMapping,
    CentreDepthPixelMapsToCentreRgbPixel)
{
    const auto [u, v] =
        map_depth_pixel_to_rgb(
            160,
            120,
            320,
            240,
            640,
            480
        );

    EXPECT_EQ(u, 320);
    EXPECT_EQ(v, 240);
}


TEST(
    RgbMapping,
    QuarterPixelMapsCorrectly)
{
    const auto [u, v] =
        map_depth_pixel_to_rgb(
            80,
            60,
            320,
            240,
            640,
            480
        );

    EXPECT_EQ(u, 160);
    EXPECT_EQ(v, 120);
}