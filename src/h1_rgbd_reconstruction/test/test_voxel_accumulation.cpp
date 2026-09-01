#include <gtest/gtest.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>


TEST(
    VoxelAccumulation,
    CollapsesNearbyDuplicatePoints)
{
    auto cloud =
        std::make_shared<
            pcl::PointCloud<pcl::PointXYZRGB>
        >();

    pcl::PointXYZRGB p1;
    p1.x = 1.000f;
    p1.y = 2.000f;
    p1.z = 0.500f;

    pcl::PointXYZRGB p2;
    p2.x = 1.010f;
    p2.y = 2.010f;
    p2.z = 0.510f;

    cloud->points.push_back(p1);
    cloud->points.push_back(p2);

    cloud->width = 2;
    cloud->height = 1;


    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;

    voxel.setInputCloud(
        cloud
    );

    voxel.setLeafSize(
        0.04f,
        0.04f,
        0.04f
    );


    pcl::PointCloud<
        pcl::PointXYZRGB
    > filtered;

    voxel.filter(
        filtered
    );


    EXPECT_EQ(
        filtered.points.size(),
        1u
    );
}


TEST(
    VoxelAccumulation,
    KeepsSpatiallySeparatePoints)
{
    auto cloud =
        std::make_shared<
            pcl::PointCloud<pcl::PointXYZRGB>
        >();

    pcl::PointXYZRGB p1;
    p1.x = 1.0f;
    p1.y = 2.0f;
    p1.z = 0.5f;

    pcl::PointXYZRGB p2;
    p2.x = 1.20f;
    p2.y = 2.0f;
    p2.z = 0.5f;

    cloud->points.push_back(p1);
    cloud->points.push_back(p2);

    cloud->width = 2;
    cloud->height = 1;


    pcl::VoxelGrid<pcl::PointXYZRGB> voxel;

    voxel.setInputCloud(
        cloud
    );

    voxel.setLeafSize(
        0.04f,
        0.04f,
        0.04f
    );


    pcl::PointCloud<
        pcl::PointXYZRGB
    > filtered;

    voxel.filter(
        filtered
    );


    EXPECT_EQ(
        filtered.points.size(),
        2u
    );
}