#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Transform.h>


class RtabmapReconstructionBackend
{
public:

    RtabmapReconstructionBackend();


    // --------------------------------------------------------
    // Current single-frame RGB-D cloud
    // --------------------------------------------------------

    pcl::PointCloud<
        pcl::PointXYZRGB
    >::Ptr createCloud(
        const sensor_msgs::msg::Image & rgb_msg,
        const sensor_msgs::msg::Image & depth_msg,
        const sensor_msgs::msg::CameraInfo & depth_camera_info
    ) const;


    // --------------------------------------------------------
    // Feed RGB-D + existing robot pose into RTAB-Map
    // --------------------------------------------------------

    bool processFrame(
        const sensor_msgs::msg::Image & rgb_msg,
        const sensor_msgs::msg::Image & depth_msg,
        const sensor_msgs::msg::CameraInfo & depth_camera_info,
        const rtabmap::Transform & odom_pose
    );


    // --------------------------------------------------------
    // Build accumulated coloured reconstruction
    // from RTAB-Map stored signatures + optimized poses
    // --------------------------------------------------------

    pcl::PointCloud<
        pcl::PointXYZRGB
    >::Ptr getReconstructedCloud() const;

    pcl::PointCloud<
        pcl::PointXYZRGB
    >::Ptr getLatestStoredKeyframeCloud() const;

    // --------------------------------------------------------
    // Diagnostics
    // --------------------------------------------------------

    int workingMemorySize() const;

    int shortTermMemorySize() const;


private:

    // RTAB-Map Core lives entirely inside OUR node.
    rtabmap::Rtabmap rtabmap_;
};