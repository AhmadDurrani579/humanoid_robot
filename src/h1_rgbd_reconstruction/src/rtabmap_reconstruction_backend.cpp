#include "h1_rgbd_reconstruction/rtabmap_reconstruction_backend.hpp"

#include <stdexcept>
#include <iostream>
#include <map>

#include <vector>

#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>

#include <rtabmap/core/Link.h>
#include <rtabmap/core/Signature.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/util3d.h>


namespace
{

// ============================================================
// ROS RGB-D -> RTAB-Map SensorData
// ============================================================

rtabmap::SensorData makeSensorData(
    const sensor_msgs::msg::Image & rgb_msg,
    const sensor_msgs::msg::Image & depth_msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info)
{
    // ========================================================
    // Validate image formats
    // ========================================================

    if (rgb_msg.encoding != "rgb8")
    {
        throw std::runtime_error(
            "RTAB backend expected rgb8 image"
        );
    }


    if (depth_msg.encoding != "32FC1")
    {
        throw std::runtime_error(
            "RTAB backend expected 32FC1 depth image"
        );
    }


    if (
        rgb_msg.width == 0 ||
        rgb_msg.height == 0 ||
        depth_msg.width == 0 ||
        depth_msg.height == 0
    )
    {
        throw std::runtime_error(
            "RTAB backend received empty RGB/depth image"
        );
    }


    // ========================================================
    // ROS RGB -> OpenCV
    // ========================================================

    cv::Mat rgb_view(
        static_cast<int>(
            rgb_msg.height
        ),
        static_cast<int>(
            rgb_msg.width
        ),
        CV_8UC3,
        const_cast<unsigned char *>(
            rgb_msg.data.data()
        ),
        rgb_msg.step
    );


    cv::Mat rgb =
        rgb_view.clone();


    // ROS rgb8 -> OpenCV BGR
    cv::cvtColor(
        rgb,
        rgb,
        cv::COLOR_RGB2BGR
    );   
    
    // ========================================================
    // TEMPORARY COLOR DIAGNOSTIC
    // Check ROS rgb8 -> OpenCV channel interpretation
    // ========================================================

    const int test_x =
        static_cast<int>(rgb_msg.width / 2);

    const int test_y =
        static_cast<int>(rgb_msg.height / 2);

    const cv::Vec3b pixel =
        rgb.at<cv::Vec3b>(
            test_y,
            test_x
        );

    // ========================================================
    // ROS Depth -> OpenCV
    // ========================================================

    cv::Mat depth_small_view(
        static_cast<int>(
            depth_msg.height
        ),
        static_cast<int>(
            depth_msg.width
        ),
        CV_32FC1,
        const_cast<unsigned char *>(
            depth_msg.data.data()
        ),
        depth_msg.step
    );


    const cv::Mat depth_small =
        depth_small_view.clone();


    // ========================================================
    // Match depth dimensions to RGB dimensions
    //
    // RGB   = 640x480
    // Depth = 320x240
    //
    // RTAB-Map SensorData expects the camera model/image
    // dimensions to be consistent.
    //
    // Nearest-neighbour preserves metric depth values.
    // ========================================================

    cv::Mat depth;


    if (
        depth_msg.width != rgb_msg.width ||
        depth_msg.height != rgb_msg.height
    )
    {
        cv::resize(
            depth_small,
            depth,
            cv::Size(
                static_cast<int>(
                    rgb_msg.width
                ),
                static_cast<int>(
                    rgb_msg.height
                )
            ),
            0.0,
            0.0,
            cv::INTER_NEAREST
        );
    }
    else
    {
        depth =
            depth_small.clone();
    }


    
    // ========================================================
    // Scale depth-camera intrinsics to RGB resolution
    // ========================================================

    const double scale_x =
        static_cast<double>(
            rgb_msg.width
        )
        /
        static_cast<double>(
            depth_msg.width
        );


    const double scale_y =
        static_cast<double>(
            rgb_msg.height
        )
        /
        static_cast<double>(
            depth_msg.height
        );


    const double fx =
        depth_camera_info.k[0] *
        scale_x;


    const double fy =
        depth_camera_info.k[4] *
        scale_y;


    const double cx =
        depth_camera_info.k[2] *
        scale_x;


    const double cy =
        depth_camera_info.k[5] *
        scale_y;


    // ========================================================
    // RTAB-Map CameraModel
    //
    // Local transform remains identity.
    //
    // SensorData geometry therefore remains in the optical
    // camera coordinate system.
    // ========================================================

    const rtabmap::CameraModel camera_model(
        fx,
        fy,
        cx,
        cy,
        rtabmap::Transform::getIdentity(),
        0.0,
        cv::Size(
            static_cast<int>(
                rgb_msg.width
            ),
            static_cast<int>(
                rgb_msg.height
            )
        )
    );


    // ========================================================
    // ROS timestamp -> seconds
    // ========================================================

    const double stamp =
        static_cast<double>(
            rgb_msg.header.stamp.sec
        )
        +
        static_cast<double>(
            rgb_msg.header.stamp.nanosec
        )
        *
        1e-9;


    // ========================================================
    // RTAB-Map SensorData
    // ========================================================

    rtabmap::SensorData sensor_data(
        rgb,
        depth,
        camera_model,
        0,
        stamp
    );


    return sensor_data;
}

}  // namespace


// ============================================================
// Constructor
// ============================================================

RtabmapReconstructionBackend::
RtabmapReconstructionBackend()
{
    rtabmap::ParametersMap parameters;


    // --------------------------------------------------------
    // Mapping mode
    // --------------------------------------------------------

    parameters.insert(
        rtabmap::ParametersPair(
            rtabmap::Parameters::
                kMemIncrementalMemory(),
            "true"
        )
    );


    // --------------------------------------------------------
    // Enable RGB-D processing
    // --------------------------------------------------------

    parameters.insert(
        rtabmap::ParametersPair(
            rtabmap::Parameters::
                kRGBDEnabled(),
            "true"
        )
    );


    // --------------------------------------------------------
    // Initialise RTAB-Map in RAM.
    //
    // No database path for now.
    // --------------------------------------------------------

    rtabmap_.init(
        parameters,
        ""
    );
}

int
RtabmapReconstructionBackend::shortTermMemorySize() const
{
    return rtabmap_.getSTMSize();
}

// ============================================================
// Create current single-frame RGB point cloud
// ============================================================

pcl::PointCloud<
    pcl::PointXYZRGB
>::Ptr
RtabmapReconstructionBackend::createCloud(
    const sensor_msgs::msg::Image & rgb_msg,
    const sensor_msgs::msg::Image & depth_msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info
) const
{
    const rtabmap::SensorData sensor_data =
        makeSensorData(
            rgb_msg,
            depth_msg,
            depth_camera_info
        );


    // ========================================================
    // RGB-D -> XYZRGB
    //
    // Decimation = 2 because:
    //
    // original depth = 320x240
    // resized depth  = 640x480
    //
    // decimation 2 restores effective 320x240 density.
    // ========================================================

    auto cloud =
        rtabmap::util3d::cloudRGBFromSensorData(
            sensor_data,
            2,
            20.0f,
            0.20f
        );


    if (!cloud)
    {
        throw std::runtime_error(
            "RTAB-Map returned null RGB cloud"
        );
    }


    cloud->header.frame_id =
        depth_msg.header.frame_id;


    return cloud;
}


// ============================================================
// Process one RGB-D frame through internal RTAB-Map engine
// ============================================================

bool
RtabmapReconstructionBackend::processFrame(
    const sensor_msgs::msg::Image & rgb_msg,
    const sensor_msgs::msg::Image & depth_msg,
    const sensor_msgs::msg::CameraInfo & depth_camera_info,
    const rtabmap::Transform & odom_pose
)
{
    // ========================================================
    // Pose must be valid
    // ========================================================

    if (odom_pose.isNull())
    {
        throw std::runtime_error(
            "RTAB backend received null odometry pose"
        );
    }


    // ========================================================
    // Build SensorData
    // ========================================================

    rtabmap::SensorData sensor_data =
        makeSensorData(
            rgb_msg,
            depth_msg,
            depth_camera_info
        );


    // ========================================================
    // Feed RGB-D + external pose to RTAB-Map
    // ========================================================

    const bool added_to_map =
        rtabmap_.process(
            sensor_data,
            odom_pose
        );


    // ========================================================
    // Inspect optimized RTAB-Map graph
    // ========================================================

    std::map<
        int,
        rtabmap::Transform
    > poses;

    std::multimap<
        int,
        rtabmap::Link
    > constraints;

    std::map<
        int,
        rtabmap::Signature
    > signatures;


    rtabmap_.getGraph(
        poses,
        constraints,

        true,   // optimized poses
        true,   // global graph

        &signatures,

        true,   // withImages
        false,  // withScan
        false,  // withUserData
        false,  // withGrid
        false,  // withWords
        false   // withGlobalDescriptors
    );


    // ========================================================
    // Temporary diagnostics
    // ========================================================

    static std::size_t last_pose_count = 0;
    static std::size_t last_signature_count = 0;


    if (
        poses.size() != last_pose_count ||
        signatures.size() != last_signature_count
    )
    {
        std::cout
            << "RTAB GRAPH | poses="
            << poses.size()
            << " | signatures="
            << signatures.size()
            << " | constraints="
            << constraints.size()
            << std::endl;


        if (!signatures.empty())
            {
                const int latest_id =
                    signatures.rbegin()->first;

                rtabmap::Signature signature =
                    rtabmap_.getSignatureCopy(
                        latest_id,
                        true,   // images
                        false,  // scan
                        false,  // userData
                        false,  // occupancyGrid
                        false,  // withWords
                        false   // withGlobalDescriptors
                    );

                auto & data =
                    signature.sensorData();

                std::cout
                    << "RTAB LATEST NODE | id="
                    << latest_id
                    << " | image_compressed="
                    << data.imageCompressed().total()
                    << " | depth_compressed="
                    << data.depthOrRightCompressed().total()
                    << std::endl;

                data.uncompressData();

                std::cout
                    << "RTAB UNCOMPRESSED | id="
                    << latest_id
                    << " | image="
                    << data.imageRaw().cols
                    << "x"
                    << data.imageRaw().rows
                    << " type="
                    << data.imageRaw().type()
                    << " | depth="
                    << data.depthOrRightRaw().cols
                    << "x"
                    << data.depthOrRightRaw().rows
                    << " type="
                    << data.depthOrRightRaw().type()
                    << std::endl;
            }


        last_pose_count =
            poses.size();

        last_signature_count =
            signatures.size();
    }
    

    return added_to_map;
}


// ============================================================
// Return latest RTAB stored keyframe as a local RGB cloud
//
// IMPORTANT:
// No graph transform.
// No accumulated-map merge.
// No final voxel filter.
//
// This lets us inspect exactly what RTAB stored.
// ============================================================

// ============================================================
// Return latest RTAB stored keyframe transformed into WORLD
//
// This is a diagnostic cloud representing ONE stored RTAB
// keyframe at its stored/global graph pose.
//
// No accumulated merge.
// No final map voxel filter.
// ============================================================

pcl::PointCloud<
    pcl::PointXYZRGB
>::Ptr
RtabmapReconstructionBackend::
getLatestStoredKeyframeCloud() const
{
    auto output =
        std::make_shared<
            pcl::PointCloud<
                pcl::PointXYZRGB
            >
        >();


    // --------------------------------------------------------
    // Get current RTAB graph poses
    // --------------------------------------------------------

    std::map<
        int,
        rtabmap::Transform
    > poses;

    std::multimap<
        int,
        rtabmap::Link
    > constraints;


    rtabmap_.getGraph(
        poses,
        constraints,

        false,  // IMPORTANT: use raw poses, same stable setting
        true,   // global

        nullptr,

        false,  // withImages
        false,  // withScan
        false,  // withUserData
        false,  // withGrid
        false,  // withWords
        false   // withGlobalDescriptors
    );


    if (poses.empty())
    {
        return output;
    }


    // --------------------------------------------------------
    // Latest graph node
    // --------------------------------------------------------

    // --------------------------------------------------------
        // TEST ONLY:
        // Always visualize the FIRST stored RTAB keyframe.
        //
        // If this cloud remains fixed while the robot moves,
        // our world transform is correct.
        //
        // Previously we always selected poses.rbegin(), meaning
        // the newest keyframe changed continuously and therefore
        // appeared to follow the robot.
        // --------------------------------------------------------

        const auto fixed_pose_it =
            poses.begin();

        const int latest_id =
            fixed_pose_it->first;

        const rtabmap::Transform &
            latest_pose =
                fixed_pose_it->second;

    if (latest_pose.isNull())
    {
        return output;
    }


    // --------------------------------------------------------
    // Retrieve this stored RTAB keyframe
    // --------------------------------------------------------

    rtabmap::Signature signature =
        rtabmap_.getSignatureCopy(
            latest_id,

            true,   // images
            false,  // scan
            false,  // userData
            false,  // occupancyGrid
            false,  // words
            false   // global descriptors
        );


    rtabmap::SensorData &
        data =
            signature.sensorData();


    // --------------------------------------------------------
    // Check stored RGB-D
    // --------------------------------------------------------

    const bool has_rgb =
        !data.imageRaw().empty() ||
        !data.imageCompressed().empty();


    const bool has_depth =
        !data.depthOrRightRaw().empty() ||
        !data.depthOrRightCompressed().empty();


    if (
        !has_rgb ||
        !has_depth
    )
    {
        return output;
    }


    // --------------------------------------------------------
    // Decompress stored RGB-D
    // --------------------------------------------------------

    data.uncompressData();


    if (
        data.imageRaw().empty() ||
        data.depthOrRightRaw().empty()
    )
    {
        return output;
    }


    // --------------------------------------------------------
    // Stored RGB-D -> local keyframe cloud
    // --------------------------------------------------------

    auto local_cloud =
        rtabmap::util3d::
            cloudRGBFromSensorData(
                data,
                2,
                20.0f,
                0.20f
            );


    if (
        !local_cloud ||
        local_cloud->empty()
    )
    {
        return output;
    }


    // --------------------------------------------------------
    // Remove NaNs
    // --------------------------------------------------------

    std::vector<int>
        valid_indices;


    auto clean_cloud =
        std::make_shared<
            pcl::PointCloud<
                pcl::PointXYZRGB
            >
        >();


    pcl::removeNaNFromPointCloud(
        *local_cloud,
        *clean_cloud,
        valid_indices
    );


    if (clean_cloud->empty())
    {
        return output;
    }


    // --------------------------------------------------------
    // Transform THIS stored keyframe into RTAB global/world
    // coordinates using its graph pose.
    // --------------------------------------------------------

    auto world_cloud =
        rtabmap::util3d::
            transformPointCloud(
                clean_cloud,
                latest_pose
            );


    if (
        !world_cloud ||
        world_cloud->empty()
    )
    {
        return output;
    }


    world_cloud->width =
        static_cast<std::uint32_t>(
            world_cloud->points.size()
        );

    world_cloud->height = 1;

    world_cloud->is_dense = true;


    std::cout
        << "RTAB STORED KEYFRAME WORLD | "
        << "id="
        << latest_id
        << " | points="
        << world_cloud->size()
        << " | pose_xyz=["
        << latest_pose.x()
        << " "
        << latest_pose.y()
        << " "
        << latest_pose.z()
        << "]"
        << std::endl;


    return world_cloud;
}

// ============================================================
// Build accumulated coloured reconstruction from
// RTAB-Map stored keyframes + optimized graph poses
// ============================================================

pcl::PointCloud<
    pcl::PointXYZRGB
>::Ptr
RtabmapReconstructionBackend::getReconstructedCloud() const
{
    // --------------------------------------------------------
    // Get RTAB-Map optimized graph
    // --------------------------------------------------------

    std::map<
        int,
        rtabmap::Transform
    > poses;

    std::multimap<
        int,
        rtabmap::Link
    > constraints;


    rtabmap_.getGraph(
        poses,
        constraints,

        false,   // optimized
        true,   // global

        nullptr,

        false,  // withImages
        false,  // withScan
        false,  // withUserData
        false,  // withGrid
        false,  // withWords
        false   // withGlobalDescriptors
    );


    // --------------------------------------------------------
    // Output accumulated cloud
    // --------------------------------------------------------

    auto accumulated =
        std::make_shared<
            pcl::PointCloud<
                pcl::PointXYZRGB
            >
        >();


    if (poses.empty())
    {
        return accumulated;
    }


    std::size_t used_nodes = 0;

    std::size_t skipped_nodes = 0;

    std::size_t total_valid_points = 0;


    // --------------------------------------------------------
    // Process every optimized RTAB-Map node
    // --------------------------------------------------------

    for (const auto & pose_item : poses)
    {
        const int node_id =
            pose_item.first;

        const rtabmap::Transform &
            optimized_pose =
                pose_item.second;


        // ----------------------------------------------------
        // Ignore invalid graph poses
        // ----------------------------------------------------

        if (optimized_pose.isNull())
        {
            ++skipped_nodes;
            continue;
        }


        // ----------------------------------------------------
        // Retrieve this RTAB keyframe including images
        // ----------------------------------------------------

        rtabmap::Signature signature =
            rtabmap_.getSignatureCopy(
                node_id,

                true,   // images
                false,  // scan
                false,  // userData
                false,  // occupancyGrid
                false,  // words
                false   // global descriptors
            );


        rtabmap::SensorData &
            data =
                signature.sensorData();


        // ----------------------------------------------------
        // Ensure RGB-D data actually exists
        // ----------------------------------------------------

        const bool has_rgb =
            !data.imageRaw().empty() ||
            !data.imageCompressed().empty();


        const bool has_depth =
            !data.depthOrRightRaw().empty() ||
            !data.depthOrRightCompressed().empty();


        if (
            !has_rgb ||
            !has_depth
        )
        {
            ++skipped_nodes;
            continue;
        }


        // ----------------------------------------------------
        // RTAB-Map native decompression
        // ----------------------------------------------------

        data.uncompressData();


        if (
            data.imageRaw().empty() ||
            data.depthOrRightRaw().empty()
        )
        {
            ++skipped_nodes;
            continue;
        }


        // ----------------------------------------------------
        // Generate this keyframe's local XYZRGB cloud
        //
        // Stored RGB-D is 640x480.
        //
        // decimation=2 gives effective 320x240 density,
        // which we already validated against our native
        // depth resolution.
        // ----------------------------------------------------

        auto local_cloud =
            rtabmap::util3d::
                cloudRGBFromSensorData(
                    data,
                    2,      // decimation
                    20.0f,  // max depth
                    0.20f   // min depth
                );


        if (
            !local_cloud ||
            local_cloud->empty()
        )
        {
            ++skipped_nodes;
            continue;
        }


        // ----------------------------------------------------
        // Remove invalid / NaN points before accumulation
        // ----------------------------------------------------

        std::vector<int>
            valid_indices;


        auto clean_cloud =
            std::make_shared<
                pcl::PointCloud<
                    pcl::PointXYZRGB
                >
            >();


        pcl::removeNaNFromPointCloud(
            *local_cloud,
            *clean_cloud,
            valid_indices
        );


        if (clean_cloud->empty())
        {
            ++skipped_nodes;
            continue;
        }


        total_valid_points +=
            clean_cloud->size();


        // ----------------------------------------------------
        // Transform local keyframe cloud using the latest
        // OPTIMIZED RTAB-Map graph pose
        // ----------------------------------------------------

        auto transformed_cloud =
            rtabmap::util3d::
                transformPointCloud(
                    clean_cloud,
                    optimized_pose
                );


        if (
            !transformed_cloud ||
            transformed_cloud->empty()
        )
        {
            ++skipped_nodes;
            continue;
        }


        // ----------------------------------------------------
        // Merge this corrected keyframe into reconstruction
        // ----------------------------------------------------

        *accumulated +=
            *transformed_cloud;


        ++used_nodes;
    }


    // --------------------------------------------------------
    // Nothing usable
    // --------------------------------------------------------

    if (accumulated->empty())
    {
        std::cout
            << "RTAB RECONSTRUCTION | "
            << "nodes="
            << poses.size()
            << " | used=0"
            << " | skipped="
            << skipped_nodes
            << " | points=0"
            << std::endl;

        return accumulated;
    }


    // --------------------------------------------------------
    // Final voxel filtering
    //
    // IMPORTANT:
    //
    // This is NOT the old IncrementalVoxelMap.
    //
    // We first reconstruct from RTAB-Map's current optimized
    // keyframe poses, then voxel-filter the resulting output.
    //
    // If RTAB-Map changes a pose later, this whole cloud can
    // be rebuilt correctly.
    // --------------------------------------------------------

    pcl::VoxelGrid<
        pcl::PointXYZRGB
    > voxel_filter;


    voxel_filter.setInputCloud(
        accumulated
    );


    voxel_filter.setLeafSize(
        0.04f,
        0.04f,
        0.04f
    );


    auto filtered =
        std::make_shared<
            pcl::PointCloud<
                pcl::PointXYZRGB
            >
        >();


    voxel_filter.filter(
        *filtered
    );


    filtered->width =
        static_cast<std::uint32_t>(
            filtered->points.size()
        );

    filtered->height = 1;

    filtered->is_dense = true;


    std::cout
        << "RTAB RECONSTRUCTION | "
        << "graph_nodes="
        << poses.size()
        << " | used="
        << used_nodes
        << " | skipped="
        << skipped_nodes
        << " | raw_valid_points="
        << total_valid_points
        << " | merged="
        << accumulated->size()
        << " | filtered="
        << filtered->size()
        << std::endl;


    return filtered;
}

// ============================================================
// Diagnostic: number of nodes in RTAB-Map working memory
// ============================================================

int
RtabmapReconstructionBackend::workingMemorySize() const
{
    return rtabmap_.getWMSize();
}