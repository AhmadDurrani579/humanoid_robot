#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>

#include <Eigen/Geometry>

#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl_conversions/pcl_conversions.h>
#include <rtabmap/core/Transform.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "geometry_msgs/msg/transform_stamped.hpp"

#include "message_filters/subscriber.hpp"
#include "message_filters/synchronizer.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"

#include "rclcpp/rclcpp.hpp"

#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "h1_rgbd_reconstruction/incremental_voxel_map.hpp"
#include "h1_rgbd_reconstruction/rtabmap_reconstruction_backend.hpp"


class RgbdReconstructionNode : public rclcpp::Node
{
public:

    using Image = sensor_msgs::msg::Image;

    using SyncPolicy =
        message_filters::sync_policies::ApproximateTime<
            Image,
            Image
        >;


    RgbdReconstructionNode()
        : Node("rgbd_reconstruction_node")
    {
        RCLCPP_INFO(
            this->get_logger(),
            "RGB-D reconstruction node starting..."
        );


        // ----------------------------------------------------
        // RGB subscriber
        // ----------------------------------------------------

        rgb_sub_.subscribe(
            this,
            "/camera/color/image_raw"
        );


        // ----------------------------------------------------
        // Depth subscriber
        // ----------------------------------------------------

        depth_sub_.subscribe(
            this,
            "/camera/depth/image_nav"
        );


        // ----------------------------------------------------
        // RGB + Depth synchronizer
        // ----------------------------------------------------

        sync_ = std::make_shared<
            message_filters::Synchronizer<SyncPolicy>
        >(
            SyncPolicy(10),
            rgb_sub_,
            depth_sub_
        );

        sync_->registerCallback(
            std::bind(
                &RgbdReconstructionNode::rgbd_callback,
                this,
                std::placeholders::_1,
                std::placeholders::_2
            )
        );


        // ----------------------------------------------------
        // CameraInfo
        // ----------------------------------------------------

        camera_info_sub_ =
            this->create_subscription<
                sensor_msgs::msg::CameraInfo
            >(
                "/camera/depth/camera_info_nav",
                10,
                std::bind(
                    &RgbdReconstructionNode::camera_info_callback,
                    this,
                    std::placeholders::_1
                )
            );


        // ----------------------------------------------------
        // Current camera-frame cloud publisher
        // ----------------------------------------------------

        cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/cloud",
                10
            );

        rtab_map_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/map_rtab",
                2
            );
        
        stored_keyframe_cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/stored_keyframe_cloud",
                2
            );

        // ----------------------------------------------------
        // Current map-frame cloud publisher
        // ----------------------------------------------------

        map_cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/cloud_map",
                10
            );


        // ----------------------------------------------------
        // Persistent accumulated reconstruction publisher
        // ----------------------------------------------------

        accumulated_cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/map_accumulated",
                10
            );

        rtab_cloud_pub_ =
            this->create_publisher<
                sensor_msgs::msg::PointCloud2
            >(
                "/rgbd_reconstruction/cloud_rtab",
                10
            );

        // ----------------------------------------------------
        // Incremental persistent voxel map
        // ----------------------------------------------------

        voxel_map_ =
            std::make_unique<
                IncrementalVoxelMap
            >(
                0.04f
            );

        rtabmap_backend_ =
            std::make_unique<
                RtabmapReconstructionBackend
            >();

        // ----------------------------------------------------
        // TF
        // ----------------------------------------------------

        tf_buffer_ =
            std::make_shared<
                tf2_ros::Buffer
            >(
                this->get_clock()
            );

        tf_listener_ =
            std::make_shared<
                tf2_ros::TransformListener
            >(
                *tf_buffer_
            );


        RCLCPP_INFO(
            this->get_logger(),
            "Waiting for RGB, depth and CameraInfo..."
        );

    }


private:

    // ========================================================
    // CameraInfo
    // ========================================================

    void camera_info_callback(
        const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {

        depth_camera_info_ = *msg;

        fx_ = msg->k[0];
        fy_ = msg->k[4];
        cx_ = msg->k[2];
        cy_ = msg->k[5];

        if (!camera_info_received_)
        {
            RCLCPP_INFO(
                this->get_logger(),
                "CameraInfo received | "
                "size=%ux%u | "
                "fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                msg->width,
                msg->height,
                fx_,
                fy_,
                cx_,
                cy_
            );

            camera_info_received_ = true;
        }
    }


    // ========================================================
    // RGB-D callback
    // ========================================================


    void rgbd_callback(
        const Image::ConstSharedPtr & rgb_msg,
        const Image::ConstSharedPtr & depth_msg)
    {
        // ========================================================
        // Total callback timer
        // ========================================================

        const auto callback_start =
            std::chrono::steady_clock::now();


        // ========================================================
        // RGB / Depth timestamps
        // ========================================================

        const double rgb_stamp =
            rclcpp::Time(
                rgb_msg->header.stamp
            ).seconds();

        const double depth_stamp =
            rclcpp::Time(
                depth_msg->header.stamp
            ).seconds();

        const double difference =
            std::abs(
                rgb_stamp - depth_stamp
            );


        // ========================================================
        // CameraInfo check
        // ========================================================

        if (!camera_info_received_)
        {
            return;
        }


        // ========================================================
        // Encoding checks
        // ========================================================

        if (depth_msg->encoding != "32FC1")
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Expected depth encoding 32FC1, received %s",
                depth_msg->encoding.c_str()
            );

            return;
        }


        if (rgb_msg->encoding != "rgb8")
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Expected RGB encoding rgb8, received %s",
                rgb_msg->encoding.c_str()
            );

            return;
        }


        // ========================================================
        // Raw depth diagnostics
        // ========================================================

        float min_depth_seen =
            std::numeric_limits<float>::max();

        float max_depth_seen = 0.0f;

        std::size_t valid_depth_count = 0;
        std::size_t over_20m_count = 0;


        // ========================================================
        // Camera cloud generation timer
        // ========================================================

        const auto cloud_generation_start =
            std::chrono::steady_clock::now();


        // ========================================================
        // Create current camera-frame XYZRGB cloud
        // ========================================================

        auto cloud =
            std::make_shared<
                pcl::PointCloud<pcl::PointXYZRGB>
            >();

        cloud->header.frame_id =
            depth_msg->header.frame_id;

        cloud->is_dense = false;

        cloud->points.reserve(
            depth_msg->width *
            depth_msg->height
        );


        // ========================================================
        // RGB / Depth scaling
        // ========================================================

        const double scale_x =
            static_cast<double>(
                rgb_msg->width
            ) /
            static_cast<double>(
                depth_msg->width
            );

        const double scale_y =
            static_cast<double>(
                rgb_msg->height
            ) /
            static_cast<double>(
                depth_msg->height
            );


        constexpr float MIN_DEPTH = 0.20f;
        constexpr float MAX_DEPTH = 20.0f;


        // ========================================================
        // Depth -> XYZRGB
        // ========================================================

        for (
            uint32_t v = 0;
            v < depth_msg->height;
            ++v
        )
        {
            const auto * depth_row =
                reinterpret_cast<const float *>(
                    &depth_msg->data[
                        v * depth_msg->step
                    ]
                );


            for (
                uint32_t u = 0;
                u < depth_msg->width;
                ++u
            )
            {
                const float z =
                    depth_row[u];


                // ------------------------------------------------
                // Raw depth diagnostics
                // ------------------------------------------------

                if (
                    std::isfinite(z) &&
                    z > 0.0f
                )
                {
                    min_depth_seen =
                        std::min(
                            min_depth_seen,
                            z
                        );

                    max_depth_seen =
                        std::max(
                            max_depth_seen,
                            z
                        );

                    ++valid_depth_count;

                    if (z > MAX_DEPTH)
                    {
                        ++over_20m_count;
                    }
                }


                // ------------------------------------------------
                // Reject invalid / far depth
                // ------------------------------------------------

                if (
                    !std::isfinite(z) ||
                    z < MIN_DEPTH ||
                    z > MAX_DEPTH
                )
                {
                    continue;
                }


                // ------------------------------------------------
                // Depth pixel -> XYZ
                // ------------------------------------------------

                const float x =
                    static_cast<float>(
                        (
                            static_cast<double>(u)
                            - cx_
                        )
                        * z
                        / fx_
                    );

                const float y =
                    static_cast<float>(
                        (
                            static_cast<double>(v)
                            - cy_
                        )
                        * z
                        / fy_
                    );


                // ------------------------------------------------
                // Depth pixel -> corresponding RGB pixel
                // ------------------------------------------------

                const uint32_t rgb_u =
                    static_cast<uint32_t>(
                        static_cast<double>(u)
                        * scale_x
                    );

                const uint32_t rgb_v =
                    static_cast<uint32_t>(
                        static_cast<double>(v)
                        * scale_y
                    );


                if (
                    rgb_u >= rgb_msg->width ||
                    rgb_v >= rgb_msg->height
                )
                {
                    continue;
                }


                const size_t rgb_index =
                    static_cast<size_t>(
                        rgb_v
                    )
                    * rgb_msg->step
                    +
                    static_cast<size_t>(
                        rgb_u
                    )
                    * 3;


                // ------------------------------------------------
                // Create point
                // ------------------------------------------------

                pcl::PointXYZRGB point;

                point.x = x;
                point.y = y;
                point.z = z;

                point.r =
                    rgb_msg->data[
                        rgb_index + 0
                    ];

                point.g =
                    rgb_msg->data[
                        rgb_index + 1
                    ];

                point.b =
                    rgb_msg->data[
                        rgb_index + 2
                    ];


                cloud->points.push_back(
                    point
                );
            }
        }


        cloud->width =
            static_cast<uint32_t>(
                cloud->points.size()
            );

        cloud->height = 1;


        const auto cloud_generation_end =
            std::chrono::steady_clock::now();


        // ========================================================
        // Publish current camera-frame cloud
        // ========================================================

        sensor_msgs::msg::PointCloud2 cloud_msg;

        pcl::toROSMsg(
            *cloud,
            cloud_msg
        );

        cloud_msg.header =
            depth_msg->header;


        cloud_pub_->publish(
            cloud_msg
        );


        // ========================================================
        // RTAB-Map Core single-frame reconstruction test
        // ========================================================

        try
        {
            auto rtab_cloud =
                rtabmap_backend_->createCloud(
                    *rgb_msg,
                    *depth_msg,
                    depth_camera_info_
                );


            sensor_msgs::msg::PointCloud2
                rtab_cloud_msg;


            pcl::toROSMsg(
                *rtab_cloud,
                rtab_cloud_msg
            );


            rtab_cloud_msg.header.stamp =
                depth_msg->header.stamp;

            rtab_cloud_msg.header.frame_id =
                depth_msg->header.frame_id;


            rtab_cloud_pub_->publish(
                rtab_cloud_msg
            );

            std::size_t rtab_valid_points = 0;
            for (const auto & point : rtab_cloud->points)
            {
                if (
                    std::isfinite(point.x) &&
                    std::isfinite(point.y) &&
                    std::isfinite(point.z)
                )
                {
                    ++rtab_valid_points;
                }
            }

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "RTAB CLOUD | "
                "total=%zu | "
                "valid=%zu | "
                "frame=%s",
                rtab_cloud->points.size(),
                rtab_valid_points,
                depth_msg->header.frame_id.c_str()
            );

        }
        catch (const std::exception & ex)
        {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "RTAB cloud generation failed: %s",
                ex.what()
            );
        }

        // ========================================================
        // Camera cloud -> map
        // ========================================================

        try
        {
            // ----------------------------------------------------
            // TF
            // ----------------------------------------------------

            const auto tf_start =
                std::chrono::steady_clock::now();


            const tf2::TimePoint query_time =
                tf2::TimePoint(
                    std::chrono::nanoseconds(
                        rclcpp::Time(
                            depth_msg->header.stamp
                        ).nanoseconds()
                    )
                );


            const auto transform =
                tf_buffer_->lookupTransform(
                    "map",
                    depth_msg->header.frame_id,
                    query_time,
                    tf2::durationFromSec(0.15)
                );
            
            const auto rtab_odom_transform =
                tf_buffer_->lookupTransform(
                    "world",
                    depth_msg->header.frame_id,
                    query_time,
                    tf2::durationFromSec(0.15)
                );
            
                // ----------------------------------------------------
                // NEW: convert ROS transform -> rtabmap::Transform
                // ----------------------------------------------------

            const auto & rt =
                    rtab_odom_transform.transform.translation;

            const auto & rq =
                    rtab_odom_transform.transform.rotation;


            rtabmap::Transform rtab_odom_pose(
                    static_cast<float>(rt.x),
                    static_cast<float>(rt.y),
                    static_cast<float>(rt.z),

                    static_cast<float>(rq.x),
                    static_cast<float>(rq.y),
                    static_cast<float>(rq.z),
                    static_cast<float>(rq.w)
                );


                // ----------------------------------------------------
                // NEW: feed RGB-D + external pose into RTAB-Map
                // ----------------------------------------------------

                // ============================================================
                // Feed RGB-D + external world-frame pose into RTAB-Map
                // ============================================================

                const bool rtab_added =
                    rtabmap_backend_->processFrame(
                        *rgb_msg,
                        *depth_msg,
                        depth_camera_info_,
                        rtab_odom_pose
                    );

                // ============================================================
                // Diagnostic:
                // publish latest RGB-D keyframe actually stored by RTAB-Map
                //
                // No accumulated-map merge.
                // No graph transform.
                // No final map voxel filter.
                // ============================================================

                if (rtab_added)
                {
                    auto stored_keyframe_cloud =
                        rtabmap_backend_->
                            getLatestStoredKeyframeCloud();


                    if (
                        stored_keyframe_cloud &&
                        !stored_keyframe_cloud->empty()
                    )
                    {
                        sensor_msgs::msg::PointCloud2
                            stored_keyframe_msg;


                        pcl::toROSMsg(
                            *stored_keyframe_cloud,
                            stored_keyframe_msg
                        );


                        stored_keyframe_msg.header.stamp =
                            depth_msg->header.stamp;

                        stored_keyframe_msg.header.frame_id =
                            "world";


                        stored_keyframe_cloud_pub_->publish(
                            stored_keyframe_msg
                        );


                        RCLCPP_INFO_THROTTLE(
                            this->get_logger(),
                            *this->get_clock(),
                            2000,
                            "RTAB STORED CLOUD | points=%zu",
                            stored_keyframe_cloud->points.size()
                        );
                    }
                }


                RCLCPP_INFO_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "RTAB PROCESS | "
                    "added=%s | "
                    "STM=%d | "
                    "WM=%d | "
                    "odom_xyz=[%.3f %.3f %.3f]",
                    rtab_added
                        ? "YES"
                        : "NO",
                    rtabmap_backend_->shortTermMemorySize(),
                    rtabmap_backend_->workingMemorySize(),
                    rt.x,
                    rt.y,
                    rt.z
                );


                // ============================================================
                // Publish RTAB-Map accumulated reconstruction
                //
                // IMPORTANT:
                //
                // getReconstructedCloud() rebuilds the reconstruction from
                // RTAB-Map's stored RGB-D keyframes + optimized graph poses.
                //
                // Do NOT run it at camera rate.
                // For this A/B test rebuild only every 2 seconds.
                // ============================================================

                static auto last_rtab_map_publish_time =
                    std::chrono::steady_clock::now();


                const auto rtab_map_now =
                    std::chrono::steady_clock::now();


                const double since_last_rtab_map_publish =
                    std::chrono::duration<double>(
                        rtab_map_now -
                        last_rtab_map_publish_time
                    ).count();


                if (
                    rtab_added &&
                    since_last_rtab_map_publish >= 2.0
                )
                {
                    const auto rtab_reconstruction_start =
                        std::chrono::steady_clock::now();


                    auto rtab_reconstructed_cloud =
                        rtabmap_backend_->
                            getReconstructedCloud();


                    const auto rtab_reconstruction_end =
                        std::chrono::steady_clock::now();


                    const double rtab_reconstruction_ms =
                        std::chrono::duration<
                            double,
                            std::milli
                        >(
                            rtab_reconstruction_end -
                            rtab_reconstruction_start
                        ).count();


                    if (
                        rtab_reconstructed_cloud &&
                        !rtab_reconstructed_cloud->empty()
                    )
                    {
                        // ----------------------------------------------------
                        // PCL -> ROS PointCloud2
                        // ----------------------------------------------------

                        sensor_msgs::msg::PointCloud2
                            rtab_map_msg;


                        pcl::toROSMsg(
                            *rtab_reconstructed_cloud,
                            rtab_map_msg
                        );


                        // ----------------------------------------------------
                        // The optimized RTAB graph is based on the
                        // world-frame pose supplied to processFrame().
                        // ----------------------------------------------------

                        rtab_map_msg.header.stamp =
                            depth_msg->header.stamp;

                        rtab_map_msg.header.frame_id =
                            "world";


                        // ----------------------------------------------------
                        // Publish RTAB reconstruction on its OWN topic.
                        //
                        // Do not replace map_accumulated yet.
                        // ----------------------------------------------------

                        rtab_map_pub_->publish(
                            rtab_map_msg
                        );


                        RCLCPP_INFO(
                            this->get_logger(),
                            "RTAB MAP | "
                            "points=%zu | "
                            "build=%.2f ms | "
                            "STM=%d | "
                            "WM=%d",
                            rtab_reconstructed_cloud->points.size(),
                            rtab_reconstruction_ms,
                            rtabmap_backend_->shortTermMemorySize(),
                            rtabmap_backend_->workingMemorySize()
                        );
                    }
                    else
                    {
                        RCLCPP_WARN(
                            this->get_logger(),
                            "RTAB MAP | reconstruction is empty"
                        );
                    }


                    last_rtab_map_publish_time =
                        rtab_map_now;
                }

            const auto & t =
                transform.transform.translation;

            const auto & q =
                transform.transform.rotation;


            Eigen::Quaternionf rotation(
                static_cast<float>(q.w),
                static_cast<float>(q.x),
                static_cast<float>(q.y),
                static_cast<float>(q.z)
            );


            Eigen::Affine3f map_from_camera =
                Eigen::Affine3f::Identity();


            map_from_camera.translation() <<
                static_cast<float>(t.x),
                static_cast<float>(t.y),
                static_cast<float>(t.z);


            map_from_camera.linear() =
                rotation
                    .normalized()
                    .toRotationMatrix();


            pcl::PointCloud<
                pcl::PointXYZRGB
            > cloud_map;


            pcl::transformPointCloud(
                *cloud,
                cloud_map,
                map_from_camera
            );


            const auto tf_end =
                std::chrono::steady_clock::now();


            // ====================================================
            // Publish CURRENT map-frame cloud
            // ====================================================

            sensor_msgs::msg::PointCloud2
                cloud_map_msg;


            pcl::toROSMsg(
                cloud_map,
                cloud_map_msg
            );


            cloud_map_msg.header.stamp =
                depth_msg->header.stamp;

            cloud_map_msg.header.frame_id =
                "map";


            map_cloud_pub_->publish(
                cloud_map_msg
            );


            // ====================================================
            // Keyframe selection
            // ====================================================

            const Eigen::Vector3f current_translation =
                map_from_camera.translation();

            const Eigen::Quaternionf current_rotation =
                rotation.normalized();


            double translation_delta = 0.0;
            double rotation_delta_deg = 0.0;

            bool accept_keyframe = false;


            // ----------------------------------------------------
            // First valid frame is always a keyframe
            // ----------------------------------------------------

            if (!have_last_keyframe_pose_)
            {
                accept_keyframe = true;
            }
            else
            {
                // ------------------------------------------------
                // Translation difference from last accepted frame
                // ------------------------------------------------

                translation_delta =
                    static_cast<double>(
                        (
                            current_translation -
                            last_keyframe_translation_
                        ).norm()
                    );


                // ------------------------------------------------
                // Rotation difference from last accepted frame
                // ------------------------------------------------

                const float quaternion_dot =
                    std::clamp(
                        std::abs(
                            current_rotation.dot(
                                last_keyframe_rotation_
                            )
                        ),
                        0.0f,
                        1.0f
                    );


                const double rotation_delta_rad =
                    2.0 *
                    std::acos(
                        static_cast<double>(
                            quaternion_dot
                        )
                    );


                rotation_delta_deg =
                    rotation_delta_rad *
                    180.0 /
                    M_PI;


                // ------------------------------------------------
                // Accept when either threshold is exceeded
                // ------------------------------------------------

                if (
                    translation_delta >=
                        keyframe_translation_threshold_
                    ||
                    rotation_delta_deg >=
                        keyframe_rotation_threshold_deg_
                )
                {
                    accept_keyframe = true;
                }
            }


            // ====================================================
            // Integrate ONLY accepted keyframes
            // ====================================================

            const auto integration_start =
                std::chrono::steady_clock::now();


            if (accept_keyframe)
            {
                voxel_map_->beginFrame();


                for (
                    const auto & point :
                    cloud_map.points
                )
                {
                    if (
                        !std::isfinite(point.x) ||
                        !std::isfinite(point.y) ||
                        !std::isfinite(point.z)
                    )
                    {
                        continue;
                    }


                    voxel_map_->integrate(
                        point.x,
                        point.y,
                        point.z,
                        point.r,
                        point.g,
                        point.b
                    );
                }


                voxel_map_->endFrame();


                last_keyframe_translation_ =
                    current_translation;

                last_keyframe_rotation_ =
                    current_rotation;

                have_last_keyframe_pose_ =
                    true;

                ++accepted_keyframes_;
            }
            else
            {
                ++rejected_frames_;
            }


            const auto integration_end =
                std::chrono::steady_clock::now();    

            // ====================================================
            // Full accumulated map publication
            //
            // Integrate every frame,
            // but export + publish only every 1 second.
            // ====================================================

            const auto now =
                std::chrono::steady_clock::now();


            const double since_last_publish =
                std::chrono::duration<double>(
                    now -
                    last_accumulated_publish_time_
                ).count();


            double export_ms = 0.0;
            double conversion_ms = 0.0;
            double publish_ms = 0.0;

            bool accumulated_map_published = false;


            if (
                since_last_publish >=
                accumulated_publish_period_sec_
            )
            {
                // -----------------------------------------------
                // Voxel map -> PCL cloud
                // -----------------------------------------------

                const auto export_start =
                    std::chrono::steady_clock::now();


                const auto accumulated_cloud =
                    voxel_map_->toFilteredPointCloud(2);


                const auto export_end =
                    std::chrono::steady_clock::now();


                // -----------------------------------------------
                // PCL -> ROS PointCloud2
                // -----------------------------------------------

                sensor_msgs::msg::PointCloud2
                    accumulated_msg;


                const auto conversion_start =
                    std::chrono::steady_clock::now();


                pcl::toROSMsg(
                    accumulated_cloud,
                    accumulated_msg
                );


                const auto conversion_end =
                    std::chrono::steady_clock::now();


                accumulated_msg.header.stamp =
                    depth_msg->header.stamp;

                accumulated_msg.header.frame_id =
                    "map";


                // -----------------------------------------------
                // Publish complete persistent map
                // -----------------------------------------------

                const auto publish_start =
                    std::chrono::steady_clock::now();


                accumulated_cloud_pub_->publish(
                    accumulated_msg
                );


                const auto publish_end =
                    std::chrono::steady_clock::now();


                // -----------------------------------------------
                // Timing
                // -----------------------------------------------

                export_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        export_end -
                        export_start
                    ).count();


                conversion_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        conversion_end -
                        conversion_start
                    ).count();


                publish_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        publish_end -
                        publish_start
                    ).count();


                last_accumulated_publish_time_ =
                    now;

                accumulated_map_published =
                    true;
            }


            // ====================================================
            // Profile calculations
            // ====================================================

            const double cloud_generation_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    cloud_generation_end -
                    cloud_generation_start
                ).count();


            const double tf_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    tf_end -
                    tf_start
                ).count();


            const double integration_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    integration_end -
                    integration_start
                ).count();


            // ====================================================
            // Profiling
            // ====================================================

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "PROFILE | "
                "cloud=%.2f ms | "
                "tf=%.2f ms | "
                "integrate=%.2f ms | "
                "export=%.2f ms | "
                "convert=%.2f ms | "
                "publish=%.2f ms | "
                "full_map_pub=%s | "
                "voxels=%zu",
                cloud_generation_ms,
                tf_ms,
                integration_ms,
                export_ms,
                conversion_ms,
                publish_ms,
                accumulated_map_published
                    ? "YES"
                    : "NO",
                voxel_map_->size()
            );


            // ====================================================
            // Hash diagnostics
            // ====================================================

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "HASH STATS | "
                "voxels=%zu | "
                "buckets=%zu | "
                "load=%.3f | "
                "max_bucket=%zu",
                voxel_map_->size(),
                voxel_map_->bucketCount(),
                voxel_map_->loadFactor(),
                voxel_map_->maxBucketSize()
            );  

            const auto voxel_stats =
                voxel_map_->getObservationStats();

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "VOXEL FRAME OBS | "
                "voxels=%zu | "
                "point_obs=%zu | "
                "frame=1:%zu | "
                "frame=2-5:%zu | "
                "frame=6-20:%zu | "
                "frame>20:%zu",
                voxel_stats.total_voxels,
                voxel_stats.total_point_observations,
                voxel_stats.frame_count_1,
                voxel_stats.frame_count_2_to_5,
                voxel_stats.frame_count_6_to_20,
                voxel_stats.frame_count_over_20
            );


            // ====================================================
            // Depth diagnostics
            // ====================================================

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "DEPTH RANGE | "
                "min=%.3f m | "
                "max=%.3f m | "
                "valid=%zu | "
                "over_20m=%zu",
                min_depth_seen,
                max_depth_seen,
                valid_depth_count,
                over_20m_count
            );

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "KEYFRAME | "
                "accepted=%s | "
                "translation_delta=%.4f m | "
                "rotation_delta=%.3f deg | "
                "accepted_total=%zu | "
                "rejected_total=%zu",
                accept_keyframe
                    ? "YES"
                    : "NO",
                translation_delta,
                rotation_delta_deg,
                accepted_keyframes_,
                rejected_frames_
            );
        }
        catch (
            const tf2::TransformException & ex
        )
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Could not transform cloud %s -> map: %s",
                depth_msg->header.frame_id.c_str(),
                ex.what()
            );
        }


        // ========================================================
        // Total callback processing time
        // ========================================================

        const auto callback_end =
            std::chrono::steady_clock::now();


        const double callback_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                callback_end -
                callback_start
            ).count();


        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Cloud callback | "
            "points=%zu | "
            "processing=%.2f ms",
            cloud->points.size(),
            callback_ms
        );


        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "RGB-D pair received | "
            "RGB=%ux%u (%s) | "
            "Depth=%ux%u (%s) | "
            "dt=%.4f sec",
            rgb_msg->width,
            rgb_msg->height,
            rgb_msg->encoding.c_str(),
            depth_msg->width,
            depth_msg->height,
            depth_msg->encoding.c_str(),
            difference
        );
    }

        // ========================================================
        // Subscribers
        // ========================================================

    message_filters::Subscriber<
        Image
    > rgb_sub_;

    message_filters::Subscriber<
        Image
    > depth_sub_;


    std::shared_ptr<
        message_filters::Synchronizer<
            SyncPolicy
        >
    > sync_;

    sensor_msgs::msg::CameraInfo
        depth_camera_info_;

    rclcpp::Subscription<
        sensor_msgs::msg::CameraInfo
    >::SharedPtr camera_info_sub_;


    // ========================================================
    // Publishers
    // ========================================================

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr cloud_pub_;


    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr map_cloud_pub_;


    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr accumulated_cloud_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr rtab_cloud_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr rtab_map_pub_;

    rclcpp::Publisher<
        sensor_msgs::msg::PointCloud2
    >::SharedPtr stored_keyframe_cloud_pub_;
    // ========================================================
    // TF
    // ========================================================

    std::shared_ptr<
        tf2_ros::Buffer
    > tf_buffer_;


    std::shared_ptr<
        tf2_ros::TransformListener
    > tf_listener_;


    // ========================================================
    // Persistent reconstruction
    // ========================================================

    std::unique_ptr<
        IncrementalVoxelMap
    > voxel_map_;

    std::unique_ptr<
        RtabmapReconstructionBackend
    > rtabmap_backend_;

    // ========================================================
    // Keyframe selection
    // ========================================================

    bool have_last_keyframe_pose_{false};

    Eigen::Vector3f last_keyframe_translation_ =
        Eigen::Vector3f::Zero();

    Eigen::Quaternionf last_keyframe_rotation_ =
        Eigen::Quaternionf::Identity();

    double keyframe_translation_threshold_{0.05};  // metres
    double keyframe_rotation_threshold_deg_{1.0};  // degrees

    std::size_t accepted_keyframes_{0};
    std::size_t rejected_frames_{0};
    

    std::chrono::steady_clock::time_point
        last_accumulated_publish_time_ =
            std::chrono::steady_clock::now();

    double accumulated_publish_period_sec_{1.0};
    // ========================================================
    // Camera intrinsics
    // ========================================================

    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};

    bool camera_info_received_{false};
};


int main(
    int argc,
    char ** argv)
{
    rclcpp::init(
        argc,
        argv
    );


    auto node =
        std::make_shared<
            RgbdReconstructionNode
        >();


    rclcpp::spin(
        node
    );


    rclcpp::shutdown();

    return 0;
}