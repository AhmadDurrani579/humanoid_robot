#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


// ============================================================
// Voxel key
// ============================================================

struct VoxelKey
{
    int x;
    int y;
    int z;

    bool operator==(
        const VoxelKey & other) const
    {
        return (
            x == other.x &&
            y == other.y &&
            z == other.z
        );
    }
};


// ============================================================
// Voxel-key hash
// ============================================================

struct VoxelKeyHash
{
    std::size_t operator()(
        const VoxelKey & key) const noexcept
    {
        std::size_t seed = 0;

        auto hash_combine =
            [&seed](int value)
            {
                const std::size_t h =
                    std::hash<int>{}(
                        value
                    );

                seed ^=
                    h +
                    0x9e3779b9 +
                    (seed << 6) +
                    (seed >> 2);
            };

        hash_combine(
            key.x
        );

        hash_combine(
            key.y
        );

        hash_combine(
            key.z
        );

        return seed;
    }
};


// ============================================================
// Data stored inside each persistent voxel
// ============================================================

struct VoxelData
{
    double sum_x{0.0};
    double sum_y{0.0};
    double sum_z{0.0};

    double sum_r{0.0};
    double sum_g{0.0};
    double sum_b{0.0};

    // Total number of individual PointXYZRGB samples
    // integrated into this voxel.
    std::uint32_t point_count{0};

    // Number of DIFFERENT accepted keyframes
    // which observed this voxel.
    std::uint32_t frame_count{0};
};


// ============================================================
// Diagnostic statistics
// ============================================================

struct VoxelObservationStats
{
    std::size_t total_voxels{0};

    std::size_t total_point_observations{0};

    std::size_t frame_count_1{0};

    std::size_t frame_count_2_to_5{0};

    std::size_t frame_count_6_to_20{0};

    std::size_t frame_count_over_20{0};
};


// ============================================================
// Incremental voxel map
// ============================================================

class IncrementalVoxelMap
{
public:

    explicit IncrementalVoxelMap(
        float voxel_size)
        : voxel_size_(
            voxel_size
        )
    {
    }


    // ========================================================
    // Start one accepted keyframe
    // ========================================================

    void beginFrame()
    {
        touched_voxels_this_frame_.clear();
    }


    // ========================================================
    // Finish one accepted keyframe
    //
    // frame_count is incremented ONCE for each unique voxel
    // touched by this keyframe.
    // ========================================================

    void endFrame()
    {
        for (
            const auto & key :
            touched_voxels_this_frame_
        )
        {
            auto it =
                voxels_.find(
                    key
                );

            if (
                it ==
                voxels_.end()
            )
            {
                continue;
            }

            ++it->second.frame_count;
        }

        touched_voxels_this_frame_.clear();
    }


    // ========================================================
    // Integrate one PointXYZRGB observation
    // ========================================================

    void integrate(
        float x,
        float y,
        float z,
        std::uint8_t r,
        std::uint8_t g,
        std::uint8_t b)
    {
        const VoxelKey key =
            makeKey(
                x,
                y,
                z
            );


        auto & voxel =
            voxels_[
                key
            ];


        // ----------------------------------------------------
        // Position accumulation
        // ----------------------------------------------------

        voxel.sum_x +=
            static_cast<double>(
                x
            );

        voxel.sum_y +=
            static_cast<double>(
                y
            );

        voxel.sum_z +=
            static_cast<double>(
                z
            );


        // ----------------------------------------------------
        // Colour accumulation
        // ----------------------------------------------------

        voxel.sum_r +=
            static_cast<double>(
                r
            );

        voxel.sum_g +=
            static_cast<double>(
                g
            );

        voxel.sum_b +=
            static_cast<double>(
                b
            );


        // ----------------------------------------------------
        // Point-observation count
        // ----------------------------------------------------

        ++voxel.point_count;


        // ----------------------------------------------------
        // Mark this voxel as observed by CURRENT keyframe.
        //
        // unordered_set guarantees that if 20 pixels from
        // this same keyframe hit this voxel, the voxel still
        // receives only ONE frame confirmation at endFrame().
        // ----------------------------------------------------

        touched_voxels_this_frame_.insert(
            key
        );
    }


    // ========================================================
    // Number of persistent voxels
    // ========================================================

    std::size_t size() const
    {
        return voxels_.size();
    }


    // ========================================================
    // Frame-observation diagnostics
    // ========================================================

    VoxelObservationStats
    getObservationStats() const
    {
        VoxelObservationStats stats;

        stats.total_voxels =
            voxels_.size();


        for (
            const auto & item :
            voxels_
        )
        {
            const auto & voxel =
                item.second;


            stats.total_point_observations +=
                static_cast<std::size_t>(
                    voxel.point_count
                );


            if (
                voxel.frame_count == 1
            )
            {
                ++stats.frame_count_1;
            }
            else if (
                voxel.frame_count >= 2 &&
                voxel.frame_count <= 5
            )
            {
                ++stats.frame_count_2_to_5;
            }
            else if (
                voxel.frame_count >= 6 &&
                voxel.frame_count <= 20
            )
            {
                ++stats.frame_count_6_to_20;
            }
            else if (
                voxel.frame_count > 20
            )
            {
                ++stats.frame_count_over_20;
            }
        }


        return stats;
    }


    // ========================================================
    // Hash-map diagnostics
    // ========================================================

    std::size_t bucketCount() const
    {
        return voxels_.bucket_count();
    }


    float loadFactor() const
    {
        return voxels_.load_factor();
    }


    std::size_t maxBucketSize() const
    {
        std::size_t max_size = 0;


        for (
            std::size_t i = 0;
            i < voxels_.bucket_count();
            ++i
        )
        {
            max_size =
                std::max(
                    max_size,
                    voxels_.bucket_size(
                        i
                    )
                );
        }


        return max_size;
    }


    // ========================================================
    // Persistent voxel map -> PointXYZRGB cloud
    // ========================================================

    pcl::PointCloud<
        pcl::PointXYZRGB
    > toPointCloud() const
    {
        pcl::PointCloud<
            pcl::PointXYZRGB
        > cloud;

        cloud.points.reserve(
            voxels_.size()
        );

        for (const auto & item : voxels_)
        {
            const auto & voxel =
                item.second;

            if (voxel.point_count == 0)
            {
                continue;
            }

            const double count =
                static_cast<double>(
                    voxel.point_count
                );

            pcl::PointXYZRGB point;

            point.x =
                static_cast<float>(
                    voxel.sum_x / count
                );

            point.y =
                static_cast<float>(
                    voxel.sum_y / count
                );

            point.z =
                static_cast<float>(
                    voxel.sum_z / count
                );

            point.r =
                static_cast<std::uint8_t>(
                    voxel.sum_r / count
                );

            point.g =
                static_cast<std::uint8_t>(
                    voxel.sum_g / count
                );

            point.b =
                static_cast<std::uint8_t>(
                    voxel.sum_b / count
                );

            cloud.points.push_back(
                point
            );
        }

        cloud.width =
            static_cast<std::uint32_t>(
                cloud.points.size()
            );

        cloud.height = 1;
        cloud.is_dense = false;

        return cloud;
    }

    pcl::PointCloud<
        pcl::PointXYZRGB
    > toFilteredPointCloud(
        std::uint32_t minimum_frame_count) const
    {
        pcl::PointCloud<
            pcl::PointXYZRGB
        > cloud;

        cloud.points.reserve(
            voxels_.size()
        );

        for (const auto & item : voxels_)
        {
            const auto & voxel =
                item.second;

            if (voxel.point_count == 0)
            {
                continue;
            }

            if (
                voxel.frame_count <
                minimum_frame_count
            )
            {
                continue;
            }

            const double count =
                static_cast<double>(
                    voxel.point_count
                );

            pcl::PointXYZRGB point;

            point.x =
                static_cast<float>(
                    voxel.sum_x / count
                );

            point.y =
                static_cast<float>(
                    voxel.sum_y / count
                );

            point.z =
                static_cast<float>(
                    voxel.sum_z / count
                );

            point.r =
                static_cast<std::uint8_t>(
                    voxel.sum_r / count
                );

            point.g =
                static_cast<std::uint8_t>(
                    voxel.sum_g / count
                );

            point.b =
                static_cast<std::uint8_t>(
                    voxel.sum_b / count
                );

            cloud.points.push_back(
                point
            );
        }

        cloud.width =
            static_cast<std::uint32_t>(
                cloud.points.size()
            );

        cloud.height = 1;
        cloud.is_dense = false;

        return cloud;
    }

private:

    // ========================================================
    // XYZ -> integer voxel index
    // ========================================================

    VoxelKey makeKey(
        float x,
        float y,
        float z) const
    {
        return VoxelKey{
            static_cast<int>(
                std::floor(
                    x /
                    voxel_size_
                )
            ),

            static_cast<int>(
                std::floor(
                    y /
                    voxel_size_
                )
            ),

            static_cast<int>(
                std::floor(
                    z /
                    voxel_size_
                )
            )
        };
    }


    // ========================================================
    // Configuration
    // ========================================================

    float voxel_size_;


    // ========================================================
    // Persistent reconstruction
    // ========================================================

    std::unordered_map<
        VoxelKey,
        VoxelData,
        VoxelKeyHash
    > voxels_;


    // ========================================================
    // Temporary state for CURRENT accepted keyframe
    //
    // Cleared by beginFrame().
    // Filled by integrate().
    // Consumed by endFrame().
    // ========================================================

    std::unordered_set<
        VoxelKey,
        VoxelKeyHash
    > touched_voxels_this_frame_;
};