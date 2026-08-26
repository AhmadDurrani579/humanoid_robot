#!/usr/bin/env python3

from __future__ import annotations

import math
import re
from dataclasses import dataclass
import time
import numpy as np
from geometry_msgs.msg import PointStamped
import tf2_geometry_msgs
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.time import Time

from tf2_ros import (
    Buffer,
    TransformException,
    TransformListener,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import String
from rclpy.duration import Duration

# ============================================================
# Detection representation
# ============================================================

@dataclass
class Detection:
    label: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int


# ============================================================
# Semantic RGB-D fusion node
# ============================================================

class SemanticDepthFusionNode(Node):

    def __init__(self) -> None:
        super().__init__("semantic_depth_fusion_node")

        # ----------------------------------------------------
        # Current YOLO image size.
        #
        # YOLO detections originate from:
        # /camera/color/image_raw
        #
        # Current resolution:
        # 640 x 480
        # ----------------------------------------------------

        self.rgb_width = 640
        self.rgb_height = 480

        # ----------------------------------------------------
        # Depth filtering
        # ----------------------------------------------------

        self.minimum_depth_m = 0.20
        self.maximum_depth_m = 8.0

        # Use only the central portion of the YOLO bbox.
        #
        # 0.50 means:
        # central 50% width
        # central 50% height
        #
        # This reduces contamination from background pixels.
        self.roi_fraction = 0.50

        # Require a minimum number of valid depth pixels.
        self.minimum_valid_pixels = 20

        # ----------------------------------------------------
        # Latest sensor data
        # ----------------------------------------------------

        self.latest_depth: np.ndarray | None = None
        self.latest_depth_stamp = None
        self.latest_depth_frame = ""

        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None

        self.camera_info_width: int | None = None
        self.camera_info_height: int | None = None
        self.camera_frame = ""
        
        # ----------------------------------------------------
        # Simple semantic object tracking
        # ----------------------------------------------------

        self.tracks = {}
        self.active_marker_ids = set()
        self.semantic_tracks = {}


        self.next_track_id = 1

        # Maximum map-plane distance for associating
        # a new detection with an existing object.
        self.track_match_distance = 0.75

        # Light exponential smoothing.
        self.track_alpha = 0.03

        # Keep a track briefly through missed detections.
        self.track_timeout_sec = 1.0
        
        self.last_marker_publish_time = 0.0
        self.marker_publish_period = 0.2

        # ----------------------------------------------------
        # Detection regex
        #
        # Example:
        #
        # person conf=0.89 box=(317,223,366,330)
        # ----------------------------------------------------

        self.semantic_filtered_positions = {}
        self.semantic_alpha = 0.15
        
        
        self.map_frame = "map"

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )
        
        self.detection_pattern = re.compile(
            r"(?P<label>[A-Za-z0-9_-]+)"
            r"\s+conf=(?P<confidence>[0-9]*\.?[0-9]+)"
            r"\s+box=\("
            r"(?P<x1>-?\d+),"
            r"(?P<y1>-?\d+),"
            r"(?P<x2>-?\d+),"
            r"(?P<y2>-?\d+)"
            r"\)"
        )

        # ----------------------------------------------------
        # Subscribers
        # ----------------------------------------------------

        self.depth_subscription = self.create_subscription(
            Image,
            "/camera/depth/image_nav",
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera/depth/camera_info_nav",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.detection_subscription = self.create_subscription(
            String,
            "/semantic/detections",
            self.detection_callback,
            10,
        )
        
        # ----------------------------------------------------
        # Semantic 3D outputs
        # ----------------------------------------------------

        self.objects_3d_pub = self.create_publisher(
            String,
            "/semantic/objects_3d",
            10,
        )

        self.markers_pub = self.create_publisher(
            MarkerArray,
            "/semantic/markers",
            10,
        )

        self.get_logger().info(
            "Semantic depth fusion node started | "
            "detections=/semantic/detections | "
            "depth=/camera/depth/image_nav | "
            "camera_info=/camera/depth/camera_info_nav"
        )

    # ========================================================
    # Camera info
    # ========================================================

    def camera_info_callback(
        self,
        msg: CameraInfo,
    ) -> None:

        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])

        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

        self.camera_info_width = int(msg.width)
        self.camera_info_height = int(msg.height)

        self.camera_frame = msg.header.frame_id

    # ========================================================
    # Depth image conversion
    # ========================================================

    def depth_callback(
        self,
        msg: Image,
    ) -> None:

        try:
            depth = self.convert_depth_image(msg)

        except Exception as error:
            self.get_logger().warning(
                f"Failed to decode depth image: {error}",
                throttle_duration_sec=2.0,
            )
            return

        self.latest_depth = depth

        self.latest_depth_stamp = msg.header.stamp
        self.latest_depth_frame = msg.header.frame_id

    def convert_depth_image(
        self,
        msg: Image,
    ) -> np.ndarray:
        """
        Convert ROS depth Image directly into a NumPy array.

        Supported:
          32FC1 -> metres
          16UC1 -> millimetres, converted to metres

        We intentionally avoid cv_bridge here.
        """

        height = int(msg.height)
        width = int(msg.width)

        encoding = msg.encoding.upper()

        if encoding == "32FC1":

            row_elements = msg.step // 4

            raw = np.frombuffer(
                msg.data,
                dtype=np.float32,
            ).reshape(
                height,
                row_elements,
            )

            depth = raw[:, :width].copy()

        elif encoding in ("16UC1", "MONO16"):

            row_elements = msg.step // 2

            raw = np.frombuffer(
                msg.data,
                dtype=np.uint16,
            ).reshape(
                height,
                row_elements,
            )

            depth = (
                raw[:, :width].astype(np.float32)
                / 1000.0
            )

        else:
            raise ValueError(
                f"Unsupported depth encoding: {msg.encoding}"
            )

        return depth

    
    
    def transform_camera_point_to_map(
        self,
        x_camera: float,
        y_camera: float,
        z_camera: float,
    ) -> tuple[float, float, float] | None:
        """
        Transform a reconstructed point from the depth optical frame
        into the fixed Nav2 map frame.
        """

        if not self.latest_depth_frame:
            return None

        point_camera = PointStamped()

        point_camera.header.frame_id = (
            self.latest_depth_frame
        )

        point_camera.header.stamp = (
            self.latest_depth_stamp
        )

        point_camera.point.x = x_camera
        point_camera.point.y = y_camera
        point_camera.point.z = z_camera

        try:
            
            
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, 
                point_camera.header.frame_id,
                Time()
            )

            point_map = tf2_geometry_msgs.do_transform_point(
                point_camera,
                transform
            )
            
        except TransformException as error:

            self.get_logger().warning(
                f"TF failed "
                f"{self.latest_depth_frame} -> "
                f"{self.map_frame}: {error}",
                throttle_duration_sec=2.0,
            )

            return None

        return (
            float(point_map.point.x),
            float(point_map.point.y),
            float(point_map.point.z),
        )
    
    # ========================================================
    # YOLO string parsing
    # ========================================================

    def parse_detections(
        self,
        text: str,
    ) -> list[Detection]:

        detections: list[Detection] = []

        for match in self.detection_pattern.finditer(text):

            detections.append(
                Detection(
                    label=match.group("label"),
                    confidence=float(
                        match.group("confidence")
                    ),
                    x1=int(match.group("x1")),
                    y1=int(match.group("y1")),
                    x2=int(match.group("x2")),
                    y2=int(match.group("y2")),
                )
            )

        return detections

    # ========================================================
    # Detection callback
    # ========================================================
    
    
    def update_semantic_track(
        self,
        label: str,
        confidence: float,
        x_map: float,
        y_map: float,
        z_map: float,
    ):
        """
        Associate detection with existing semantic object.
        Returns:
            track_id,
            smoothed x,y,z
        """

        now = time.monotonic()

        # ----------------------------------------------------
        # Remove old tracks
        # ----------------------------------------------------

        remove_ids = []

        for track_id, track in self.semantic_tracks.items():

            if (
                now - track["last_seen"]
                > self.track_timeout_sec
            ):
                remove_ids.append(track_id)


        for track_id in remove_ids:
            del self.semantic_tracks[track_id]


        # ----------------------------------------------------
        # Search nearest existing object
        # ----------------------------------------------------

        matched_id = None

        best_distance = float("inf")


        for track_id, track in self.semantic_tracks.items():

            if track["label"] != label:
                continue


            distance = math.sqrt(
                (x_map - track["x"]) ** 2 +
                (y_map - track["y"]) ** 2
            )


            if (
                distance < self.track_match_distance
                and distance < best_distance
            ):

                best_distance = distance
                matched_id = track_id


        # ----------------------------------------------------
        # New object
        # ----------------------------------------------------

        if matched_id is None:

            track_id = self.next_track_id

            self.next_track_id += 1


            self.semantic_tracks[track_id] = {

                "label": label,

                "x": x_map,
                "y": y_map,
                "z": z_map,

                "confidence": confidence,

                "last_seen": now,
            }


            return (
                track_id,
                x_map,
                y_map,
                z_map,
            )


        # ----------------------------------------------------
        # Update existing object
        # ----------------------------------------------------

        track = self.semantic_tracks[matched_id]


        a = self.track_alpha


        track["x"] = (
            a * x_map
            +
            (1.0 - a) * track["x"]
        )


        track["y"] = (
            a * y_map
            +
            (1.0 - a) * track["y"]
        )

        track["z"] = (
            self.track_alpha * z_map
            +
            (1.0 - self.track_alpha) * track["z"]
        )
        
        track["confidence"] = confidence

        track["last_seen"] = now


        return (
            matched_id,
            track["x"],
            track["y"],
            track["z"],
        )
        
        
    def filter_semantic_position(
        self,
        object_id: str,
        x: float,
        y: float,
        z: float,
    ):
        """
        Smooth semantic object position using EMA.
        """

        alpha = self.semantic_alpha

        if object_id not in self.semantic_filtered_positions:

            self.semantic_filtered_positions[object_id] = (
                x,
                y,
                z,
            )

            return x, y, z


        old_x, old_y, old_z = (
            self.semantic_filtered_positions[object_id]
        )


        new_x = (
            alpha * x +
            (1.0-alpha) * old_x
        )

        new_y = (
            alpha * y +
            (1.0-alpha) * old_y
        )

        new_z = (
            alpha * z +
            (1.0-alpha) * old_z
        )


        self.semantic_filtered_positions[object_id] = (
            new_x,
            new_y,
            new_z,
        )

        return (
            new_x,
            new_y,
            new_z,
        )   
    
    def detection_callback(
        self,
        msg: String,
    ) -> None:

        # --------------------------------------------------------
        # No detections
        # --------------------------------------------------------

        if msg.data.strip().lower() == "none":

            output_msg = String()
            output_msg.data = "none"

            self.objects_3d_pub.publish(
                output_msg
            )

          #  self.clear_semantic_markers()

            return

        # --------------------------------------------------------
        # Make sure depth is available
        # --------------------------------------------------------

        if self.latest_depth is None:

            self.get_logger().warning(
                "Detection received but no depth image available yet.",
                throttle_duration_sec=2.0,
            )

            return

        # --------------------------------------------------------
        # Make sure camera calibration is available
        # --------------------------------------------------------

        if (
            self.fx is None
            or self.fy is None
            or self.cx is None
            or self.cy is None
        ):

            self.get_logger().warning(
                "Detection received but CameraInfo is unavailable.",
                throttle_duration_sec=2.0,
            )

            return

        # --------------------------------------------------------
        # Parse YOLO detections
        # --------------------------------------------------------

        detections = self.parse_detections(
            msg.data
        )
        detections = self.remove_duplicate_detections(
            detections
        )

        if not detections:

            self.get_logger().warning(
                f"Could not parse detection string: {msg.data}",
                throttle_duration_sec=2.0,
            )

            return

        # --------------------------------------------------------
        # Output containers for this detection frame
        # --------------------------------------------------------

        marker_array = MarkerArray()

        object_output_lines: list[str] = []

        detection_index = 0

        # --------------------------------------------------------
        # Process every detected object
        # --------------------------------------------------------

        for detection in detections:

            # ----------------------------------------------------
            # YOLO bbox + depth -> camera-frame XYZ
            # ----------------------------------------------------

            result = self.reconstruct_detection(
                detection
            )

            if result is None:
                continue

            (
                x_camera,
                y_camera,
                z_camera,
                depth_m,
                valid_count,
            ) = result

            # ----------------------------------------------------
            # camera optical frame -> map frame
            # ----------------------------------------------------

            map_result = (
                self.transform_camera_point_to_map(
                    x_camera,
                    y_camera,
                    z_camera,
                )
            )

            if map_result is None:
                continue

            (
                x_map,
                y_map,
                z_map,
            ) = map_result
            
            (
                x_map,
                y_map,
                z_map,
            ) = self.filter_semantic_position(
                detection.label,
                x_map,
                y_map,
                z_map,
            )

            (
                track_id,
                tracked_x,
                tracked_y,
                tracked_z,
            ) = self.update_semantic_track(
                label=detection.label,
                confidence=detection.confidence,
                x_map=x_map,
                y_map=y_map,
                z_map=z_map,
            )
            
            # ----------------------------------------------------
            # Build /semantic/objects_3d output
            # ----------------------------------------------------
            
            object_output_lines.append(
                f"{detection.label}_{track_id:02d} "
                f"conf={detection.confidence:.2f} "
                f"map=("
                f"{tracked_x:.3f},"
                f"{tracked_y:.3f},"
                f"{tracked_z:.3f}"
                f")"
            )
            
            # ----------------------------------------------------
            # Build RViz markers
            # ----------------------------------------------------
            
            new_markers = (
                self.create_semantic_markers(
                    label=f"{detection.label}_{track_id:02d}",
                    confidence=detection.confidence,
                    x_map=tracked_x,
                    y_map=tracked_y,
                    z_map=tracked_z,
                    detection_index=track_id,
                )
            )

            marker_array.markers.extend(
                new_markers
            )

            detection_index += 1

            # ----------------------------------------------------
            # Console diagnostics
            # ----------------------------------------------------

        # --------------------------------------------------------
        # Publish all valid reconstructed objects from this frame
        # --------------------------------------------------------

        if object_output_lines:

            output_msg = String()

            output_msg.data = "\n".join(
                object_output_lines
            )

            self.objects_3d_pub.publish(
                output_msg
            )

            now = time.monotonic()

            if (
                now - self.last_marker_publish_time
                >= self.marker_publish_period
            ):

                # --------------------------------------------------------
                # Remove markers that disappeared
                # --------------------------------------------------------
                current_ids = set()

                for marker in marker_array.markers:

                    if marker.ns == "semantic_objects":
                        current_ids.add(marker.id)


                for old_id in self.active_marker_ids - current_ids:

                    delete_sphere = Marker()

                    delete_sphere.header.frame_id = self.map_frame
                    delete_sphere.header.stamp = (
                        self.get_clock().now().to_msg()
                    )

                    delete_sphere.ns = "semantic_objects"
                    delete_sphere.id = old_id
                    delete_sphere.action = Marker.DELETE


                    marker_array.markers.append(
                        delete_sphere
                    )


                    delete_text = Marker()

                    delete_text.header.frame_id = self.map_frame
                    delete_text.header.stamp = (
                        self.get_clock().now().to_msg()
                    )

                    delete_text.ns = "semantic_labels"
                    delete_text.id = 1000 + old_id
                    delete_text.action = Marker.DELETE


                    marker_array.markers.append(
                        delete_text
                    )

                self.active_marker_ids = current_ids

                # Publish markers
                self.markers_pub.publish(
                    marker_array
)
                self.last_marker_publish_time = now
            
    def clear_semantic_markers(self) -> None:
        """
        Remove currently displayed semantic markers from RViz.
        """

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.map_frame
        delete_marker.header.stamp = self.get_clock().now().to_msg()

        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)

        self.markers_pub.publish(marker_array)
        
        
    def remove_duplicate_detections(
        self,
        detections,
        iou_threshold=0.5,
    ):
        """
        Remove duplicate YOLO boxes.
        Keep highest confidence detection.
        """

        if len(detections) <= 1:
            return detections


        keep = []

        detections = sorted(
            detections,
            key=lambda d: d.confidence,
            reverse=True,
        )


        while detections:

            current = detections.pop(0)

            keep.append(current)

            remaining = []

            for other in detections:

                iou = self.compute_iou(
                    current,
                    other
                )

                if iou < iou_threshold:
                    remaining.append(other)

            detections = remaining


        return keep

    def compute_iou(
        self,
        a: Detection,
        b: Detection,
    ) -> float:

        ax1 = a.x1
        ay1 = a.y1
        ax2 = a.x2
        ay2 = a.y2

        bx1 = b.x1
        by1 = b.y1
        bx2 = b.x2
        by2 = b.y2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if (
            inter_x2 <= inter_x1
            or inter_y2 <= inter_y1
        ):
            return 0.0

        intersection = (
            (inter_x2 - inter_x1)
            *
            (inter_y2 - inter_y1)
        )

        area_a = (
            (ax2 - ax1)
            *
            (ay2 - ay1)
        )

        area_b = (
            (bx2 - bx1)
            *
            (by2 - by1)
        )

        union = (
            area_a
            +
            area_b
            -
            intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union
    
    
    def create_semantic_markers(
        self,
        label: str,
        confidence: float,
        x_map: float,
        y_map: float,
        z_map: float,
        detection_index: int,
    ) -> list[Marker]:
        """
        Create:
        1. sphere at reconstructed 3D position
        2. floating text label above the sphere
        """

        markers: list[Marker] = []

        # ------------------------------------------------
        # Stable ID from tracking label
        # Example:
        # person_01 -> marker id 1
        # ------------------------------------------------

        try:
            track_id = int(
                label.split("_")[-1]
            )

        except Exception:
            track_id = detection_index + 1


        sphere_id = track_id

        text_id = 1000 + track_id


        # ------------------------------------------------
        # 3D point marker
        # ------------------------------------------------

        sphere = Marker()

        sphere.header.frame_id = self.map_frame
        sphere.header.stamp = self.get_clock().now().to_msg()

        sphere.ns = "semantic_objects"
        sphere.id = sphere_id

        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD

        sphere.pose.position.x = x_map
        sphere.pose.position.y = y_map
        
        if label.startswith("person"):
            sphere.pose.position.z = z_map
        else:
            sphere.pose.position.z = z_map


        sphere.scale.x = 0.20
        sphere.scale.y = 0.20
        sphere.scale.z = 0.20

        sphere.color.r = 0.1
        sphere.color.g = 1.0
        sphere.color.b = 0.1
        sphere.color.a = 0.95


        # Keep marker alive
        sphere.lifetime = Duration().to_msg()
        
        markers.append(sphere)


        # ------------------------------------------------
        # Text marker
        # ------------------------------------------------

        text = Marker()

        text.header.frame_id = self.map_frame
        text.header.stamp = self.get_clock().now().to_msg()

        text.ns = "semantic_labels"

        text.id = text_id

        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD


        text.pose.position.x = x_map
        text.pose.position.y = y_map

        if label.startswith("person"):
            text.pose.position.z = z_map + 0.35
        else:
            text.pose.position.z = 1.8

        text.scale.z = 0.25


        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0


        text.text = (
            f"{label} "
            f"{confidence:.2f}\n"
            f"({x_map:.2f}, {y_map:.2f})"
        )


        text.lifetime = Duration().to_msg()

        markers.append(text)


        return markers
 
            
    # ========================================================
    # 3D reconstruction
    # ========================================================

    def reconstruct_detection(
        self,
        detection: Detection,
    ) -> tuple[
        float,
        float,
        float,
        float,
        int,
    ] | None:

        assert self.latest_depth is not None
        assert self.fx is not None
        assert self.fy is not None
        assert self.cx is not None
        assert self.cy is not None

        depth = self.latest_depth

        depth_height, depth_width = depth.shape

        # ----------------------------------------------------
        # Scale RGB bbox coordinates into nav-depth image.
        #
        # Current expected scale:
        #
        # RGB:   640 x 480
        # Depth: 320 x 240
        #
        # => scale_x = scale_y = 0.5
        # ----------------------------------------------------

        scale_x = depth_width / float(self.rgb_width)
        scale_y = depth_height / float(self.rgb_height)

        x1 = detection.x1 * scale_x
        y1 = detection.y1 * scale_y

        x2 = detection.x2 * scale_x
        y2 = detection.y2 * scale_y

        # ----------------------------------------------------
        # Original bbox centre
        # ----------------------------------------------------

        center_u = (x1 + x2) * 0.5
        center_v = (y1 + y2) * 0.5

        bbox_width = max(1.0, x2 - x1)
        bbox_height = max(1.0, y2 - y1)

        # ----------------------------------------------------
        # Central ROI
        # ----------------------------------------------------

        roi_half_width = (
            bbox_width
            * self.roi_fraction
            * 0.5
        )

        roi_half_height = (
            bbox_height
            * self.roi_fraction
            * 0.5
        )

        roi_x1 = int(
            max(
                0,
                math.floor(
                    center_u - roi_half_width
                ),
            )
        )

        roi_x2 = int(
            min(
                depth_width,
                math.ceil(
                    center_u + roi_half_width
                ),
            )
        )

        roi_y1 = int(
            max(
                0,
                math.floor(
                    center_v - roi_half_height
                ),
            )
        )

        roi_y2 = int(
            min(
                depth_height,
                math.ceil(
                    center_v + roi_half_height
                ),
            )
        )

        if (
            roi_x2 <= roi_x1
            or roi_y2 <= roi_y1
        ):
            return None

        roi = depth[
            roi_y1:roi_y2,
            roi_x1:roi_x2,
        ]

        # ----------------------------------------------------
        # Remove invalid depth
        # ----------------------------------------------------

        valid_mask = (
            np.isfinite(roi)
            & (roi > self.minimum_depth_m)
            & (roi < self.maximum_depth_m)
        )

        valid_depth = roi[valid_mask]

        valid_count = int(valid_depth.size)

        if valid_count < self.minimum_valid_pixels:

            self.get_logger().warning(
                f"{detection.label} "
                f"conf={detection.confidence:.2f} | "
                f"insufficient valid depth pixels "
                f"({valid_count})",
                throttle_duration_sec=1.0,
            )

            return None

        # ----------------------------------------------------
        # Robust object depth estimate
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Robust object depth estimate
        # ----------------------------------------------------

        # Remove extreme depth values
        q1 = np.percentile(valid_depth, 25)
        q3 = np.percentile(valid_depth, 75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        filtered_depth = valid_depth[
            (valid_depth >= lower)
            &
            (valid_depth <= upper)
        ]

        if filtered_depth.size < self.minimum_valid_pixels:
            return None


        z_camera = float(
            np.median(filtered_depth)
        )
        
        
        # Use bbox centre for the projection ray.
        u = float(center_u)
        v = float(center_v)

        # ----------------------------------------------------
        # ROS optical-frame convention:
        #
        # X = right
        # Y = down
        # Z = forward
        # ----------------------------------------------------

        x_camera = (
            (u - self.cx)
            * z_camera
            / self.fx
        )

        y_camera = (
            (v - self.cy)
            * z_camera
            / self.fy
        )

        return (
            float(x_camera),
            float(y_camera),
            float(z_camera),
            float(z_camera),
            valid_count,
        )


# ============================================================
# Main
# ============================================================

def main(args=None) -> None:

    rclpy.init(args=args)

    node = SemanticDepthFusionNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()