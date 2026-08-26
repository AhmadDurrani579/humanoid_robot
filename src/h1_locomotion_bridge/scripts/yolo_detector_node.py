#!/usr/bin/env python3

import traceback
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__("yolo_detector_node")

        self.bridge = CvBridge()

        self.get_logger().info("Loading YOLO11n...")
        self.model = YOLO("yolo11n.pt")

        self.image_pub = self.create_publisher(
            Image,
            "/semantic/annotated_image",
            5
        )

        self.classes_pub = self.create_publisher(
            String,
            "/semantic/detections",
            5
        )

        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            "YOLO detector ready | "
            "input=/camera/color/image_raw | "
            "device=cuda"
        )


    def image_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="rgb8"
            )
            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )
            
            results = self.model.predict(
                source=image,
                device=0,
                imgsz=640,
                conf=0.50,
                classes=[0],   # COCO class 0 = person
                verbose=False
            )
            result = results[0]

            # ------------------------------------
            # Publish annotated image
            # ------------------------------------

            annotated = result.plot()

            # Make sure YOLO output is contiguous uint8 BGR
            annotated = np.ascontiguousarray(
                annotated,
                dtype=np.uint8
            )

            out_msg = Image()

            out_msg.header = msg.header
            out_msg.height = annotated.shape[0]
            out_msg.width = annotated.shape[1]

            out_msg.encoding = "bgr8"
            out_msg.is_bigendian = 0
            out_msg.step = annotated.shape[1] * 3

            out_msg.data = annotated.tobytes()

            self.image_pub.publish(out_msg)
            
            # ------------------------------------
            # Publish readable detections
            # ------------------------------------

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    class_name = self.model.names[
                        class_id
                    ]

                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = [
                        int(v) for v in xyxy
                    ]

                    detections.append(
                        f"{class_name}"
                        f" conf={confidence:.2f}"
                        f" box=({x1},{y1},{x2},{y2})"
                    )

            detection_msg = String()

            if detections:
                detection_msg.data = " | ".join(
                    detections
                )
            else:
                detection_msg.data = "none"

            self.classes_pub.publish(
                detection_msg
            )

            # if detections:
            #     self.get_logger().info(
            #         detection_msg.data,
            #         throttle_duration_sec=1.0
            #     )

        except Exception as e:

            self.get_logger().error(
                f"YOLO callback error: {repr(e)}"
            )
            print(traceback.format_exc())
            

def main(args=None):

    rclpy.init(args=args)

    node = YoloDetectorNode()

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
