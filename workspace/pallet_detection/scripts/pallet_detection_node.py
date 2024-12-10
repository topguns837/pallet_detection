#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from pallet_detection.YOLOv8 import YOLOv8


# Class to detect pallets using YOLOV8 
class PalletDetection(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Create CV Bridge object
        self.bridge = CvBridge()

        # Find location of pallet_detection package
        self.pallet_detection_dir = self.find_package("pallet_detection")

        # Path to the weight file
        model_dir_path = os.path.join(self.pallet_detection_dir, "models", "yolov8")
        weight_file = "train3.pt"

        # Load the YOLOv8 model
        self.model = YOLOv8()
        self.model.build_model(model_dir_path, weight_file)
        self.model.load_classes(model_dir_path)

        # ROS 2 Subscriber for image and depth topics
        self.image_subscription = self.create_subscription(
            Image,
            '/zed2i/zed_node/rgb/image_rect_color',
            self.rgb_image_callback,
            1)
        
        self.depth_subscription = self.create_subscription(
            Image,
            '/zed2i/zed_node/depth/depth_registered',
            self.depth_image_callback,
            1)

    def rgb_image_callback(self, msg):
        # Convert image message to numpy array
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Pallet Detection inference
        predictions = self.model.get_predictions(cv_image)

    def depth_image_callback(self, msg):
        # Convert depth image message to numpy array
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
    def find_package(self, package_name):
        try:
            # Get the share directory of the specified package
            package_path = get_package_share_directory(package_name)
            print(f"Package '{package_name}' is located at: {package_path}")
        except Exception as e:
            print(f"Error: {e}")

        return package_path


def main(args=None):
    rclpy.init(args=args)

    pallet_detection = PalletDetection()

    rclpy.spin(pallet_detection)

    pallet_detection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
