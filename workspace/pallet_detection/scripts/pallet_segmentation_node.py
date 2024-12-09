#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from pallet_detection.SegForm import SegForm


class PalletSegmentation(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        hf_repo_id = "topguns/segformer-b0-finetuned-pallet-detection"
        #hf_repo_id = "topguns/segformer-b0-finetuned-segments-sidewalk-outputs"

        self.model = SegForm()
        self.model.build_model(hf_repo_id)

        self.image_subscription = self.create_subscription(
            Image,
            'image_rgb_topic',
            self.rgb_image_callback,
            1)
        
        self.depth_subscription = self.create_subscription(
            Image,
            'image_depth_topic',
            self.depth_image_callback,
            1)

    def rgb_image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.model.infer(cv_image)
        
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

    pallet_segmentation = PalletSegmentation()

    rclpy.spin(pallet_segmentation)

    pallet_segmentation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
