#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge

from pallet_detection.SegForm import SegForm


# Class to segment pallets using SegForm
class PalletSegmentation(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        # Create CV Bridge object
        self.bridge = CvBridge()

        # HuggingFaceHub Repo ID
        hf_repo_id = "topguns/segformer-b0-finetuned-pallet-detection"

        # Load the SegForm model
        self.model = SegForm()
        self.model.build_model(hf_repo_id)

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

        # Pallet Segmentation inference
        self.model.infer(cv_image)

    def depth_image_callback(self, msg):
        # Convert depth image message to numpy array
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


def main(args=None):
    rclpy.init(args=args)

    pallet_segmentation = PalletSegmentation()

    rclpy.spin(pallet_segmentation)

    pallet_segmentation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
