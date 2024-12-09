#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os


class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.publisher_ = self.create_publisher(Image, '/zed2i/zed_node/rgb/image_rect_color', 10)
        self.bridge = CvBridge()
        
        # Timer to periodically call the publish_image method (e.g., 1Hz)
        self.timer = self.create_timer(1.0, self.publish_image)
        
        # Path to the image file (modify this to your file path)
        self.image_path = '/root/pallet_ws/src/pallet_detection/models/images/test1.jpg'
        
        if not os.path.isfile(self.image_path):
            self.get_logger().error(f"Image file {self.image_path} does not exist.")
            return
        
        # Read the image using OpenCV
        self.image = cv2.imread(self.image_path)
        
        if self.image is None:
            self.get_logger().error(f"Failed to load image from {self.image_path}")
            return

    def publish_image(self):
        if self.image is not None:
            # Convert the image (OpenCV format) to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(self.image, encoding='rgb8')
            self.publisher_.publish(ros_image)
            self.get_logger().info('Publishing image...')
        else:
            self.get_logger().warn('No image to publish.')


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    node.destroy()

if __name__ == '__main__':
    main()
