import os
import cv2
import torch
from torchvision import transforms
from torch import nn
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


class SegForm:
    def __init__(self):
        """
        Initialize the SegForm class for segmentation inference.

        Args:
            repo_id (str): The Hugging Face Hub repository ID.
        """
        self.repo_id = None
        self.model = None
        self.processor = None

    def build_model(self, repo_id):
        """
        Load the segmentation model and processor from the Hugging Face Hub.
        """

        self.repo_id = repo_id
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.repo_id)
            print("[SegForm] Model successfully loaded.")
        except Exception as e:
            print(f"[SegForm] Failed to load model with exception: {e}")
            raise Exception("Error loading the model. Ensure the repository ID is correct.")

    def preprocess_image(self, image):
        """
        Preprocess the input image for segmentation.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            torch.Tensor: Preprocessed image.
        """
        try:
            # Convert BGR to RGB
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (512, 512))
            # Preprocess using the processor
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs
        except Exception as e:
            print(f"[SegForm] Error in preprocessing image: {e}")
            raise

    def infer(self, image):
        """
        Perform segmentation inference on the input image.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Segmentation mask as a numpy array.
        """
        if self.model is None:
            raise Exception("[SegForm] Model is not loaded. Call `load_model()` first.")

        try:
            inputs = self.preprocess_image(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Extract logits and apply argmax
                logits = outputs.logits  # Shape: (batch_size, num_classes, height, width)

                segmentation_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                self.visualize_segmentation(image, segmentation_mask)
            
            return segmentation_mask
        except Exception as e:
            print(f"[SegForm] Error during inference: {e}")
            raise

    def visualize_segmentation(self, image, segmentation_mask):
        """
        Visualize the segmentation results on the input image with specific colors for each class.

        Args:
            image (np.ndarray): Input image in BGR format.
            segmentation_mask (np.ndarray): Segmentation mask as a numpy array.
        """
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize the segmentation mask to match the original image dimensions
            segmentation_mask_resized = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create a blank color overlay
            overlay_colored = np.zeros_like(image, dtype=np.uint8)

            # Define colors for each class (BGR format for OpenCV)
            class_colors = {
                0: (0, 0, 0),
                1: (0, 0, 255), 
                2: (0, 255, 0)
            }

            # Apply the color mapping
            for class_id, color in class_colors.items():
                mask = segmentation_mask_resized == class_id
                overlay_colored[mask] = color

            # Blend the original image and the overlay
            blended = cv2.addWeighted(image, 0.5, overlay_colored, 0.5, 0)

            # Show the blended image
            cv2.imshow("Segmentation Result", blended)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        except Exception as e:
            print(f"[SegForm] Error visualizing segmentation: {e}")
            raise


if __name__ == "__main__":
    # Input image
    image_path = "/root/pallet_ws/src/pallet_detection/models/yolov8/test4.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("[SegForm] Failed to load input image. Check the path.")
        exit()

    # Initialize and use SegForm
    repo_id = "topguns/segformer-b0-finetuned-segments-sidewalk-outputs"
    segform = SegForm(repo_id)
    segform.load_model()
    print(f"Number of output classes: {segform.model.config.num_labels}")
    seg_mask = segform.infer(image)
    segform.visualize_segmentation(image, seg_mask)
