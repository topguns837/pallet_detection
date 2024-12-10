import os
import cv2
import torch
from torchvision import transforms
from torch import nn
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


# Class to perform SegForm inference
class SegForm:
    def __init__(self):
        # Initialize vaiables
        self.repo_id = None
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SegForm] Using device : {self.device}")

    def build_model(self, repo_id):
        # Function to load model to memory
        self.repo_id = repo_id
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.repo_id)
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.repo_id).to(self.device)
            print(f"[SegForm] Model successfully loaded to {self.device}.")
        except Exception as e:
            print(f"[SegForm] Failed to load model with exception: {e}")
            raise Exception("Error loading the model. Ensure the repository ID is correct.")

    def preprocess_image(self, image):
        # Function to perform image pre-processing
        try:
            image = cv2.resize(image, (512, 512))
            # Preprocess using the processor
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            return inputs
        except Exception as e:
            print(f"[SegForm] Error in preprocessing image: {e}")
            raise

    def infer(self, image):
        # Function perform inference
        if self.model is None:
            raise Exception("[SegForm] Model is not loaded. Call `load_model()` first.")

        try:
            inputs = self.preprocess_image(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Extract logits and apply argmax
                logits = outputs.logits

                segmentation_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
                self.visualize_segmentation(image, segmentation_mask)
            
            return segmentation_mask
        except Exception as e:
            print(f"[SegForm] Error during inference: {e}")
            raise

    def visualize_segmentation(self, image, segmentation_mask):
        # Function to visualize inference results
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize the segmentation mask to match the original image dimensions
            segmentation_mask_resized = cv2.resize(segmentation_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Create a blank color overlay
            overlay_colored = np.zeros_like(image, dtype=np.uint8)

            # Define colors for each class (BGR format for OpenCV)
            class_colors = {
                0: (0, 0, 0),      # Background
                1: (0, 0, 255),    # Ground
                2: (0, 255, 0)     # Pallet
            }

            for class_id, color in class_colors.items():
                mask = segmentation_mask_resized == class_id
                overlay_colored[mask] = color

            # Blend the original image and the overlay
            blended = cv2.addWeighted(image, 0.5, overlay_colored, 0.5, 0)

            print("[SegForm] Visualized predictions on the image.")

            # Show inference output
            cv2.imshow("Segmentation Result", blended)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        except Exception as e:
            print(f"[SegForm] Error visualizing segmentation: {e}")
            raise
