import cv2
import os
from ultralytics import YOLO


class YOLOv8:
    def __init__(self):
        pass

    def build_model(self, model_dir_path, weight_file_name):
        try:
            model_path = os.path.join(model_dir_path, weight_file_name)
            self.model = YOLO(model_path)
            print("[YOLOv8] Model successfully loaded from: {}".format(model_path))
        except Exception as e:
            print("[YOLOv8] Failed to load model with exception: {}".format(e))
            raise Exception("Error loading the given model from path: {}.".format(model_path) +
                            " Make sure the file exists and the format is correct.")

    def load_classes(self, model_dir_path):
        self.class_list = []
        fpath = os.path.join(model_dir_path, "classes.txt")

        try:
            with open(fpath, "r") as f:
                self.class_list = [cname.strip() for cname in f.readlines()]
                print("Classes file loaded successfully")
        except FileNotFoundError:
            print("[YOLOv8] Classes file not found at path: {}".format(fpath))
            raise FileNotFoundError("Classes file not found. Make sure the file exists at the specified path.")
        except Exception as e:
            print("[YOLOv8] Error loading classes with exception: {}".format(e))
            raise Exception("Error loading classes from file: {}".format(fpath))

        return self.class_list
    
    def create_predictions_list(self, class_ids, confidences, boxes):
        self.predictions = []
        for i in range(len(class_ids)):
            obj_dict = {
                "class_id": class_ids[i],
                "confidence": confidences[i],
                "box": boxes[i]
            }

            self.predictions.append(obj_dict)

    def get_predictions(self, cv_image):
        if cv_image is None:
            print("[YOLOv8] Input image is None. No predictions will be generated.")
            return None, None
        else:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.frame = cv_image
            class_id = []
            confidence = []
            boxes = []

            # Perform object detection on image
            try:
                result = self.model.predict(self.frame, verbose=False)
                row = result[0].boxes.cpu()

                for box in row:
                    class_id.append(box.cls.numpy().tolist()[0])
                    confidence.append(box.conf.numpy().tolist()[0])
                    boxes.append(box.xyxy.numpy().tolist()[0])

                self.create_predictions_list(class_id, confidence, boxes)
                print("[YOLOv8] Object detection successfully performed on the input image.")
            except Exception as e:
                print("[YOLOv8] Object detection failed with exception: {}".format(e))
                raise Exception("Error performing object detection on the input image.")

            self.visualize_predictions(cv_image)
            return self.predictions

    def visualize_predictions(self, image):
        if not hasattr(self, "predictions") or not hasattr(self, "class_list"):
            print("[YOLOv8] Predictions or class list is not available. Ensure detection is run first.")
            return image

        for pred in self.predictions:
            x1, y1, x2, y2 = map(int, pred["box"])  # Bounding box coordinates
            class_id = int(pred["class_id"])
            confidence = pred["confidence"]

            label = f"{self.class_list[class_id]}: {confidence:.2f}"
            color = (0, 255, 0)  # Green for bounding box

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw the label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(y1, label_size[1] + 10)
            cv2.rectangle(image, (x1, label_y1 - label_size[1] - 10), 
                          (x1 + label_size[0], label_y1 + 3), color, -1)
            cv2.putText(image, label, (x1, label_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        print("[YOLOv8] Visualized predictions on the image.")

        cv2.imshow("Output Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
