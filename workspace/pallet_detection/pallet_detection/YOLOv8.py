import os
from ultralytics import YOLO

class YOLOv8:
    def __init__(self):
        pass

    def build_model(self, model_dir_path, weight_file_name):
        try:
            model_path = os.path.join(model_dir_path, weight_file_name)
            self.model = YOLO(model_path)
            self.logger.info("[YOLOv8] Model successfully loaded from: {}".format(model_path))
        except Exception as e:
            self.logger.error("[YOLOv8] Failed to load model with exception: {}".format(e))
            raise Exception("Error loading the given model from path: {}.".format(model_path) +
                            " Make sure the file exists and the format is correct.")

    def load_classes(self, model_dir_path):
        self.class_list = []
        fpath = os.path.join(model_dir_path, "classes.txt")

        try:
            with open(fpath, "r") as f:
                self.class_list = [cname.strip() for cname in f.readlines()]
                self.logger.info("[YOLOv8] Loaded classes from {}".format(fpath))
        except FileNotFoundError:
            self.logger.error("[YOLOv8] Classes file not found at path: {}".format(fpath))
            raise FileNotFoundError("Classes file not found. Make sure the file exists at the specified path.")
        except Exception as e:
            self.logger.error("[YOLOv8] Error loading classes with exception: {}".format(e))
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
            self.logger.warning("[YOLOv8] Input image is None. No predictions will be generated.")
            return None, None
        else:
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

                self().create_predictions_list(class_id, confidence, boxes)
                self.logger.debug("[YOLOv8] Object detection successfully performed on the input image.")
            except Exception as e:
                self.logger.error("[YOLOv8] Object detection failed with exception: {}".format(e))
                raise Exception("Error performing object detection on the input image.")

            return self.predictions
