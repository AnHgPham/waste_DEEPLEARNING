"""
Utility functions for object detection with YOLOv8 (Week 3)

This module provides helper functions for:
- Loading the YOLOv8 model.
- Performing object detection on an image.
- Cropping detected objects from the image.

Author: Pham An
"""

import cv2
import numpy as np
from ultralytics import YOLO

# Import from parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def load_yolo_model(model_path=YOLO_MODEL):
    """
    Loads the YOLOv8 model from the specified path.

    Arguments:
    model_path -- str, path to the YOLOv8 model file (e.g., 'yolov8n.pt').

    Returns:
    model -- YOLO, the loaded YOLOv8 model object.
    """
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path)
    print("✅ YOLOv8 model loaded.")
    return model

def detect_objects(model, image, confidence_threshold=YOLO_CONFIDENCE):
    """
    Performs object detection on a single image using the YOLOv8 model.

    Arguments:
    model -- YOLO, the loaded YOLOv8 model.
    image -- np.ndarray, the input image in BGR format.
    confidence_threshold -- float, the minimum confidence for a detection.

    Returns:
    boxes -- list, a list of bounding boxes for detected objects.
             Each box is in the format [x1, y1, x2, y2, confidence, class_id].
    """
    results = model(image, conf=confidence_threshold, verbose=False)
    return results[0].boxes.data.cpu().numpy()

def crop_objects(image, boxes):
    """
    Crops the detected objects from the image based on bounding boxes.

    Arguments:
    image -- np.ndarray, the original image.
    boxes -- list, a list of bounding boxes from `detect_objects`.

    Returns:
    cropped_images -- list, a list of cropped image arrays (np.ndarray).
    """
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cropped_img = image[y1:y2, x1:x2]
        cropped_images.append(cropped_img)
    return cropped_images
