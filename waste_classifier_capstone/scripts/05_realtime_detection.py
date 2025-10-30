"""
Week 3 - Real-time Detection Script

This script implements real-time waste detection and classification using:
- YOLOv8 for object detection
- Trained classifier for waste type classification

Usage:
    python scripts/week3_realtime_detection.py
    python scripts/week3_realtime_detection.py --model mobilenetv2 --camera 0
    python scripts/week3_realtime_detection.py --video path/to/video.mp4

"""

import os
import sys
import argparse
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.detection import load_yolo_model, detect_objects, crop_objects, classify_images, draw_results

def process_frame(frame, yolo_model, classifier_model):
    """
    Process a single frame: detect objects and classify them.

    Arguments:
    frame -- np.ndarray, the input frame.
    yolo_model -- YOLO, the YOLOv8 model.
    classifier_model -- tf.keras.Model, the waste classifier.

    Returns:
    annotated_frame -- np.ndarray, the frame with annotations.
    num_detections -- int, number of objects detected.
    """
    # Detect objects
    boxes = detect_objects(yolo_model, frame, confidence_threshold=YOLO_CONFIDENCE)
    
    if len(boxes) == 0:
        return frame, 0
    
    # Crop detected objects
    cropped_images = crop_objects(frame, boxes)
    
    # Classify cropped images
    predictions = classify_images(classifier_model, cropped_images)
    
    # Draw results on frame
    annotated_frame = draw_results(frame, boxes, predictions)
    
    return annotated_frame, len(boxes)

def run_realtime_detection(args):
    """Main function for real-time detection."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - REAL-TIME DETECTION")
    print("=" * 70)
    
    # Load classifier model
    model_name = args.model
    model_path = get_model_path(model_name, 'final')
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Classifier model not found: {model_path}")
        print(f"\n   Please train the {model_name} model first:")
        if model_name == 'baseline':
            print("   python scripts/week1_baseline_training.py")
        else:
            print("   python scripts/week2_transfer_learning.py")
        return
    
    print(f"\n📦 Loading classifier model: {model_name}")
    classifier_model = tf.keras.models.load_model(model_path)
    print(f"   ✅ Classifier loaded from {model_path}")
    
    # Load YOLO model
    print(f"\n📦 Loading YOLOv8 detection model...")
    yolo_model = load_yolo_model(YOLO_MODEL)
    
    # Setup video source
    if args.video:
        print(f"\n🎥 Opening video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_name = "Video File"
    else:
        print(f"\n📷 Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        source_name = f"Camera {args.camera}"
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video source")
        return
    
    print(f"   ✅ {source_name} opened successfully")
    
    # Create screenshots directory
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Main loop
    print("\n" + "=" * 70)
    print("🚀 Starting real-time detection...")
    print("=" * 70)
    print("\nControls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save screenshot")
    print("   - Press 'p' to pause/resume")
    print("\n")
    
    frame_count = 0
    detection_count = 0
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠️ End of video stream")
                    break
                
                frame_count += 1
                
                # Process frame
                annotated_frame, num_detections = process_frame(frame, yolo_model, classifier_model)
                detection_count += num_detections
                
                # Add FPS and info
                fps_text = f"Frame: {frame_count} | Detections: {num_detections}"
                cv2.putText(annotated_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Waste Detection - Press Q to quit, S to save, P to pause', annotated_frame)
            else:
                # Show paused message
                paused_frame = annotated_frame.copy()
                cv2.putText(paused_frame, "PAUSED - Press P to resume", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Waste Detection - Press Q to quit, S to save, P to pause', paused_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⏹️  Stopping detection...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = SCREENSHOTS_DIR / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(screenshot_path), annotated_frame)
                print(f"📸 Screenshot saved: {screenshot_path}")
            elif key == ord('p'):
                paused = not paused
                status = "⏸️  PAUSED" if paused else "▶️  RESUMED"
                print(status)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("✅ Real-time detection session complete!")
        print("=" * 70)
        print(f"\n📊 Session Statistics:")
        print(f"   - Total frames processed: {frame_count}")
        print(f"   - Total objects detected: {detection_count}")
        print(f"   - Average detections per frame: {detection_count/max(frame_count, 1):.2f}")
        print(f"\n📁 Screenshots saved to: {SCREENSHOTS_DIR}")

def main():
    parser = argparse.ArgumentParser(description='Real-time Waste Detection')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Classifier model to use (default: mobilenetv2)')
    parser.add_argument('--camera', type=int, default=CAMERA_INDEX,
                        help=f'Camera index (default: {CAMERA_INDEX})')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file (if not using camera)')
    args = parser.parse_args()
    
    run_realtime_detection(args)

if __name__ == "__main__":
    main()

