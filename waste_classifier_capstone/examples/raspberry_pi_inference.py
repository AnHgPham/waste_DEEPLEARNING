"""
Raspberry Pi Deployment Example

This script shows how to use INT8 TFLite model on Raspberry Pi
for real-time waste classification.

Requirements:
    pip install tensorflow-lite
    pip install pillow
    pip install picamera2  # For Pi Camera

Usage:
    # Single image
    python examples/raspberry_pi_inference.py --image waste.jpg

    # Camera stream (if you have Pi Camera)
    python examples/raspberry_pi_inference.py --camera

"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path

# Check if running on Raspberry Pi
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE_RUNTIME = True
    print("[INFO] Using TFLite Runtime (optimized for Raspberry Pi)")
except ImportError:
    import tensorflow as tf
    USE_TFLITE_RUNTIME = False
    print("[INFO] Using TensorFlow (standard)")

from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CLASS_NAMES, MODELS_DIR


class WasteClassifier:
    """
    Optimized waste classifier for Raspberry Pi using INT8 quantized model.
    """

    def __init__(self, model_path):
        """Initialize the classifier with TFLite model."""
        # Load TFLite model
        if USE_TFLITE_RUNTIME:
            self.interpreter = tflite.Interpreter(model_path=str(model_path))
        else:
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        self.interpreter.allocate_tensors()

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Cache input properties
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        self.is_quantized = (self.input_dtype == np.uint8)

        print(f"[OK] Model loaded: {Path(model_path).name}")
        print(f"   Input shape: {self.input_shape}")
        print(f"   Quantized: {self.is_quantized}")
        print(f"   Expected inference time: ~15-30ms on Pi 4")

    def preprocess(self, image):
        """Preprocess image for model input."""
        # Resize
        img_resized = image.resize((self.input_shape[1], self.input_shape[2]))

        # Convert to numpy
        img_array = np.array(img_resized)

        # Prepare for model
        if self.is_quantized:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = img_array.astype(np.float32)

        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    def predict(self, image):
        """
        Classify waste image.

        Arguments:
        image -- PIL Image

        Returns:
        class_name -- Predicted class
        confidence -- Confidence score (0-1)
        inference_time -- Time in milliseconds
        """
        # Preprocess
        input_data = self.preprocess(image)

        # Set input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Dequantize if needed
        if self.is_quantized:
            scale, zero_point = self.output_details[0]['quantization']
            output = scale * (output.astype(np.float32) - zero_point)

        # Get prediction
        class_idx = np.argmax(output)
        confidence = output[class_idx]
        class_name = CLASS_NAMES[class_idx]

        return class_name, confidence, inference_time


def classify_image(model_path, image_path):
    """Classify a single image."""
    # Load classifier
    classifier = WasteClassifier(model_path)

    # Load image
    print(f"\n[LOADING] Image: {image_path}")
    image = Image.open(image_path).convert('RGB')

    # Classify
    print(f"[INFERENCE] Running...")
    class_name, confidence, inference_time = classifier.predict(image)

    # Results
    print(f"\n{'='*60}")
    print(f"[RESULTS]")
    print(f"{'='*60}")
    print(f"   Waste Type: {class_name.upper()}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Inference Time: {inference_time:.2f} ms")
    print(f"\n   [OK] Classification complete!")

    return class_name, confidence


def camera_stream(model_path):
    """
    Real-time classification from Pi Camera.

    Note: Requires picamera2 library and Pi Camera module.
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        print("[ERROR] picamera2 not installed!")
        print("   Install: sudo apt install -y python3-picamera2")
        return

    # Initialize camera
    print("[CAMERA] Initializing Pi Camera...")
    camera = Picamera2()
    camera.configure(camera.create_preview_configuration(main={"size": (640, 480)}))
    camera.start()
    time.sleep(2)  # Warm up

    # Load classifier
    classifier = WasteClassifier(model_path)

    print("\n[START] Camera stream started!")
    print("   Press Ctrl+C to stop")
    print("   Classifying every 2 seconds...\n")

    frame_count = 0

    try:
        while True:
            # Capture frame
            frame = camera.capture_array()

            # Convert to PIL Image
            image = Image.fromarray(frame).convert('RGB')

            # Classify
            class_name, confidence, inference_time = classifier.predict(image)

            # Display results
            frame_count += 1
            print(f"[Frame {frame_count}] {class_name.upper()} ({confidence:.1%}) - {inference_time:.1f}ms")

            # Wait before next classification
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n[STOP] Camera stream stopped")
        camera.stop()


def benchmark(model_path, num_iterations=100):
    """
    Benchmark model performance on Raspberry Pi.

    Arguments:
    model_path -- Path to TFLite model
    num_iterations -- Number of test runs
    """
    print(f"\n[BENCHMARK] Running {num_iterations} iterations...")

    # Load classifier
    classifier = WasteClassifier(model_path)

    # Create dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')

    # Warm up (first inference is slower)
    classifier.predict(dummy_image)

    # Benchmark
    times = []
    for i in range(num_iterations):
        _, _, inference_time = classifier.predict(dummy_image)
        times.append(inference_time)

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_iterations}")

    # Statistics
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_time = np.std(times)

    print(f"\n{'='*60}")
    print(f"[RESULTS] Benchmark Results")
    print(f"{'='*60}")
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Min: {min_time:.2f} ms")
    print(f"   Max: {max_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Throughput: {1000/avg_time:.1f} FPS")

    # Model info
    model_size = Path(model_path).stat().st_size / (1024 * 1024)
    print(f"\n   Model: {Path(model_path).name}")
    print(f"   Size: {model_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Waste Classification')
    parser.add_argument('--image', type=str,
                        help='Path to input image')
    parser.add_argument('--camera', action='store_true',
                        help='Use Pi Camera for real-time classification')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run performance benchmark')
    parser.add_argument('--model', type=str, default='int8',
                        choices=['int8', 'fp32'],
                        help='Model type (default: int8 - recommended for Pi)')
    args = parser.parse_args()

    # Get model path
    if args.model == 'int8':
        model_path = MODELS_DIR / 'mobilenetv2_int8.tflite'
    else:
        model_path = MODELS_DIR / 'mobilenetv2_fp32.tflite'

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(f"   Download models: python scripts/download_models.py")
        return

    # Run requested operation
    if args.benchmark:
        benchmark(model_path)
    elif args.camera:
        camera_stream(model_path)
    elif args.image:
        classify_image(model_path, args.image)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
