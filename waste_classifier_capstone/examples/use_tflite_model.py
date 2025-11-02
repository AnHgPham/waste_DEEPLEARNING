"""
Example: How to use TFLite optimized models

This script demonstrates how to use TFLite models for inference.
Suitable for: Raspberry Pi, edge devices, production deployment.

Usage:
    python examples/use_tflite_model.py --image test.jpg
    python examples/use_tflite_model.py --image test.jpg --model int8

"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


def load_tflite_model(model_path):
    """
    Load TFLite model and return interpreter.

    Arguments:
    model_path -- Path to .tflite file

    Returns:
    interpreter -- TFLite interpreter
    input_details -- Input tensor details
    output_details -- Output tensor details
    """
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"[OK] TFLite model loaded: {model_path.name}")
    print(f"   Input shape: {input_details[0]['shape']}")
    print(f"   Input type: {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output type: {output_details[0]['dtype']}")

    return interpreter, input_details, output_details


def preprocess_image(image_path, input_details):
    """
    Load and preprocess image for TFLite model.

    Arguments:
    image_path -- Path to image file
    input_details -- Input tensor details from interpreter

    Returns:
    input_data -- Preprocessed image ready for inference
    """
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Get input shape and type
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Resize to model input size
    image = image.resize((input_shape[1], input_shape[2]))

    # Convert to numpy array
    image_array = np.array(image)

    # Check if model is quantized (INT8)
    is_quantized = (input_dtype == np.uint8)

    if is_quantized:
        # For INT8 models: keep as uint8 in [0, 255]
        input_data = image_array.astype(np.uint8)
    else:
        # For FP32 models: keep as float32 in [0, 255]
        # MobileNetV2 preprocessing is built into the model
        input_data = image_array.astype(np.float32)

    # Add batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    return input_data


def run_inference(interpreter, input_details, output_details, input_data):
    """
    Run inference on TFLite model.

    Arguments:
    interpreter -- TFLite interpreter
    input_details -- Input tensor details
    output_details -- Output tensor details
    input_data -- Preprocessed input data

    Returns:
    predictions -- Model output (class probabilities)
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize if needed (INT8 models)
    output_dtype = output_details[0]['dtype']
    if output_dtype == np.uint8:
        # Dequantize: convert INT8 back to float
        scale, zero_point = output_details[0]['quantization']
        output = scale * (output.astype(np.float32) - zero_point)

    return output[0]  # Remove batch dimension


def classify_image(model_path, image_path):
    """
    Classify a single image using TFLite model.

    Arguments:
    model_path -- Path to TFLite model
    image_path -- Path to image file

    Returns:
    class_name -- Predicted class name
    confidence -- Confidence score
    """
    import time

    # Load model
    interpreter, input_details, output_details = load_tflite_model(model_path)

    # Preprocess image
    print(f"\n[LOADING] Loading image: {image_path}")
    input_data = preprocess_image(image_path, input_details)

    # Run inference
    print(f"[INFERENCE] Running inference...")
    start_time = time.time()
    predictions = run_inference(interpreter, input_details, output_details, input_data)
    inference_time = (time.time() - start_time) * 1000  # Convert to ms

    # Get top prediction
    class_idx = np.argmax(predictions)
    confidence = predictions[class_idx]
    class_name = CLASS_NAMES[class_idx]

    # Get top 3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]

    # Print results
    print(f"\n{'='*70}")
    print(f"[RESULTS] Classification Results")
    print(f"{'='*70}")
    print(f"\n   Top Prediction:")
    print(f"      Class: {class_name.upper()}")
    print(f"      Confidence: {confidence:.2%}")
    print(f"\n   Top 3 Predictions:")
    for i, idx in enumerate(top3_indices, 1):
        print(f"      {i}. {CLASS_NAMES[idx]}: {predictions[idx]:.2%}")

    print(f"\n   Inference Time: {inference_time:.2f} ms")
    print(f"   Model: {model_path.name}")

    return class_name, confidence


def compare_models(image_path):
    """
    Compare inference results and speed between FP32 and INT8 models.

    Arguments:
    image_path -- Path to test image
    """
    import time

    print(f"\n{'='*70}")
    print(f"[COMPARISON] Comparing FP32 vs INT8 Models")
    print(f"{'='*70}")

    models = {
        'FP32': MODELS_DIR / 'mobilenetv2_fp32.tflite',
        'INT8': MODELS_DIR / 'mobilenetv2_int8.tflite'
    }

    results = {}

    for name, model_path in models.items():
        if not model_path.exists():
            print(f"\n[SKIP] {name}: Model not found")
            continue

        print(f"\n[TESTING] {name} Model")
        print(f"-" * 70)

        # Load model
        interpreter, input_details, output_details = load_tflite_model(model_path)
        input_data = preprocess_image(image_path, input_details)

        # Warm up (first inference is slower)
        run_inference(interpreter, input_details, output_details, input_data)

        # Benchmark (run 10 times)
        times = []
        for _ in range(10):
            start = time.time()
            predictions = run_inference(interpreter, input_details, output_details, input_data)
            times.append((time.time() - start) * 1000)

        avg_time = np.mean(times)
        class_idx = np.argmax(predictions)
        confidence = predictions[class_idx]

        results[name] = {
            'class': CLASS_NAMES[class_idx],
            'confidence': confidence,
            'avg_time': avg_time,
            'size': model_path.stat().st_size / (1024 * 1024)  # MB
        }

        print(f"   Prediction: {CLASS_NAMES[class_idx]} ({confidence:.2%})")
        print(f"   Avg inference time: {avg_time:.2f} ms")
        print(f"   Model size: {results[name]['size']:.2f} MB")

    # Summary
    print(f"\n{'='*70}")
    print(f"[SUMMARY] Comparison Results")
    print(f"{'='*70}")
    print(f"\n{'Model':<10} {'Size (MB)':<12} {'Speed (ms)':<12} {'Prediction':<15} {'Confidence'}")
    print(f"-" * 70)

    for name, res in results.items():
        print(f"{name:<10} {res['size']:>8.2f}    {res['avg_time']:>8.2f}    {res['class']:<15} {res['confidence']:>8.2%}")

    if len(results) == 2:
        speedup = results['FP32']['avg_time'] / results['INT8']['avg_time']
        size_reduction = (1 - results['INT8']['size'] / results['FP32']['size']) * 100
        print(f"\n[STATS]")
        print(f"   INT8 is {speedup:.2f}x faster than FP32")
        print(f"   INT8 is {size_reduction:.1f}% smaller than FP32")


def main():
    parser = argparse.ArgumentParser(description='Use TFLite Optimized Models')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model', type=str, default='fp32',
                        choices=['fp32', 'int8'],
                        help='Model type (default: fp32)')
    parser.add_argument('--compare', action='store_true',
                        help='Compare FP32 vs INT8 performance')
    args = parser.parse_args()

    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    if args.compare:
        # Compare both models
        compare_models(image_path)
    else:
        # Use single model
        if args.model == 'fp32':
            model_path = MODELS_DIR / 'mobilenetv2_fp32.tflite'
        else:
            model_path = MODELS_DIR / 'mobilenetv2_int8.tflite'

        if not model_path.exists():
            print(f"[ERROR] Model not found: {model_path}")
            print(f"   Run optimization first: python scripts/06_model_optimization.py")
            return

        classify_image(model_path, image_path)


if __name__ == "__main__":
    main()
