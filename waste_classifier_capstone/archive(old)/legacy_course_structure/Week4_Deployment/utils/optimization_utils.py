"""
Utility functions for model optimization (Week 4)

This module provides helper functions for:
- Converting a Keras model to TensorFlow Lite (TFLite).
- Applying post-training quantization.
- Evaluating the performance of TFLite models.

Author: Pham An
"""

import tensorflow as tf
import numpy as np

# Import from parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def convert_to_tflite(model, tflite_model_path):
    """
    Converts a Keras model to a TensorFlow Lite model.

    Arguments:
    model -- tf.keras.Model, the trained Keras model.
    tflite_model_path -- Path, the path to save the TFLite model.

    Returns:
    None
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ Model converted to TFLite and saved at {tflite_model_path}")

def quantize_model(model, tflite_quant_model_path, representative_dataset):
    """
    Applies post-training INT8 quantization to a Keras model.

    Arguments:
    model -- tf.keras.Model, the trained Keras model.
    tflite_quant_model_path -- Path, path to save the quantized TFLite model.
    representative_dataset -- tf.data.Dataset, a dataset for calibration.

    Returns:
    None
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for input_value, _ in representative_dataset.batch(1).take(100):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs that support this)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()

    with open(tflite_quant_model_path, 'wb') as f:
        f.write(tflite_model_quant)
    print(f"✅ Model quantized and saved at {tflite_quant_model_path}")

def evaluate_tflite_model(tflite_model_path, test_dataset):
    """
    Evaluates the accuracy of a TFLite model on a test dataset.

    Arguments:
    tflite_model_path -- Path, path to the TFLite model file.
    test_dataset -- tf.data.Dataset, the dataset for evaluation.

    Returns:
    accuracy -- float, the accuracy of the TFLite model.
    """
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    accurate_count = 0
    total_count = 0

    for image, label in test_dataset.unbatch():
        total_count += 1
        # Check if the model is quantized
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            image = (image / input_scale) + input_zero_point
            image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
        else:
             image = np.expand_dims(image, axis=0)

        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details["index"])[0]
        predicted_label = np.argmax(output)
        true_label = np.argmax(label)

        if predicted_label == true_label:
            accurate_count += 1

    accuracy = accurate_count / total_count
    return accuracy
