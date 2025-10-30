"""
Utility functions for building transfer learning models (Week 2)

This module provides helper functions for:
- Building a model using a pretrained base (e.g., MobileNetV2).
- Adding a custom classification head.
- Freezing and unfreezing layers for fine-tuning.

Author: Pham An
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import from parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def build_transfer_model(input_shape, num_classes, freeze_base=True):
    """
    Builds a transfer learning model using MobileNetV2 as the base.

    Arguments:
    input_shape -- tuple, shape of the input images.
    num_classes -- int, number of output classes.
    freeze_base -- bool, whether to freeze the base model layers.

    Returns:
    model -- tf.keras.Model, the compiled transfer learning model.
    """
    # 1. Load the pretrained base model
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Do not include the original classifier
        weights=TRANSFER_WEIGHTS
    )

    # 2. Freeze the base model layers
    base_model.trainable = not freeze_base

    # 3. Create the new model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing layer for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False) # Always run in inference mode
    
    # 4. Add the classification head
    x = layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)
    x = layers.Dense(TRANSFER_DENSE_UNITS, activation='relu', name="Dense_1")(x)
    x = layers.BatchNormalization(name="BatchNorm")(x)
    x = layers.Dropout(TRANSFER_DROPOUT_RATE, name="Dropout")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="Classifier")(x)
    
    # 5. Build the final model
    model = keras.Model(inputs, outputs, name="MobileNetV2_Transfer_Learning")
    
    return model

def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreezes the top N layers of the base model for fine-tuning.

    Arguments:
    model -- tf.keras.Model, the model to modify.
    num_layers_to_unfreeze -- int, the number of layers to unfreeze from the top.

    Returns:
    model -- tf.keras.Model, the modified model.
    """
    base_model = model.get_layer('MobileNetV2')
    base_model.trainable = True

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the top N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
        
    print(f"Unfroze {num_layers_to_unfreeze} layers from the base model.")
    return model
