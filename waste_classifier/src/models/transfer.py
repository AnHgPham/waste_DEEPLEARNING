"""
Transfer learning model using MobileNetV2.

This module provides functions for:
- Building a model using a pretrained base (MobileNetV2)
- Adding a custom classification head
- Freezing and unfreezing layers for fine-tuning

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import *


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
    # Input should be in range [0, 255], will be normalized to [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model - use training mode appropriately for BatchNorm
    # When freeze_base=True, use training=False for frozen BN stats
    # When freeze_base=False (fine-tuning), use training=True to update BN
    x = base_model(x, training=not freeze_base)
    
    # 4. Add the classification head
    # Using a deeper classification head for better feature learning
    x = layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)
    
    # First dense layer
    x = layers.Dense(TRANSFER_DENSE_UNITS, activation='relu', name="Dense_1")(x)
    x = layers.BatchNormalization(name="BatchNorm_1")(x)
    x = layers.Dropout(TRANSFER_DROPOUT_RATE, name="Dropout_1")(x)
    
    # Second dense layer for more capacity
    x = layers.Dense(TRANSFER_DENSE_UNITS // 2, activation='relu', name="Dense_2")(x)
    x = layers.BatchNormalization(name="BatchNorm_2")(x)
    x = layers.Dropout(TRANSFER_DROPOUT_RATE, name="Dropout_2")(x)
    
    # Output layer
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
    # Get base model (name is 'mobilenetv2_1.00_224' not 'MobileNetV2')
    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True

    # Freeze all layers first
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the top N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
        
    print(f"Unfroze {num_layers_to_unfreeze} layers from the base model.")
    return model

