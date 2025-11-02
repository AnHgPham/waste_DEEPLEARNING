"""
Baseline CNN model.

This module provides functions to build the baseline CNN architecture.

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import *


def build_baseline_model(input_shape, num_classes):
    """
    Builds a baseline Convolutional Neural Network model.

    The architecture consists of several convolutional blocks followed by a
    dense classifier head. Batch Normalization and Dropout are used for
    regularization.

    Arguments:
    input_shape -- tuple, shape of the input images (height, width, channels).
    num_classes -- int, number of output classes.

    Returns:
    model -- tf.keras.Model, the compiled CNN model.
    """
    model = keras.Sequential(name="Baseline_CNN")
    model.add(layers.Input(shape=input_shape))

    # CRITICAL: Rescale pixel values from [0, 255] to [0, 1]
    # This is essential for baseline CNN training stability
    model.add(layers.Rescaling(1./255))

    # Convolutional Blocks
    # Each block: Conv -> Conv -> Batch Norm -> Max Pooling
    for filters in BASELINE_FILTERS:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classifier Head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(BASELINE_DENSE_UNITS, activation='relu'))
    model.add(layers.Dropout(BASELINE_DROPOUT_RATE))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

