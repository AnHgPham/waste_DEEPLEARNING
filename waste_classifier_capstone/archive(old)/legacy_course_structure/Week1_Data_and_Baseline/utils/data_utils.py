"""
Utility functions for data preparation and processing (Week 1)

This module provides helper functions for:
- Splitting raw data into train/val/test sets
- Creating TensorFlow data generators
- Applying data augmentation

Author: Pham An
"""

import os
import shutil
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Import from parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import *

def split_data(raw_dir, processed_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    Splits raw image data into train, validation, and test sets.

    Arguments:
    raw_dir -- Path, directory with raw images organized in class subfolders.
    processed_dir -- Path, directory to save the split datasets.
    train_ratio -- float, proportion of data for training.
    val_ratio -- float, proportion of data for validation.
    test_ratio -- float, proportion of data for testing.
    seed -- int, random seed for reproducibility.

    Returns:
    None
    """
    random.seed(seed)
    
    if processed_dir.exists():
        print(f"⚠️ Processed data directory already exists. Skipping split.")
        return

    print(f"Splitting data from {raw_dir} to {processed_dir}...")

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)

        num_images = len(images)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)

        train_images = images[:num_train]
        val_images = images[num_train : num_train + num_val]
        test_images = images[num_train + num_val:]

        # Create directories
        (processed_dir / 'train' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'val' / class_name).mkdir(parents=True, exist_ok=True)
        (processed_dir / 'test' / class_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(img, processed_dir / 'train' / class_name / img.name)
        for img in val_images:
            shutil.copy(img, processed_dir / 'val' / class_name / img.name)
        for img in test_images:
            shutil.copy(img, processed_dir / 'test' / class_name / img.name)

    print("✅ Data splitting complete.")

def create_data_generators(train_dir, val_dir, img_size, batch_size, seed):
    """
    Creates training and validation data generators with augmentation.

    Arguments:
    train_dir -- Path, directory for training data.
    val_dir -- Path, directory for validation data.
    img_size -- tuple, target image size (height, width).
    batch_size -- int, number of samples per batch.
    seed -- int, random seed for reproducibility.

    Returns:
    train_ds -- tf.data.Dataset, training dataset.
    val_ds -- tf.data.Dataset, validation dataset.
    """
    # Data augmentation pipeline
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(AUGMENTATION_CONFIG['rotation_factor']),
            layers.RandomZoom(AUGMENTATION_CONFIG['zoom_factor']),
            layers.RandomContrast(AUGMENTATION_CONFIG['contrast_factor']),
        ],
        name="data_augmentation",
    )

    # Create datasets
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True,
        seed=seed
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False,
        seed=seed
    )

    # Normalization layer
    normalization_layer = layers.Rescaling(1./255)

    # Apply augmentation and normalization
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds
