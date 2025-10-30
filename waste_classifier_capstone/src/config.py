"""
Central Configuration for Waste Classification Capstone Project

This module contains all hyperparameters, paths, and settings used throughout
the project. Centralizing configuration ensures consistency and makes it easy
to experiment with different settings.

Following academic best practices, all magic numbers are defined here with
clear documentation of their purpose and typical ranges.

"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Root directory of the project (go up from src/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"

# Week-specific directories
WEEK1_DIR = PROJECT_ROOT / "Week1_Data_and_Baseline"
WEEK2_DIR = PROJECT_ROOT / "Week2_Transfer_Learning"
WEEK3_DIR = PROJECT_ROOT / "Week3_Realtime_Detection"
WEEK4_DIR = PROJECT_ROOT / "Week4_Deployment"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Dataset source
DATASET_NAME = "sumn2u/garbage-classification-v2"
DATASET_URL = "https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2"

# Data split ratios
# Following standard practice: 80% train, 10% val, 10% test
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Class names (alphabetically sorted for consistency)
CLASS_NAMES = [
    'battery',      # Batteries and power cells
    'biological',   # Organic/food waste
    'cardboard',    # Cardboard boxes and packaging
    'clothes',      # Textile and fabric items
    'glass',        # Glass bottles and containers
    'metal',        # Metal cans and objects
    'paper',        # Paper and documents
    'plastic',      # Plastic bottles and items
    'shoes',        # Footwear
    'trash'         # General waste
]

NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

# Input image size
# MobileNetV2 expects 224x224, but can work with other sizes
# Larger sizes = more detail but slower inference
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

# Baseline CNN architecture
BASELINE_FILTERS = [32, 64, 128, 256]  # Filters per conv block
BASELINE_DENSE_UNITS = 128             # Units in dense layer
BASELINE_DROPOUT_RATE = 0.5            # Dropout rate

# Transfer learning architecture
TRANSFER_BASE_MODEL = 'MobileNetV2'    # Pretrained model
TRANSFER_WEIGHTS = 'imagenet'          # Pretrained weights
TRANSFER_DENSE_UNITS = 256             # Units in classification head (increased)
TRANSFER_DROPOUT_RATE = 0.3            # Dropout rate (reduced for better learning)
TRANSFER_POOLING = 'avg'               # Global pooling type

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Batch size
# Larger batch = more stable gradients but more memory
# Typical values: 16, 32, 64
BATCH_SIZE = 32

# Number of epochs
EPOCHS_BASELINE = 30                   # Baseline CNN training epochs
EPOCHS_TRANSFER_PHASE1 = 20            # Feature extraction epochs (increased)
EPOCHS_TRANSFER_PHASE2 = 15            # Fine-tuning epochs (increased)

# Learning rates
# Following best practices for transfer learning:
# - Higher LR for training from scratch
# - Lower LR for fine-tuning pretrained models
LEARNING_RATE_BASELINE = 1e-3          # 0.001
LEARNING_RATE_TRANSFER_PHASE1 = 1e-4   # 0.0001 (feature extraction - lower for stability)
LEARNING_RATE_TRANSFER_PHASE2 = 1e-5   # 0.00001 (fine-tuning - very low)

# Optimizer
OPTIMIZER = 'adam'                     # Adam optimizer
BETA_1 = 0.9                          # Adam beta_1
BETA_2 = 0.999                        # Adam beta_2
EPSILON = 1e-7                        # Adam epsilon

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics
METRICS = ['accuracy', 'top_k_categorical_accuracy']

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

# Enable/disable augmentation
USE_AUGMENTATION = True

# Augmentation parameters
# These values are empirically chosen to be realistic for waste images
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,           # Mirror flip
    'rotation_factor': 0.2,            # ±20% rotation (±72 degrees)
    'zoom_factor': 0.2,                # ±20% zoom
    'contrast_factor': 0.2,            # ±20% contrast
    'brightness_factor': 0.1,          # ±10% brightness (enabled for robustness)
    'width_shift_factor': 0.1,         # ±10% horizontal shift (enabled)
    'height_shift_factor': 0.1,        # ±10% vertical shift (enabled)
}

# =============================================================================
# CALLBACKS
# =============================================================================

# Early stopping
# Stop training if val_loss doesn't improve for N epochs
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MODE = 'min'
EARLY_STOPPING_RESTORE_BEST = True

# Reduce learning rate on plateau
# Reduce LR if val_loss doesn't improve for N epochs
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5                 # Multiply LR by this factor
REDUCE_LR_MONITOR = 'val_loss'
REDUCE_LR_MODE = 'min'
REDUCE_LR_MIN_LR = 1e-7                # Minimum learning rate

# Model checkpoint
# Save best model based on val_loss
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_MODE = 'min'
CHECKPOINT_SAVE_BEST_ONLY = True

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

# Random seed for reproducibility
# Setting this ensures consistent results across runs
RANDOM_SEED = 42

# TensorFlow determinism
# Note: Some operations are non-deterministic even with seed
TF_DETERMINISTIC_OPS = True

# =============================================================================
# REAL-TIME DETECTION (YOLOv8)
# =============================================================================

# YOLOv8 model
YOLO_MODEL = 'yolov8n.pt'              # Nano model (fastest)
YOLO_CONFIDENCE = 0.5                  # Detection confidence threshold
YOLO_IOU_THRESHOLD = 0.45              # NMS IoU threshold

# Camera settings
CAMERA_INDEX = 0                       # Default camera
CAMERA_WIDTH = 640                     # Camera resolution width
CAMERA_HEIGHT = 480                    # Camera resolution height
CAMERA_FPS = 30                        # Target FPS

# Display settings
DISPLAY_WIDTH = 1280                   # Display window width
DISPLAY_HEIGHT = 720                   # Display window height
FONT_SCALE = 0.6                       # Text font scale
FONT_THICKNESS = 2                     # Text thickness
BOX_THICKNESS = 2                      # Bounding box thickness

# =============================================================================
# DEPLOYMENT
# =============================================================================

# Model optimization
QUANTIZATION_TYPE = 'int8'             # int8, float16, or None
PRUNING_SPARSITY = 0.5                 # Target sparsity for pruning

# API settings
API_HOST = '0.0.0.0'
API_PORT = 8000
API_WORKERS = 4

# Edge deployment
EDGE_DEVICE = 'raspberry_pi'           # raspberry_pi, jetson_nano, etc.
EDGE_MODEL_FORMAT = 'tflite'           # tflite, onnx, etc.

# =============================================================================
# LOGGING AND VISUALIZATION
# =============================================================================

# Logging level
LOG_LEVEL = 'INFO'                     # DEBUG, INFO, WARNING, ERROR

# Visualization
PLOT_STYLE = 'seaborn'                 # Matplotlib style
FIGURE_DPI = 100                       # Figure resolution
SAVE_FORMAT = 'png'                    # Image save format

# Confusion matrix
CM_FIGSIZE = (12, 10)                  # Figure size
CM_CMAP = 'Blues'                      # Color map
CM_NORMALIZE = True                    # Normalize values

# Training history
HISTORY_FIGSIZE = (14, 5)              # Figure size
HISTORY_METRICS = ['loss', 'accuracy'] # Metrics to plot

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_directories():
    """
    Create all necessary directories if they don't exist.
    
    This function should be called at the start of any script that writes
    to the outputs directory.
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        TRAIN_DIR, VAL_DIR, TEST_DIR,
        OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR, SCREENSHOTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_model_path(model_name, phase=None):
    """
    Get the path for saving/loading a model.
    
    Arguments:
    model_name -- string, name of the model (e.g., 'baseline', 'mobilenetv2')
    phase -- string or None, training phase (e.g., 'phase1', 'phase2', 'final')
    
    Returns:
    path -- Path object, full path to model file
    
    Example:
    >>> get_model_path('mobilenetv2', 'final')
    PosixPath('/home/user/waste_classifier_capstone/outputs/models/mobilenetv2_final.keras')
    """
    if phase:
        filename = f"{model_name}_{phase}.keras"
    else:
        filename = f"{model_name}.keras"
    
    return MODELS_DIR / filename

def get_report_path(model_name, report_type):
    """
    Get the path for saving a report.
    
    Arguments:
    model_name -- string, name of the model
    report_type -- string, type of report (e.g., 'confusion_matrix', 'classification_report')
    
    Returns:
    path -- Path object, full path to report file
    
    Example:
    >>> get_report_path('mobilenetv2', 'confusion_matrix')
    PosixPath('/home/user/waste_classifier_capstone/outputs/reports/mobilenetv2_confusion_matrix.png')
    """
    extension = 'png' if 'matrix' in report_type else 'txt'
    filename = f"{model_name}_{report_type}.{extension}"
    return REPORTS_DIR / filename

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """
    Print a summary of the current configuration.
    
    Useful for debugging and ensuring correct settings before training.
    """
    print("=" * 70)
    print("WASTE CLASSIFICATION - CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"\n📁 PROJECT ROOT: {PROJECT_ROOT}")
    print(f"\n📊 DATASET:")
    print(f"   - Classes: {NUM_CLASSES}")
    print(f"   - Split: {TRAIN_RATIO:.0%} train / {VAL_RATIO:.0%} val / {TEST_RATIO:.0%} test")
    print(f"\n🏗️  MODEL:")
    print(f"   - Input size: {IMG_SIZE}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"\n🎓 TRAINING:")
    print(f"   - Baseline epochs: {EPOCHS_BASELINE}")
    print(f"   - Transfer phase 1: {EPOCHS_TRANSFER_PHASE1} epochs @ LR={LEARNING_RATE_TRANSFER_PHASE1}")
    print(f"   - Transfer phase 2: {EPOCHS_TRANSFER_PHASE2} epochs @ LR={LEARNING_RATE_TRANSFER_PHASE2}")
    print(f"\n🔄 AUGMENTATION: {'Enabled' if USE_AUGMENTATION else 'Disabled'}")
    print(f"\n🎯 RANDOM SEED: {RANDOM_SEED}")
    print("=" * 70)

if __name__ == "__main__":
    # Test configuration
    create_directories()
    print_config()
