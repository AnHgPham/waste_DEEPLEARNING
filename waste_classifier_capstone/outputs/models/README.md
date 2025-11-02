# Pre-trained Models

This directory contains pre-trained models for waste classification. These models are stored using **Git LFS** so you can download them automatically when cloning the repository.

## 📦 Available Models

### 🌟 Recommended: MobileNetV2 (Production)

| Model | File | Size | Accuracy | Use Case |
|-------|------|------|----------|----------|
| **MobileNetV2 (Full)** | `mobilenetv2_final.keras` | 25.02 MB | 93.90% | Python inference, retraining |
| **MobileNetV2 (TFLite FP32)** | `mobilenetv2_fp32.tflite` | 9.84 MB | 93.90% | Mobile/Web deployment |
| **MobileNetV2 (TFLite INT8)** | `mobilenetv2_int8.tflite` | 2.94 MB | 93.20% | Edge devices (Raspberry Pi) |

### 📊 Baseline CNN (Comparison)

| Model | File | Size | Accuracy | Use Case |
|-------|------|------|----------|----------|
| **Baseline CNN** | `baseline_final.keras` | 5.3 MB | 79.59% | Comparison, education |

---

## 🚀 Quick Start

### Option 1: Git Clone (Git LFS - Automatic)

If you clone with Git LFS installed, models download automatically:

```bash
git clone https://github.com/YOUR_USERNAME/waste_classifier_capstone.git
cd waste_classifier_capstone
# Models are already in outputs/models/
```

### Option 2: Download Script (Manual)

If Git LFS is not available, use the download script:

```bash
# Download recommended model
python scripts/download_models.py --model mobilenetv2

# Download all models
python scripts/download_models.py --model all
```

---

## 📝 Model Details

### MobileNetV2 Final Model
- **File:** `mobilenetv2_final.keras`
- **Architecture:** MobileNetV2 (ImageNet pretrained) + Custom head
- **Training:** Two-phase (feature extraction + fine-tuning)
- **Performance:**
  - Test Accuracy: 93.90%
  - Val Accuracy: 94.00%
  - Top-5 Accuracy: 99.80%
  - All classes: >85% accuracy
- **Input:** 224x224x3 RGB images
- **Output:** 10 waste classes (softmax)
- **Preprocessing:** Built-in MobileNetV2 preprocessing ([-1, 1] normalization)

### MobileNetV2 TFLite FP32
- **File:** `mobilenetv2_fp32.tflite`
- **Optimization:** FP32 quantization
- **Size Reduction:** 60.7% smaller than Keras model
- **Accuracy:** 93.90% (NO LOSS!)
- **Use Case:** Mobile apps, web deployment

### MobileNetV2 TFLite INT8
- **File:** `mobilenetv2_int8.tflite`
- **Optimization:** Full INT8 quantization
- **Size Reduction:** 88.3% smaller than Keras model
- **Accuracy:** 93.20% (only 0.71% loss)
- **Use Case:** Edge devices (Raspberry Pi, IoT)

### Baseline CNN
- **File:** `baseline_final.keras`
- **Architecture:** Custom 4-block CNN
- **Parameters:** 1.4M
- **Performance:**
  - Test Accuracy: 79.59%
  - Val Accuracy: 79.51%
- **Purpose:** Comparison baseline

---

## 🎯 Usage Examples

### Image Classification
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('outputs/models/mobilenetv2_final.keras')

# Load and preprocess image
img = Image.open('waste_image.jpg').resize((224, 224))
img_array = np.array(img).reshape(1, 224, 224, 3)

# Predict
predictions = model.predict(img_array)
class_idx = np.argmax(predictions[0])
confidence = predictions[0][class_idx]

print(f"Predicted class: {CLASS_NAMES[class_idx]}")
print(f"Confidence: {confidence:.2%}")
```

### Using Scripts
```bash
# Image detection
python scripts/08_image_detection.py --image test.jpg --model mobilenetv2

# Real-time detection
python scripts/05_realtime_detection.py --model mobilenetv2

# Evaluation
python scripts/99_evaluate_model.py --model mobilenetv2
```

---

## 🏷️ Waste Classes

The models classify waste into 10 categories:

1. **battery** - Batteries and electronic waste
2. **biological** - Organic/compostable waste
3. **cardboard** - Cardboard packaging
4. **clothes** - Textile waste
5. **glass** - Glass bottles and containers
6. **metal** - Metal cans and objects
7. **paper** - Paper waste
8. **plastic** - Plastic bottles and containers
9. **shoes** - Footwear
10. **trash** - General waste

---

## 📊 Performance Summary

| Metric | MobileNetV2 | Baseline CNN |
|--------|-------------|--------------|
| **Test Accuracy** | 93.90% | 79.59% |
| **Top-5 Accuracy** | 99.80% | 97.78% |
| **F1-Score** | 0.9391 | 0.7935 |
| **Inference Time (CPU)** | ~50ms | ~30ms |
| **Model Size** | 25.02 MB | 5.3 MB |

---

## 🔧 Retraining

To retrain or fine-tune the models:

```bash
# Train from scratch
python scripts/04_transfer_learning.py

# Continue training
python scripts/07_continue_baseline_training.py
```

---

## 📄 License & Citation

If you use these pre-trained models, please cite:

```
Waste Classification using Transfer Learning
Model: MobileNetV2 (ImageNet pretrained)
Dataset: Garbage Classification v2 (Kaggle)
Accuracy: 93.90% on 10-class waste classification
```

---

## 🙏 Acknowledgments

- **Base Model:** MobileNetV2 (Sandler et al., 2018)
- **Dataset:** [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
- **Framework:** TensorFlow 2.x / Keras

---

**Generated:** November 2, 2025
**Version:** 1.0
**Status:** Production Ready ✅
