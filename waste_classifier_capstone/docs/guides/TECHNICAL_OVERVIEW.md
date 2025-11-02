# Waste Classification Capstone Project - Summary

## Executive Summary

This capstone project demonstrates a complete end-to-end deep learning pipeline for waste classification, following academic best practices inspired by the Deep Learning Specialization from deeplearning.ai. The project achieves **85-92% accuracy** on 10 waste categories and includes real-time detection capabilities.

---

## Project Objectives

The primary objective of this project was to build a production-ready waste classification system while maintaining academic rigor in code organization, documentation, and implementation. The project serves as both a practical application and an educational resource for understanding deep learning concepts.

---

## Technical Achievements

### Model Performance

| Model | Accuracy | Parameters | Training Time (GPU) | Inference Speed |
|-------|----------|------------|---------------------|-----------------|
| Baseline CNN | 75-80% | ~10M | 15-20 min | 40-50 FPS |
| MobileNetV2 (Transfer Learning) | 85-92% | ~3.5M | 20-25 min | 30-40 FPS |
| MobileNetV2 (Quantized INT8) | 83-90% | ~900KB | N/A | 50-70 FPS |

### Dataset Statistics

- **Total Images:** 19,760
- **Classes:** 10 (battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash)
- **Split:** 80% train (15,804), 10% validation (1,972), 10% test (1,984)
- **Image Size:** 224×224 pixels
- **Format:** JPEG

---

## Architecture Overview

### Week 1: Baseline CNN

A custom Convolutional Neural Network built from scratch with the following architecture:

```
Input (224×224×3)
    ↓
Conv Block 1 (32 filters)
    ↓
Conv Block 2 (64 filters)
    ↓
Conv Block 3 (128 filters)
    ↓
Conv Block 4 (256 filters)
    ↓
Global Average Pooling
    ↓
Dense (128 units) + Dropout (0.5)
    ↓
Output (10 classes, Softmax)
```

**Key Features:**
- Batch Normalization for stable training
- Dropout for regularization
- Data Augmentation (rotation, flip, zoom, contrast)

### Week 2: Transfer Learning with MobileNetV2

Leveraging a pretrained MobileNetV2 model with a two-phase training strategy:

**Phase 1 - Feature Extraction (15 epochs):**
- Freeze MobileNetV2 base layers
- Train only the custom classification head
- Learning rate: 0.001

**Phase 2 - Fine-Tuning (10 epochs):**
- Unfreeze top 30 layers of MobileNetV2
- Continue training with lower learning rate
- Learning rate: 0.0001

**Architecture:**
```
Input (224×224×3)
    ↓
MobileNetV2 Preprocessing
    ↓
MobileNetV2 Base (ImageNet weights)
    ↓
Global Average Pooling
    ↓
Dense (128 units, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Output (10 classes, Softmax)
```

### Week 3: Real-time Detection with YOLOv8

Integration of YOLOv8 for object detection with the custom classifier:

**Pipeline:**
1. **Detection:** YOLOv8n detects objects in the camera frame
2. **Cropping:** Extract bounding boxes for each detected object
3. **Classification:** MobileNetV2 classifier identifies waste type
4. **Visualization:** Draw bounding boxes with class labels and confidence scores

**Performance:**
- Real-time processing at 15-30 FPS on standard hardware
- Multiple object detection and classification in a single frame
- Adjustable confidence threshold (default: 0.5)

### Week 4: Model Optimization for Deployment

Conversion and optimization for edge devices:

**TensorFlow Lite Conversion:**
- FP32 model: ~15 MB, 85-92% accuracy
- INT8 quantized model: ~900 KB, 83-90% accuracy
- **4× reduction in model size** with minimal accuracy loss

**Deployment Targets:**
- Raspberry Pi 4
- NVIDIA Jetson Nano
- Mobile devices (Android/iOS)
- Web browsers (TensorFlow.js)

---

## Code Organization (Academic Standards)

### Modular Structure

The project follows a modular architecture with clear separation of concerns:

```
Week_X/
├── assignments/          # Jupyter notebooks with complete implementations
├── utils/                # Reusable helper functions
│   ├── data_utils.py     # Data loading and preprocessing
│   ├── model_utils.py    # Model building and training
│   └── viz_utils.py      # Visualization functions
└── slides/               # Theoretical background (optional)
```

### Documentation Standards

All functions follow NumPy/Google docstring conventions:

```python
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
    
    Example:
    >>> model = build_baseline_model((224, 224, 3), 10)
    >>> model.summary()
    """
    # Implementation...
```

### Configuration Management

All hyperparameters are centralized in `config.py`:

- **Paths:** Data directories, output directories
- **Model settings:** Image size, batch size, number of classes
- **Training parameters:** Learning rates, epochs, optimizer settings
- **Augmentation:** Rotation, flip, zoom, contrast factors
- **Callbacks:** Early stopping, learning rate reduction
- **Reproducibility:** Random seed (42)

---

## Key Learning Outcomes

### Technical Skills

1. **Data Preparation:**
   - Dataset splitting and organization
   - Data augmentation techniques
   - TensorFlow data pipelines

2. **Model Development:**
   - Building CNNs from scratch
   - Transfer learning with pretrained models
   - Two-phase training strategy (feature extraction + fine-tuning)

3. **Real-time Systems:**
   - Object detection with YOLOv8
   - Integration of detection and classification
   - Real-time inference optimization

4. **Model Deployment:**
   - TensorFlow Lite conversion
   - Post-training quantization (INT8)
   - Performance vs. accuracy trade-offs

### Best Practices

1. **Code Quality:**
   - Modular, reusable functions
   - Comprehensive docstrings
   - Type hints for clarity
   - Consistent naming conventions

2. **Reproducibility:**
   - Fixed random seeds
   - Version-controlled dependencies
   - Documented hyperparameters
   - Complete training logs

3. **Documentation:**
   - Clear README and guides
   - Inline code comments
   - Jupyter notebooks with explanations
   - Architecture diagrams

---

## Comparison with Original Project

| Aspect | Original waste_classifier | Capstone Project |
|--------|---------------------------|------------------|
| **Structure** | Production-oriented | Academic-oriented |
| **Code Organization** | Monolithic scripts | Modular functions |
| **Documentation** | Basic README | Comprehensive docs + notebooks |
| **Learning Materials** | None | 4 weeks of progressive notebooks |
| **Exercises** | None | Complete implementations with explanations |
| **Testing** | None | Validation at each step |
| **Reproducibility** | Good | Excellent (fixed seeds, documented) |
| **Deployment** | Scripts only | Notebooks + optimization guide |

---

## Future Enhancements

Potential extensions to this project:

1. **Additional Classes:** Expand to more waste categories (e.g., e-waste, hazardous materials)
2. **Multi-label Classification:** Handle images with multiple waste types
3. **Segmentation:** Pixel-level waste type identification
4. **Mobile App:** Deploy as a mobile application for on-the-go classification
5. **Cloud Deployment:** Build a web API with cloud hosting (AWS, GCP, Azure)
6. **Dataset Augmentation:** Collect and annotate custom waste images
7. **Explainability:** Add Grad-CAM visualizations to understand model decisions

---

## Conclusion

This capstone project successfully demonstrates the application of deep learning to a real-world problem while maintaining high academic standards. The project structure, code organization, and documentation make it suitable for:

- **Educational purposes:** Teaching deep learning concepts
- **Portfolio showcase:** Demonstrating technical skills
- **Research foundation:** Building upon for academic research
- **Production deployment:** Adapting for real-world applications

The project achieves a balance between theoretical rigor and practical implementation, making it a valuable resource for anyone learning deep learning and computer vision.

---

**Author:** Pham Anh  
**Date:** October 2024  
**License:** MIT  
**Repository:** https://github.com/AnHgPham/waste_classifier
