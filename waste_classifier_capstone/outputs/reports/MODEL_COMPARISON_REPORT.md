# Model Comparison Report
## Waste Classification Capstone Project

**Date:** November 2, 2025
**Author:** Capstone Project Team

---

## Executive Summary

This report presents a comprehensive comparison between two deep learning approaches for waste classification:
1. **Baseline CNN** - Custom CNN trained from scratch
2. **MobileNetV2 Transfer Learning** - Pretrained model with fine-tuning

**Key Finding:** Transfer learning with MobileNetV2 achieves **14.31% higher accuracy** than the baseline CNN, demonstrating the power of pretrained models for image classification tasks.

---

## 1. Dataset Overview

- **Total Images:** 19,840
- **Number of Classes:** 10
- **Classes:** battery, biological, cardboard, clothes, glass, metal, paper, plastic, shoes, trash
- **Data Split:**
  - Training: 80% (15,872 images)
  - Validation: 10% (1,984 images)
  - Test: 10% (1,984 images)
- **Image Size:** 224x224x3
- **Batch Size:** 32

---

## 2. Model Architectures

### 2.1 Baseline CNN

**Architecture:**
- 4 Convolutional Blocks (32, 64, 128, 256 filters)
- Each block: Conv2D → Conv2D → BatchNorm → MaxPooling
- Global Average Pooling
- Dense(128) → Dropout(0.5) → Dense(10)
- **Total Parameters:** 1,385,674
- **Trainable Parameters:** 1,385,674

**Training Configuration:**
- Optimizer: Adam
- Learning Rate: 0.001 (initial), reduced to ~0.000003
- Epochs: 30 + 20 (continued training)
- Data Augmentation: Yes (flip, rotation, zoom, contrast, brightness)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### 2.2 MobileNetV2 Transfer Learning

**Architecture:**
- Base: MobileNetV2 (pretrained on ImageNet)
- Classification Head:
  - GlobalAveragePooling2D
  - Dense(256) → BatchNorm → Dropout(0.3)
  - Dense(128) → BatchNorm → Dropout(0.3)
  - Dense(10, softmax)
- **Total Parameters:** 2,651,914
- **Trainable Parameters (Phase 1):** 328,970
- **Trainable Parameters (Phase 2):** 1,297,674

**Training Configuration:**
- **Phase 1 (Feature Extraction):**
  - Base frozen, train classification head only
  - Learning Rate: 0.0001
  - Epochs: 20

- **Phase 2 (Fine-Tuning):**
  - Unfreeze top 30 layers
  - Learning Rate: 0.00001
  - Epochs: 15

---

## 3. Performance Comparison

### 3.1 Overall Metrics

| Metric | Baseline CNN | MobileNetV2 | Improvement |
|--------|--------------|-------------|-------------|
| **Test Accuracy** | 79.59% | **93.90%** | **+14.31%** |
| **Test Loss** | 0.6833 | **0.1867** | **-72.7%** |
| **Top-5 Accuracy** | 97.78% | **99.80%** | **+2.02%** |
| **Validation Accuracy** | 79.51% | 94.00% | +14.49% |
| **Training Time** | ~50 epochs | ~35 epochs | Faster convergence |

### 3.2 Generalization Analysis

| Model | Val Accuracy | Test Accuracy | Gap |
|-------|--------------|---------------|-----|
| **Baseline CNN** | 79.51% | 79.59% | +0.08% |
| **MobileNetV2** | 94.00% | 93.90% | -0.10% |

**Observation:** Both models generalize excellently with minimal val-test gap, indicating no overfitting.

---

## 4. Per-Class Performance Comparison

### 4.1 Accuracy by Class

| Class | Baseline CNN | MobileNetV2 | Improvement |
|-------|--------------|-------------|-------------|
| **battery** | 60.00% | **90.53%** | **+30.53%** |
| **biological** | 64.36% | **96.04%** | **+31.68%** |
| **cardboard** | 81.42% | **89.62%** | **+8.20%** |
| **clothes** | 94.01% | **99.25%** | **+5.24%** |
| **glass** | 80.07% | **91.50%** | **+11.43%** |
| **metal** | 74.76% | **90.29%** | **+15.53%** |
| **paper** | 81.07% | **94.08%** | **+13.01%** |
| **plastic** | 73.87% | **88.94%** | **+15.07%** |
| **shoes** | 75.88% | **98.49%** | **+22.61%** |
| **trash** | 51.58% | **85.26%** | **+33.68%** |

### 4.2 Key Insights

**Baseline CNN - Challenging Classes:**
- **trash (51.58%)** - Generic category, difficult to distinguish
- **battery (60.00%)** - Often confused with metal (10.53%)
- **biological (64.36%)** - Confused with clothes (12.87%)

**MobileNetV2 - Top Performers:**
- **clothes (99.25%)** - Near-perfect classification
- **shoes (98.49%)** - Distinctive shape features
- **biological (96.04%)** - Texture/color differentiation
- **paper (94.08%)** - Clear visual patterns

**MobileNetV2 - Improvement Champions:**
- **trash:** +33.68% (largest improvement)
- **biological:** +31.68%
- **battery:** +30.53%
- **shoes:** +22.61%

---

## 5. Confusion Pattern Analysis

### 5.1 Baseline CNN - Top Confusions

| True Class | Predicted Class | Confusion Rate |
|------------|-----------------|----------------|
| biological | clothes | 12.87% |
| trash | paper | 12.63% |
| trash | glass | 11.58% |
| battery | metal | 10.53% |
| biological | shoes | 8.91% |

### 5.2 MobileNetV2 - Top Confusions

| True Class | Predicted Class | Confusion Rate |
|------------|-----------------|----------------|
| cardboard | paper | 8.20% |
| trash | plastic | 6.32% |
| plastic | glass | 5.03% |
| glass | plastic | 3.59% |
| trash | glass | 3.16% |

**Analysis:**
- MobileNetV2 has **significantly lower confusion rates** overall
- Both models struggle with **paper-based materials** (cardboard/paper)
- **Transparent materials** (plastic/glass) remain challenging for both
- **Trash category** confusion reduced from 12.63% → 6.32%

---

## 6. F1-Score Comparison

### 6.1 Detailed Classification Metrics

#### Baseline CNN

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| battery | 0.8769 | 0.6000 | 0.7125 | 95 |
| biological | 0.9420 | 0.6436 | 0.7647 | 101 |
| cardboard | 0.7968 | 0.8142 | 0.8054 | 183 |
| clothes | 0.8700 | 0.9401 | 0.9037 | 534 |
| glass | 0.8362 | 0.8007 | 0.8180 | 306 |
| metal | 0.6417 | 0.7476 | 0.6906 | 103 |
| paper | 0.7062 | 0.8107 | 0.7548 | 169 |
| plastic | 0.7136 | 0.7387 | 0.7259 | 199 |
| shoes | 0.7156 | 0.7588 | 0.7366 | 199 |
| trash | 0.7903 | 0.5158 | 0.6242 | 95 |
| **Weighted Avg** | **0.8012** | **0.7959** | **0.7935** | **1984** |

#### MobileNetV2

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| battery | 0.9663 | 0.9053 | 0.9348 | 95 |
| biological | 0.9700 | 0.9604 | 0.9652 | 101 |
| cardboard | 0.9591 | 0.8962 | 0.9266 | 183 |
| clothes | 0.9888 | 0.9925 | 0.9907 | 534 |
| glass | 0.9365 | 0.9150 | 0.9256 | 306 |
| metal | 0.8692 | 0.9029 | 0.8857 | 103 |
| paper | 0.8736 | 0.9408 | 0.9060 | 169 |
| plastic | 0.8634 | 0.8894 | 0.8762 | 199 |
| shoes | 0.9655 | 0.9849 | 0.9751 | 199 |
| trash | 0.8804 | 0.8526 | 0.8663 | 95 |
| **Weighted Avg** | **0.9398** | **0.9390** | **0.9391** | **1984** |

---

## 7. Training Characteristics

### 7.1 Convergence Speed

| Model | Epochs to Best Val Acc | Final Epochs | Training Time |
|-------|------------------------|--------------|---------------|
| **Baseline CNN** | ~25 | 50 (30+20) | ~2.5 hours |
| **MobileNetV2** | ~20 (Phase 1) | 35 (20+15) | ~1.5 hours |

**MobileNetV2 advantages:**
- Faster convergence due to pretrained weights
- Lower computational cost for Phase 1 (frozen base)
- Less prone to overfitting

### 7.2 Learning Curve Stability

| Model | Overfitting | Val Loss Fluctuation | Stability |
|-------|-------------|----------------------|-----------|
| **Baseline CNN** | Minimal | Moderate | Good |
| **MobileNetV2** | None | Low | Excellent |

---

## 8. Model Strengths & Weaknesses

### 8.1 Baseline CNN

**Strengths:**
- Simple architecture, easy to understand
- No dependency on pretrained weights
- Good performance on clothes (94.01%)
- Lightweight model (1.4M parameters)

**Weaknesses:**
- Lower overall accuracy (79.59%)
- Struggles with battery (60%), biological (64%), trash (51%)
- Requires more training epochs
- High confusion rates between similar classes

### 8.2 MobileNetV2 Transfer Learning

**Strengths:**
- Excellent overall accuracy (93.90%)
- Consistent performance across all classes (>85%)
- Fast training convergence
- Robust to challenging categories
- Low confusion rates

**Weaknesses:**
- Larger model (2.7M parameters)
- Requires pretrained weights
- Still confuses cardboard/paper (8.20%)
- Slightly more complex architecture

---

## 9. Practical Implications

### 9.1 Deployment Considerations

**Baseline CNN:**
- **Use Cases:** Resource-constrained environments, edge devices
- **Pros:** Small model size, fast inference
- **Cons:** Lower accuracy may require human verification

**MobileNetV2:**
- **Use Cases:** Production applications, high-accuracy requirements
- **Pros:** Superior accuracy, reliable predictions
- **Cons:** Slightly larger model, but still mobile-friendly

### 9.2 Cost-Benefit Analysis

| Aspect | Baseline CNN | MobileNetV2 | Winner |
|--------|--------------|-------------|--------|
| **Accuracy** | 79.59% | 93.90% | MobileNetV2 |
| **Model Size** | 1.4M params | 2.7M params | Baseline |
| **Training Time** | ~2.5 hours | ~1.5 hours | MobileNetV2 |
| **Inference Speed** | Fast | Fast | Tie |
| **Reliability** | Moderate | High | MobileNetV2 |
| **Production Ready** | No | Yes | MobileNetV2 |

---

## 10. Recommendations

### For Production Deployment:
1. **Use MobileNetV2** - Superior accuracy justifies minimal size increase
2. **Monitor trash/plastic/glass** - Higher confusion rates warrant attention
3. **Consider ensemble** - Combine both models for critical applications
4. **Optimize with TFLite** - Reduce model size for mobile deployment

### For Further Improvement:
1. **Collect more data** for challenging classes (battery, trash, metal)
2. **Apply class weighting** to balance performance
3. **Experiment with data augmentation** for confused pairs
4. **Try deeper architectures** (EfficientNet, ResNet50)
5. **Implement active learning** for misclassified samples

---

## 11. Conclusion

The **MobileNetV2 transfer learning approach significantly outperforms the baseline CNN** with:
- **+14.31% absolute improvement** in test accuracy
- **Consistent performance** across all 10 classes
- **Faster training convergence** and better stability
- **Production-ready reliability** with 93.90% accuracy

For waste classification applications, **MobileNetV2 is the clear winner** and recommended model for deployment.

---

## Appendix: Key Metrics Summary

```
BASELINE CNN:
├── Test Accuracy: 79.59%
├── Top-5 Accuracy: 97.78%
├── Weighted F1: 0.7935
└── Best Class: clothes (94.01%)
    Worst Class: trash (51.58%)

MOBILENETV2:
├── Test Accuracy: 93.90%
├── Top-5 Accuracy: 99.80%
├── Weighted F1: 0.9391
└── Best Class: clothes (99.25%)
    Worst Class: trash (85.26%)

IMPROVEMENT:
├── Accuracy: +14.31%
├── F1-Score: +0.1456
└── All classes >85% accuracy ✓
```

---

**Report Generated:** November 2, 2025
**Project:** Waste Classification Capstone
**Models Evaluated:** Baseline CNN, MobileNetV2 Transfer Learning
