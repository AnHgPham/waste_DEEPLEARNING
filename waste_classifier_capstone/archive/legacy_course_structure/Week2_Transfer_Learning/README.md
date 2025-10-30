# Week 2: Transfer Learning with MobileNetV2

## 📚 Mục tiêu học tập

Tuần này bạn sẽ học:
1. **Transfer Learning** - Tái sử dụng pretrained models
2. **MobileNetV2 Architecture** - Efficient CNN cho mobile/edge devices
3. **Two-Phase Training** - Feature extraction → Fine-tuning

---

## 🎯 Lý thuyết Transfer Learning

### 1. Transfer Learning là gì?

**Định nghĩa:** Sử dụng kiến thức (weights) đã học từ task này cho task khác.

**Tại sao hiệu quả?**
- 🎓 Lower layers học **general features** (edges, colors, textures)
- 🎯 Higher layers học **specific features** (cho task cụ thể)
- 💪 Pretrained trên ImageNet (1.4M images, 1000 classes) → có kiến thức rộng

**Khi nào dùng Transfer Learning?**
- ✅ Dataset nhỏ (< 10k images)
- ✅ Task tương tự (cùng domain: computer vision)
- ✅ Cần training nhanh
- ✅ Limited computational resources

### 2. MobileNetV2 Architecture

**Đặc điểm:**
- 📱 Thiết kế cho mobile/edge devices
- ⚡ Efficient: 3.5M parameters (vs ResNet50: 25M)
- 🎯 Accuracy cao: ~72% top-1 on ImageNet
- 🔧 Depthwise Separable Convolutions

#### Inverted Residual Block

**Cấu trúc cơ bản của MobileNetV2:**

```
Input
  ↓
[Expansion] 1×1 Conv (expand channels 6x)
  ↓
[Depthwise] 3×3 Depthwise Conv (spatial filtering)
  ↓
[Projection] 1×1 Conv (reduce channels)
  ↓
[Residual Connection] Add input (if stride=1)
```

**Linear Bottleneck:**
- Không dùng ReLU ở output layer cuối
- Giữ information flow tốt hơn

**Depthwise Separable Convolution:**

Thay vì standard convolution:
```
Standard Conv: C_in × C_out × K × K parameters
```

Tách thành 2 bước:
```
1. Depthwise: C_in × K × K (filter từng channel riêng)
2. Pointwise: C_in × C_out × 1 × 1 (kết hợp channels)
→ Giảm ~8-9x parameters
```

### 3. Two-Phase Training Strategy

#### Phase 1: Feature Extraction (Frozen Base)

```python
base_model.trainable = False  # Freeze all base layers
```

**Mục đích:**
- Train chỉ classification head (top layers)
- Sử dụng features đã học từ ImageNet
- Nhanh, ổn định

**Hyperparameters:**
- Learning rate: 0.001 (cao hơn)
- Epochs: 15
- Freeze: 100% base model

**Expected:** 80-85% accuracy

#### Phase 2: Fine-Tuning (Unfrozen Top Layers)

```python
# Unfreeze top 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True
```

**Mục đích:**
- Adapt features cho waste classification
- Học features specific cho domain
- Improve accuracy

**Hyperparameters:**
- Learning rate: 0.0001 (thấp hơn 10x)
- Epochs: 10
- Unfreeze: Top 30 layers (~20% của base)

**Expected:** 85-92% accuracy

**⚠️ Quan trọng:**
- **Phải train Phase 1 trước** Phase 2
- **Lower learning rate** trong Phase 2 (tránh phá hỏng pretrained weights)
- **Unfreeze từ từ** (không unfreeze tất cả ngay)

---

## 📂 Cấu trúc

```
Week2_Transfer_Learning/
├── README.md                    # File này
├── assignments/                 # Notebooks
│   ├── W2_Feature_Extraction.ipynb
│   └── W2_Fine_Tuning.ipynb
├── transfer_learning.py         # Script (full pipeline)
└── utils/
    └── model_utils.py           # build_transfer_model, unfreeze_layers
```

---

## 🚀 Cách sử dụng

### Python Script (Recommended)

```bash
# Chạy full pipeline (Phase 1 + Phase 2)
python Week2_Transfer_Learning/transfer_learning.py

# Custom epochs
python Week2_Transfer_Learning/transfer_learning.py --phase1-epochs 20 --phase2-epochs 15

# Hoặc dùng main.py
python main.py --week 2
```

### Jupyter Notebooks (Step-by-step)

```bash
cd Week2_Transfer_Learning/assignments
jupyter notebook W2_Feature_Extraction.ipynb
```

---

## 📊 Kiến trúc Model

```
Input (224×224×3)
    ↓
[Preprocessing] MobileNetV2 specific normalization
    ↓
[Base Model: MobileNetV2]
    - 154 layers
    - Pretrained on ImageNet
    - Output: 1280 features
    ↓
[Classification Head]
    GlobalAveragePooling2D
    ↓
    Dense(128, ReLU)
    ↓
    BatchNormalization
    ↓
    Dropout(0.5)
    ↓
    Dense(10, Softmax)
```

**Total parameters:** ~3.5M  
**Trainable (Phase 1):** ~170K (only head)  
**Trainable (Phase 2):** ~1.2M (head + top 30 layers)

---

## 📈 Kết quả mong đợi

| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Val Accuracy | 80-85% | 85-92% |
| Training Time (GPU) | 10-15 min | 8-10 min |
| Parameters Trained | 170K | 1.2M |

**Improvement vs Baseline:**
- 📈 +10-15% accuracy
- ⚡ 3x fewer parameters
- 🚀 Faster convergence

---

## 🔬 So sánh Training Strategies

### Strategy 1: Feature Extraction Only
```python
base_model.trainable = False
```
- ✅ Nhanh nhất
- ✅ Ổn định
- ⚠️ Accuracy trung bình

### Strategy 2: Fine-Tuning All Layers
```python
base_model.trainable = True  # Tất cả
```
- ⚠️ Dễ overfit
- ⚠️ Cần dataset lớn
- ⚠️ Có thể phá hỏng pretrained weights

### Strategy 3: Two-Phase (RECOMMENDED)
```python
# Phase 1: Frozen
# Phase 2: Unfreeze top layers
```
- ✅ Best accuracy
- ✅ Stable training
- ✅ Balance giữa speed và performance

---

## 💡 Tips và Best Practices

### 1. Learning Rate

**Rule of thumb:**
```
Fine-tuning LR = Transfer LR / 10
```

**Tại sao?**
- Pretrained weights đã tốt
- Chỉ cần điều chỉnh nhẹ
- LR cao → phá hỏng features đã học

### 2. Số layers unfreeze

**Guidelines:**
- Small dataset (< 5k): Unfreeze 10-20 layers
- Medium dataset (5-10k): Unfreeze 20-40 layers
- Large dataset (> 10k): Unfreeze 40-80 layers

**Với waste dataset (~20k images):**
→ Unfreeze 30 layers là hợp lý

### 3. Data Augmentation

**Vẫn cần!** Dù đã dùng transfer learning:
- ✅ Horizontal flip
- ✅ Rotation (±10%)
- ✅ Zoom (±10%)
- ✅ Contrast adjustment

### 4. Batch Size

**MobileNetV2 với 224×224:**
- GPU 8GB: batch_size = 32-64
- GPU 16GB: batch_size = 64-128

---

## 🔗 Tài liệu tham khảo

1. **Transfer Learning:**
   - "A Survey on Transfer Learning" - Pan & Yang (2010)
   - CS231n: Transfer Learning lecture

2. **MobileNetV2:**
   - "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
   - Original paper: https://arxiv.org/abs/1801.04381

3. **Fine-tuning:**
   - "How transferable are features in deep neural networks?" (2014)
   - Keras Applications Documentation

---

## 🎓 Kiến thức học được

Sau Week 2, bạn sẽ hiểu:

✅ Transfer learning và khi nào dùng  
✅ MobileNetV2 architecture và depthwise convolutions  
✅ Two-phase training strategy  
✅ Fine-tuning best practices  
✅ Cải thiện 10-15% accuracy so với baseline  

---

**Previous:** [Week 1 - Baseline CNN](../Week1_Data_and_Baseline/README.md)  
**Next:** [Week 3 - Real-time Detection](../Week3_Realtime_Detection/README.md)

