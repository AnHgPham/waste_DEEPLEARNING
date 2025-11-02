# 🏗️ BASELINE CNN CHI TIẾT

**Thời gian:** 1 giờ
**Mục tiêu:** Hiểu code Baseline CNN trong dự án và cách nó hoạt động

---

## 📌 1. BASELINE CNN LÀ GÌ? (5 phút)

### **Định nghĩa:**

**Baseline CNN = Model đơn giản để làm baseline (điểm chuẩn)**

```
Mục đích:
1. ✅ Tạo baseline để so sánh với models khác
2. ✅ Học từ đầu (train from scratch) trên waste data
3. ✅ Không dùng pretrained weights
4. ✅ Kiểm tra dataset có đủ tốt không

Kết quả:
- Train Acc: 81.28%
- Val Acc: 79.59%
- Test Acc: 79.51%
→ BASELINE để so sánh!
```

### **Trong dự án này:**

```python
# File: src/models/baseline.py
def build_baseline_model(input_shape, num_classes):
    # Build model từ đầu
    # Input: (224, 224, 3)
    # Output: (10,) - 10 waste classes
```

---

## 🏛️ 2. ARCHITECTURE CHI TIẾT (20 phút)

### **A. Tổng Quan Architecture**

```
Input Image (224x224x3)
    ↓
[Rescaling Layer]        # [0, 255] → [0, 1]
    ↓
[Conv Block 1]           # 32 filters
    ↓
[Conv Block 2]           # 64 filters
    ↓
[Conv Block 3]           # 128 filters
    ↓
[Conv Block 4]           # 256 filters
    ↓
[GlobalAveragePooling2D] # Flatten
    ↓
[Dense 128]              # Classification head
    ↓
[Dropout 0.5]            # Regularization
    ↓
[Dense 10]               # Output
    ↓
Softmax → [10 probabilities]
```

---

### **B. Layer-by-Layer Breakdown**

#### **Layer 0: Rescaling (Chuẩn hóa)**

```python
model.add(layers.Rescaling(1./255))
```

**Mục đích:** Chuyển pixel values từ [0, 255] → [0, 1]

**Tại sao?**
```python
# TRƯỚC rescaling:
pixel = 255  # White pixel
→ Neural network nhận input LỚN (255)
→ Weights cần LỚN để học
→ Training KHÔNG ổn định!

# SAU rescaling:
pixel = 255 / 255 = 1.0  # White pixel
→ Neural network nhận input NHỎ (0-1)
→ Training ổn định hơn
→ Gradients không explode!
```

**Ví dụ:**
```python
Original Image:
[255, 128, 0]  # Red-ish pixel

After Rescaling:
[1.0, 0.5, 0.0]  # Same red, normalized
```

---

#### **Convolutional Blocks (4 blocks)**

**Cấu trúc MỖI block:**

```python
# Pseudo-code cho 1 block với N filters:
Conv2D(N, 3x3, ReLU, padding='same')  # First conv
Conv2D(N, 3x3, ReLU, padding='same')  # Second conv
BatchNormalization()                   # Normalize
MaxPooling2D(2x2)                      # Downsample
```

**Code thực tế:**

```python
# From config.py:
BASELINE_FILTERS = [32, 64, 128, 256]

# From baseline.py:
for filters in BASELINE_FILTERS:
    model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
```

**Chi tiết từng block:**

##### **Block 1: 32 filters**

```
Input: [224, 224, 3]

Conv2D(32, 3x3, ReLU, same) → [224, 224, 32]
  ↓ Học 32 filters (edge detectors, color detectors)
Conv2D(32, 3x3, ReLU, same) → [224, 224, 32]
  ↓ Refine features
BatchNormalization() → [224, 224, 32]
  ↓ Normalize để training ổn định
MaxPooling2D(2x2) → [112, 112, 32]
  ↓ Giảm kích thước xuống 1/2
```

**Filters học gì?**
```
Filter 1: Vertical edges   (phát hiện cạnh dọc)
Filter 2: Horizontal edges (phát hiện cạnh ngang)
Filter 3: Red color        (phát hiện màu đỏ)
Filter 4: Blue color       (phát hiện màu xanh)
...
Filter 32: Complex patterns
```

---

##### **Block 2: 64 filters**

```
Input: [112, 112, 32]

Conv2D(64, 3x3, ReLU, same) → [112, 112, 64]
  ↓ Học 64 patterns phức tạp hơn
Conv2D(64, 3x3, ReLU, same) → [112, 112, 64]
  ↓ Refine
BatchNormalization() → [112, 112, 64]
  ↓ Normalize
MaxPooling2D(2x2) → [56, 56, 64]
  ↓ Downsample
```

**Filters học gì?**
```
Combine low-level features từ Block 1:
- Texture patterns (nhám, mịn)
- Simple shapes (circles, rectangles)
- Color combinations (plastic transparent)
```

---

##### **Block 3: 128 filters**

```
Input: [56, 56, 64]

Conv2D(128, 3x3, ReLU, same) → [56, 56, 128]
Conv2D(128, 3x3, ReLU, same) → [56, 56, 128]
BatchNormalization() → [56, 56, 128]
MaxPooling2D(2x2) → [28, 28, 128]
```

**Filters học gì?**
```
Mid-level features:
- Object parts (bottle cap, bottle body)
- Material textures (metal shine, glass clarity)
- Complex patterns
```

---

##### **Block 4: 256 filters**

```
Input: [28, 28, 128]

Conv2D(256, 3x3, ReLU, same) → [28, 28, 256]
Conv2D(256, 3x3, ReLU, same) → [28, 28, 256]
BatchNormalization() → [28, 28, 256]
MaxPooling2D(2x2) → [14, 14, 256]
```

**Filters học gì?**
```
High-level features:
- Whole objects (bottle, can, box)
- Semantic concepts (plastic-ness, metal-ness)
- Class-specific patterns
```

---

#### **GlobalAveragePooling2D**

```python
model.add(layers.GlobalAveragePooling2D())
```

**Mục đích:** Chuyển 3D tensor → 1D vector

```
Input: [14, 14, 256]  # Feature maps từ Block 4

Process:
  For each of 256 channels:
    Take average of all 14x14 pixels
    → 1 number per channel

Output: [256,]  # 1D vector
```

**Ví dụ:**

```python
# Channel 0 (ví dụ: "plastic detector"):
channel_0 = [
  [0.1, 0.2, ..., 0.3],  # 14x14 grid
  [0.4, 0.5, ..., 0.1],
  ...
]

average_0 = mean(channel_0) = 0.25
→ "Plastic confidence = 0.25"

# Repeat cho 256 channels
→ Output: [0.25, 0.82, 0.15, ..., 0.91]  # 256 numbers
```

**Tại sao dùng GAP?**
```
✅ Giảm params (không cần Dense layer lớn)
✅ Spatial invariance (object ở đâu cũng được)
✅ Prevent overfitting
```

---

#### **Classification Head**

```python
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
```

**Dense 128:**

```
Input: [256,]  # From GAP

Dense(128, ReLU):
  output = ReLU(W @ input + b)
  → [128,]

Mục đích: Kết hợp features để classify
```

**Dropout 0.5:**

```python
Dropout(0.5)
# Randomly set 50% neurons to 0 during training

Example:
Before: [0.5, 0.8, 0.3, 0.9, ...]
After:  [0.5, 0.0, 0.3, 0.0, ...]  # Random 50% dropped
```

**Tại sao?**
```
✅ Prevent overfitting
✅ Force network to learn redundant representations
✅ Improve generalization
```

**Chỉ dùng khi training!**
```python
# Training mode:
model.fit(...) → Dropout ACTIVE

# Inference mode:
model.predict(...) → Dropout OFF (all neurons used)
```

**Dense 10 (Output):**

```
Input: [128,]

Dense(10, Softmax):
  logits = W @ input + b  → [10,]  # Raw scores
  probs = Softmax(logits) → [10,]  # Probabilities (sum=1)

Output:
[0.02, 0.01, 0.05, 0.02, 0.03, 0.01, 0.02, 0.82, 0.01, 0.01]
  ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑     ↑
 bat   bio   card cloth glass metal paper plas  shoes trash

Prediction: "plastic" (index 7, prob=0.82)
```

---

### **C. Model Summary**

```
Total Parameters: ~1.4M

Breakdown:
- Conv layers: ~1.2M params (majority)
- Dense layers: ~200K params
- BatchNorm: ~2K params

Receptive Field: ~61x61 pixels (27% of image)
```

**Compare với MobileNetV2:**

```
Baseline CNN:
  Params: 1.4M
  Depth: 8 conv layers
  Receptive Field: 61x61 (27%)
  Accuracy: 79.59%

MobileNetV2:
  Params: 2.7M
  Depth: 53 layers
  Receptive Field: 150x150 (70%)
  Accuracy: 93.90% (+14.31%!)
```

---

## 💻 3. CODE WALKTHROUGH (15 phút)

### **File: src/models/baseline.py**

```python
def build_baseline_model(input_shape, num_classes):
    """
    Build Baseline CNN.

    Arguments:
    input_shape: (224, 224, 3)
    num_classes: 10 (waste classes)

    Returns:
    model: Compiled Keras model
    """

    # 1. Create Sequential model
    model = keras.Sequential(name="Baseline_CNN")
    model.add(layers.Input(shape=input_shape))

    # 2. CRITICAL: Rescale [0,255] → [0,1]
    model.add(layers.Rescaling(1./255))

    # 3. Convolutional Blocks
    for filters in BASELINE_FILTERS:  # [32, 64, 128, 256]
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # 4. Classifier Head
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(BASELINE_DENSE_UNITS, activation='relu'))  # 128
    model.add(layers.Dropout(BASELINE_DROPOUT_RATE))  # 0.5
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
```

**Giải thích từng bước:**

**Bước 1: Create model**
```python
model = keras.Sequential(name="Baseline_CNN")
```
- Sequential = Layers xếp tuần tự
- Name = "Baseline_CNN" (để debug dễ)

**Bước 2: Rescaling**
```python
model.add(layers.Rescaling(1./255))
```
- CRITICAL! Không có layer này → Training sẽ fail
- Pixel [0, 255] → [0, 1]

**Bước 3: Conv Blocks**
```python
for filters in [32, 64, 128, 256]:
    # 2x Conv + BN + MaxPool
```
- 4 blocks = 8 conv layers
- Filters tăng dần (32→256)
- Feature maps giảm dần (224→14)

**Bước 4: Classification**
```python
GlobalAveragePooling2D()  # [14,14,256] → [256]
Dense(128, ReLU)          # [256] → [128]
Dropout(0.5)              # Regularization
Dense(10, Softmax)        # [128] → [10]
```

---

### **File: scripts/03_baseline_training.py**

**Training process:**

```python
# 1. Load Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# 2. Build Model
model = build_baseline_model(
    input_shape=INPUT_SHAPE,  # (224, 224, 3)
    num_classes=NUM_CLASSES    # 10
)

# 3. Compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_BASELINE),  # 0.001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5),
    ModelCheckpoint('baseline.keras', save_best_only=True)
]

# 5. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.2%}")
```

---

## 📊 4. TRAINING PROCESS (10 phút)

### **A. Training History**

```
Epoch 1-5: FAST IMPROVEMENT
  Epoch 1:  Val Acc = 70.12%
  Epoch 5:  Val Acc = 76.34%

  Model đang học:
  ✓ Basic edges, colors
  ✓ Simple shapes

Epoch 6-15: MODERATE IMPROVEMENT
  Epoch 10: Val Acc = 77.89%
  Epoch 15: Val Acc = 78.56%

  Model đang học:
  ✓ Textures (plastic smooth, metal shiny)
  ✓ Mid-level patterns

Epoch 16-25: SLOW IMPROVEMENT
  Epoch 20: Val Acc = 79.12%
  Epoch 25: Val Acc = 79.41%

  Model struggling:
  ⚠ High-level features khó học
  ⚠ Model capacity gần đạt ceiling

Epoch 26-30: PLATEAU
  Epoch 26: Val Acc = 79.51%
  Epoch 30: Val Acc = 79.59%

  MODEL CEILING REACHED!
  → Không cải thiện thêm được
```

### **B. Learning Rate Schedule**

```
Initial LR: 0.001

ReduceLROnPlateau (patience=3, factor=0.5):
  Epoch 10: LR → 0.0005  (val_loss không giảm 3 epochs)
  Epoch 18: LR → 0.00025
  Epoch 25: LR → 0.000125

Final LR: 0.000125
```

**Tại sao reduce LR?**
```
High LR (0.001):
  → Large weight updates
  → Fast learning
  → Coarse optimization

Low LR (0.0001):
  → Small weight updates
  → Slow learning
  → Fine-tuning
```

---

## 📈 5. KẾT QUẢ VÀ PHÂN TÍCH (10 phút)

### **A. Final Results**

```
Train Accuracy:      81.28%
Validation Accuracy: 79.59%
Test Accuracy:       79.51%

Gap (Train - Val):   1.69%  ← Small gap = Good generalization!
```

**Interpretation:**
```
✅ Model generalize tốt (gap nhỏ)
✅ Không overfitting nghiêm trọng
⚠  Accuracy không cao (chỉ ~80%)
→ Cần model tốt hơn!
```

---

### **B. Per-Class Performance**

```
Easy Classes (>85%):
  ✓ clothes:    94.10%  (Distinct texture & shape)
  ✓ shoes:      89.90%  (Unique appearance)

Medium Classes (75-85%):
  ⚠ paper:      81.70%  (Confused with cardboard)
  ⚠ plastic:    78.30%  (Confused with glass)
  ⚠ metal:      76.20%  (Confused with foil)

Hard Classes (<75%):
  ✗ trash:      52.11%  (No clear pattern!)
  ✗ glass:      74.50%  (Confused with plastic)
```

**Pattern:**
```
Baseline handles DISTINCT classes well
  → clothes, shoes có appearance khác biệt

Baseline struggles with SIMILAR classes
  → plastic vs glass (both transparent)
  → paper vs cardboard (similar texture)
  → trash (general waste, no pattern)
```

---

## 🔍 6. TẠI SAO CHỈ ĐẠT 79.59%? (15 phút)

### **A. Model Capacity Limitation**

**1. Receptive Field Quá Nhỏ**

```
Baseline Receptive Field: 61x61 pixels (27% of image)

Plastic Bottle trong ảnh:
┌──────────────────────┐
│                      │
│    [Plastic Bottle]  │
│    ┌──────┐          │
│    │ Cap  │          │
│    │      │          │
│    │Label │          │
│    │      │          │
│    └──────┘          │
│                      │
└──────────────────────┘

Baseline chỉ nhìn:
┌───┐
│Cap│  ← Chỉ thấy 1 phần nhỏ!
└───┘

→ Không thấy TOÀN BỘ object
→ Khó classify đúng!
```

**MobileNetV2 nhìn thấy:**
```
Receptive Field: 150x150 pixels (70% of image)

┌──────────────────┐
│ [Bottle]         │
│ ┌──────┐         │
│ │ Cap  │         │
│ │ Body │         │
│ │Label │         │
│ └──────┘         │
└──────────────────┘

→ Thấy FULL object!
→ Classify tốt hơn!
```

---

**2. Depth Không Đủ**

```
Baseline: 8 conv layers
  → Chỉ học được 3 levels of abstraction

  Level 1: Edges, colors
  Level 2: Textures, patterns
  Level 3: Simple shapes

  ✗ KHÔNG HỌC ĐƯỢC complex high-level features!

MobileNetV2: 53 layers
  → Học được 6-7 levels

  Level 1: Edges
  Level 2: Textures
  Level 3: Object parts
  Level 4: Whole objects
  Level 5: Object relationships
  Level 6: Semantic concepts

  ✅ Học được complex patterns!
```

---

**3. Parameters Không Đủ**

```
Baseline: 1.4M parameters
  → Có thể học ~1.4M patterns
  → Với 15,777 training images
  → Và 224x224x3 high-dimensional data
  → KHÔNG ĐỦ capacity!

MobileNetV2: 2.7M parameters (but pretrained on ImageNet!)
  → Đã học 1.2M ImageNet images
  → Transfer knowledge to waste classification
  → Đủ capacity cho complex task!
```

---

### **B. Continue Training Thí Nghiệm**

**Câu hỏi:** Nếu train thêm 20 epochs nữa, accuracy có tăng không?

**Kết quả:**

```
Epoch 31: Val Acc = 79.41%
Epoch 32: Val Acc = 79.12%  ← Giảm!
Epoch 33: Val Acc = 78.90%  ← Giảm thêm!
...
Epoch 36: Val Acc = 78.90%

EarlyStopping triggered (patience=5)
Training stopped!
```

**Tại sao GIẢM?**

```
1. Model Capacity Exhausted:
   → Model đã học HẾT những gì nó có thể
   → Architecture quá đơn giản
   → Không thể improve thêm

2. Learning Rate Quá Thấp:
   → LR = 0.000003 (very small!)
   → Updates quá nhỏ
   → Không giúp gì

3. Overfitting:
   → Train Acc tăng (81.28% → 81.70%)
   → Val Acc giảm (79.59% → 78.90%)
   → Gap tăng (1.69% → 2.80%)
   → Model đang MEMORIZE training data!
```

**Kết luận:**
```
❌ KHÔNG phải do thiếu epochs!
❌ KHÔNG phải do learning rate!
✅ ĐÂY LÀ ARCHITECTURE LIMITATION!

Solution: Cần model DEEPER, WIDER
→ Transfer Learning (MobileNetV2)!
```

---

### **C. Visualization: Loss Landscape**

```
Loss (Baseline stuck here)
 ↑
 │     ╱╲
 │    ╱  ╲
 │   ╱    ╲________  ← Local minimum (79.5%)
 │  ╱
 └──────────────────→ Epochs

Loss (MobileNetV2 reaches here)
 ↑
 │              ╱╲
 │             ╱  ╲
 │____________╱    ╲__  ← Global minimum (93.9%)
 │
 └──────────────────→ Epochs
```

**Baseline bị stuck vì:**
- ✗ Architecture constraints
- ✗ Limited receptive field
- ✗ Shallow depth
- ✗ Cannot escape local minimum

---

## 🎓 TỔNG KẾT

### **Baseline CNN Characteristics:**

**Strengths (Điểm mạnh):**
```
✅ Simple, dễ hiểu
✅ Train nhanh (~30 mins trên GPU)
✅ Good baseline (79.59%)
✅ Không overfitting nghiêm trọng
✅ Generalize tốt (train-val gap nhỏ)
```

**Weaknesses (Điểm yếu):**
```
❌ Accuracy không cao (79.59%)
❌ Receptive field nhỏ (61x61)
❌ Depth không đủ (8 layers)
❌ Struggle với similar classes
❌ Không học được complex patterns
❌ Model ceiling ở ~80%
```

### **Key Takeaways:**

1. **Baseline CNN đạt 79.59%** - Acceptable cho baseline!
2. **Model Ceiling** - Architecture limitation, không phải training issue
3. **Receptive Field matters** - 61x61 quá nhỏ để nhìn toàn bộ object
4. **Depth matters** - 8 layers không đủ học complex features
5. **Solution** → Transfer Learning (MobileNetV2: 93.90%)

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] Baseline CNN có 4 conv blocks, 8 conv layers
- [ ] Rescaling layer chuyển [0,255] → [0,1]
- [ ] GlobalAveragePooling thay cho Flatten
- [ ] Dropout 0.5 để prevent overfitting
- [ ] Model đạt 79.59% accuracy
- [ ] Ceiling ở ~80% do architecture limitation
- [ ] Receptive field 61x61 quá nhỏ
- [ ] Continue training KHÔNG giúp (capacity exhausted)

**Nếu OK →** Tiếp tục `04_Transfer_Learning_Chi_Tiet.md` 🚀

---

## 📝 BÀI TẬP TỰ KIỂM TRA

### **Câu 1:** Tại sao cần Rescaling layer?

<details>
<summary>Đáp án</summary>

Rescaling chuyển pixel [0,255] → [0,1] để:
- Training ổn định hơn (input nhỏ)
- Gradients không explode
- Weights dễ học hơn
</details>

### **Câu 2:** Baseline có bao nhiêu parameters?

<details>
<summary>Đáp án</summary>

~1.4M parameters
- Conv layers: ~1.2M
- Dense layers: ~200K
- BatchNorm: ~2K
</details>

### **Câu 3:** Tại sao continue training làm accuracy GIẢM?

<details>
<summary>Đáp án</summary>

Vì:
1. Model capacity exhausted (architecture quá đơn giản)
2. LR quá thấp (0.000003)
3. Overfitting (memorize training data)
→ KHÔNG phải training issue, là architecture limitation!
</details>
