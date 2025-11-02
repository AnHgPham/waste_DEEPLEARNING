# 🎯 TRANSFER LEARNING CHI TIẾT

**Thời gian:** 1.5 giờ
**Mục tiêu:** Hiểu Transfer Learning và tại sao nó tốt hơn Baseline

---

## 📌 1. TRANSFER LEARNING LÀ GÌ? (15 phút)

### **Định nghĩa đơn giản:**

**Transfer Learning = Dùng lại kiến thức đã học từ task khác**

```
Ví dụ thực tế:

Bạn học lái xe hơi:
  ✓ Đã biết giao thông (từ đi xe máy)
  ✓ Đã biết luật đường (từ học lý thuyết)
  → Chỉ cần học KỸ NĂNG MỚI: điều khiển xe hơi
  → HỌC NHANH HƠN người chưa biết gì!

Transfer Learning:
  ✓ Model đã biết nhận dạng ảnh (từ ImageNet)
  ✓ Đã biết detect edges, textures (pre-trained)
  → Chỉ cần học KỸ NĂNG MỚI: phân loại rác
  → ACCURACY CAO HƠN model train from scratch!
```

### **Trong dự án:**

```
Baseline CNN (Train from Scratch):
  - Bắt đầu từ 0 (random weights)
  - Học TẤT CẢ từ waste data (15,777 images)
  - Kết quả: 79.59%

MobileNetV2 (Transfer Learning):
  - Bắt đầu từ pretrained weights (ImageNet)
  - Đã học 1.2M images (1000 classes)
  - Chỉ cần adapt cho waste data
  - Kết quả: 93.90% (+14.31%!) ✅
```

---

## 🤔 2. TẠI SAO CẦN TRANSFER LEARNING? (20 phút)

### **A. Data Limitation**

**Problem:**

```
Waste Classification Dataset:
  Train: 15,777 images
  Val:   1,972 images
  Test:  1,974 images
  Total: 19,723 images
  Classes: 10

→ KHÔNG ĐỦ để train deep CNN from scratch!
```

**Tại sao không đủ?**

```
Deep CNN (như MobileNetV2):
  - 53 layers
  - 2.7M parameters
  - Cần học complex patterns

Rule of thumb:
  Parameters × 10 = Minimum data needed
  2.7M × 10 = 27M images needed!

Waste data chỉ có:
  19,723 images << 27M images

→ Train from scratch = OVERFITTING!
```

**Minh họa:**

```
Train from Scratch với ít data:
Epoch 1:  Train=65%, Val=60%  ✓ Learning
Epoch 10: Train=85%, Val=70%  ⚠ Gap tăng
Epoch 20: Train=95%, Val=65%  ✗ OVERFITTING!
          ↑                ↑
     Memorizing        Not generalizing

Transfer Learning với ít data:
Epoch 1:  Train=85%, Val=83%  ✓ Already good!
Epoch 10: Train=94%, Val=93%  ✓ Learning well
Epoch 20: Train=95%, Val=94%  ✅ EXCELLENT!
          ↑                ↑
     Small gap         Generalizing
```

---

### **B. Feature Reusability**

**Key Insight:**

```
Low-level features (edges, textures) là UNIVERSAL!
→ Giống nhau across different datasets!
```

**Ví dụ:**

```
ImageNet (1000 classes):
  - Cats, dogs, cars, trees, ...
  - Features learned:
    Layer 1: Edges (vertical, horizontal)
    Layer 2: Textures (fur, metal, wood)
    Layer 3: Shapes (circles, rectangles)
    Layer 4: Object parts (wheels, eyes)

Waste Classification (10 classes):
  - Plastic, glass, metal, ...
  - Features needed:
    Layer 1: Edges ✅ SAME as ImageNet!
    Layer 2: Textures ✅ SAME as ImageNet!
    Layer 3: Shapes ✅ SAME as ImageNet!
    Layer 4: Object parts ⚠ Different, need fine-tuning

→ Dùng lại Layer 1-3 từ ImageNet!
→ Chỉ cần học lại Layer 4 cho waste!
```

**Visualize:**

```
ImageNet Features (Transferable):
┌──────────────────────────────────┐
│ Layer 1: Edges                   │ ← REUSE
│ Layer 2: Textures                │ ← REUSE
│ Layer 3: Basic Shapes            │ ← REUSE
│ Layer 4: ImageNet-specific parts │ ← FINE-TUNE
│ Layer 5: ImageNet classes (1000) │ ← REPLACE
└──────────────────────────────────┘

Waste Classifier (Transfer):
┌──────────────────────────────────┐
│ Layer 1: Edges (from ImageNet)   │ ✓ Frozen
│ Layer 2: Textures (from ImageNet)│ ✓ Frozen
│ Layer 3: Shapes (from ImageNet)  │ ✓ Frozen
│ Layer 4: Waste-specific patterns │ ✓ Fine-tuned
│ Layer 5: Waste classes (10)      │ ✓ Trained
└──────────────────────────────────┘
```

---

### **C. Training Time**

```
Baseline CNN (Train from Scratch):
  - 30 epochs
  - ~2 mins/epoch
  - Total: ~60 minutes
  - Result: 79.59%

MobileNetV2 (Transfer Learning):
  Phase 1 (Feature Extraction):
    - 20 epochs
    - ~1.5 mins/epoch
    - Subtotal: ~30 mins

  Phase 2 (Fine-Tuning):
    - 15 epochs
    - ~2 mins/epoch
    - Subtotal: ~30 mins

  Total: ~60 minutes
  Result: 93.90%

→ CÙNG THỜI GIAN, nhưng ACCURACY CAO HƠN 14.31%!
```

---

## 🏗️ 3. IMAGENET PRE-TRAINING (15 phút)

### **A. ImageNet Dataset**

```
ImageNet ILSVRC:
  - 1.2 million training images
  - 1,000 classes
  - Classes: animals, vehicles, objects, ...

Examples:
  - Class 1: Persian cat
  - Class 2: Golden retriever
  - Class 281: Tabby cat
  - Class 817: Sports car
  - ...
  - Class 1000: Toilet paper
```

**Tại sao ImageNet quan trọng?**

```
✅ LARGE-SCALE: 1.2M images >> 19K waste images
✅ DIVERSE: 1000 classes → Rich features
✅ HIGH-QUALITY: Human-annotated labels
✅ STANDARD: Industry benchmark
```

---

### **B. Pre-trained Weights**

**MobileNetV2 trained on ImageNet:**

```
Training process:
  1. Random initialize weights
  2. Train 100+ epochs on ImageNet
  3. Achieve ~72% Top-1 accuracy (on 1000 classes!)
  4. Save weights → "imagenet weights"

Learned features:
  Early layers: Generic (edges, textures)
  Middle layers: Mid-level (object parts)
  Late layers: Specific (ImageNet classes)
```

**Downloading pretrained weights:**

```python
# Keras automatically downloads ImageNet weights
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # Exclude ImageNet classifier
    weights='imagenet'  # Load pretrained weights
)

# Weights downloaded from:
# https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/...
# Size: ~14 MB
```

---

### **C. Feature Hierarchy**

```
MobileNetV2 Pretrained on ImageNet:

Layer Group 1 (Early Layers):
  ┌─────────────────────┐
  │ Edges, Colors       │ ← GENERIC (transferable)
  │ - Vertical edges    │
  │ - Horizontal edges  │
  │ - Diagonal edges    │
  │ - Color blobs       │
  └─────────────────────┘

Layer Group 2 (Middle Layers):
  ┌─────────────────────┐
  │ Textures, Patterns  │ ← SEMI-GENERIC (transferable)
  │ - Fur texture       │
  │ - Metal shine       │
  │ - Wood grain        │
  │ - Glass clarity     │
  └─────────────────────┘

Layer Group 3 (Late Layers):
  ┌─────────────────────┐
  │ Object Parts        │ ← SEMI-SPECIFIC (fine-tune)
  │ - Animal ears       │
  │ - Car wheels        │
  │ - Bottle shapes     │ ← Useful for waste!
  └─────────────────────┘

Layer Group 4 (Final Layers):
  ┌─────────────────────┐
  │ ImageNet Classes    │ ← SPECIFIC (replace)
  │ - Cat vs Dog        │
  │ - Car vs Truck      │
  │ (NOT useful)        │
  └─────────────────────┘
```

**Transfer strategy:**

```
✓ FREEZE Layer Group 1 (Generic features)
✓ FREEZE Layer Group 2 (Textures)
⚠ FINE-TUNE Layer Group 3 (Object parts)
✗ REPLACE Layer Group 4 (Classifier)
```

---

## 🔧 4. FEATURE EXTRACTION VS FINE-TUNING (25 phút)

### **A. Feature Extraction (Phase 1)**

**Concept:**

```
Feature Extraction = Dùng pretrained model như FEATURE EXTRACTOR

Pretrained Model:
  Input → [Frozen Layers] → Features → [New Classifier] → Output
          ↑ NOT trained          ↑ Trained
```

**Implementation:**

```python
# 1. Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False)

# 2. FREEZE all layers
base_model.trainable = False  # ← KEY!

# 3. Add new classifier
model = keras.Sequential([
    base_model,                          # Frozen feature extractor
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 waste classes
])

# 4. Compile & Train
model.compile(
    optimizer=Adam(lr=1e-4),  # Lower LR for stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_ds, epochs=20, validation_data=val_ds)
```

**Tại sao freeze base_model?**

```
Pretrained weights đã TỐT:
  ✓ Learned on 1.2M images
  ✓ Generic features work well
  ✓ Don't want to DESTROY them!

If NOT frozen:
  ✗ Random new classifier weights
  ✗ Large gradients backprop to base_model
  ✗ DESTROY pretrained features!
  ✗ Result: Worse than baseline!

Frozen:
  ✓ Preserve pretrained features
  ✓ Only train new classifier
  ✓ Stable training
  ✓ Fast convergence
```

**Training dynamics:**

```
Phase 1: Feature Extraction (20 epochs)

Epoch 1:  Val Acc = 85.12%  ← Already GOOD! (vs Baseline 70%)
  → Pretrained features work well!

Epoch 5:  Val Acc = 90.34%
  → New classifier adapting

Epoch 10: Val Acc = 92.10%
  → Near convergence

Epoch 20: Val Acc = 92.78%
  → PLATEAU (classifier learned)
```

---

### **B. Fine-Tuning (Phase 2)**

**Concept:**

```
Fine-Tuning = UN-FREEZE some layers, train với LR RẤT THẤP

Pretrained Model:
  Input → [Frozen Early] → [Trainable Late] → [Classifier] → Output
          ↑ Still frozen   ↑ Fine-tuned       ↑ Already trained
```

**Tại sao Fine-Tuning?**

```
After Phase 1:
  ✓ Classifier learned (92.78%)
  ⚠ Base model features still "ImageNet-specific"
  ⚠ Not perfectly adapted to waste data

Phase 2 Goal:
  → Adapt high-level features to waste domain
  → Improve 92.78% → 93.90%
```

**Implementation:**

```python
# 1. UN-FREEZE base model
base_model.trainable = True

# 2. FREEZE early layers (keep generic features)
fine_tune_at = 100  # Freeze first 100 layers

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False  # Keep frozen

for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True   # Fine-tune these

# 3. Compile with VERY LOW learning rate
model.compile(
    optimizer=Adam(lr=1e-5),  # 10x smaller than Phase 1!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Fine-tune
history_fine = model.fit(
    train_ds,
    epochs=15,
    validation_data=val_ds,
    initial_epoch=20  # Continue from Phase 1
)
```

**Tại sao LR RẤT THẤP?**

```
High LR (1e-4):
  → Large weight updates
  → DESTROY pretrained features!
  → Overfitting!

Very Low LR (1e-5):
  → Small, careful updates
  → Gently adapt features
  → Preserve pretrained knowledge
  → Stable fine-tuning
```

**Training dynamics:**

```
Phase 2: Fine-Tuning (15 epochs)

Epoch 21: Val Acc = 92.95%  (+0.17% from Phase 1)
  → Late layers adapting

Epoch 25: Val Acc = 93.45%  (+0.50%)
  → Learning waste-specific patterns

Epoch 30: Val Acc = 93.78%  (+0.33%)
  → Approaching optimal

Epoch 35: Val Acc = 93.90%  (+0.12%)
  → BEST RESULT!
  → EarlyStopping (val_loss not improving)
```

---

### **C. Comparison Table**

| Aspect | Feature Extraction | Fine-Tuning |
|--------|-------------------|-------------|
| **Base Model** | Frozen (not trained) | Partially frozen |
| **Trainable Layers** | Only new classifier | Late layers + classifier |
| **Learning Rate** | 1e-4 (moderate) | 1e-5 (very low) |
| **Training Time** | Fast (20 epochs) | Moderate (15 epochs) |
| **Risk** | Low (safe) | Medium (can destroy features) |
| **Accuracy** | 92.78% | 93.90% (+1.12%) |
| **When to use** | Always start here | After Feature Extraction |

---

## 📊 5. TWO-PHASE TRAINING STRATEGY (20 phút)

### **A. Why Two Phases?**

**Problem nếu Fine-Tune ngay từ đầu:**

```
Scenario: Fine-tune all layers từ epoch 1

base_model.trainable = True  # ALL layers trainable
model.compile(optimizer=Adam(lr=1e-5))
model.fit(...)

Result:
  Epoch 1:  Val Acc = 72%  ← SỤT so với baseline!
  Epoch 10: Val Acc = 78%
  Epoch 30: Val Acc = 85%  ← Worse than 2-phase!

Tại sao?
  ✗ New classifier weights = RANDOM
  ✗ Large gradients từ random classifier
  ✗ Backprop → Destroy pretrained features!
  ✗ Model phải học lại từ đầu (but với LR thấp → slow!)
```

**Solution: Two-Phase Training**

```
Phase 1: Feature Extraction (Frozen base)
  → Train ONLY new classifier
  → Classifier learns to use pretrained features
  → Safe, stable, fast convergence
  → Result: 92.78%

Phase 2: Fine-Tuning (Partial unfreeze)
  → Classifier đã tốt rồi (not random!)
  → Now safe to fine-tune late layers
  → Small LR → Gentle adaptation
  → Result: 93.90% (+1.12%)
```

---

### **B. Layer Freezing Strategy**

**MobileNetV2 Architecture (53 layers):**

```
Layers 0-30 (Early):
  ┌──────────────────────┐
  │ Generic Features     │ ← ALWAYS FROZEN
  │ - Edges, colors      │
  │ - Basic textures     │
  └──────────────────────┘

Layers 31-100 (Middle):
  ┌──────────────────────┐
  │ Mid-level Features   │ ← FROZEN in Phase 1 & 2
  │ - Textures, patterns │   (Generic enough)
  └──────────────────────┘

Layers 101-154 (Late):
  ┌──────────────────────┐
  │ High-level Features  │ ← FROZEN in Phase 1
  │ - Object parts       │ ← TRAINABLE in Phase 2
  └──────────────────────┘

New Classifier:
  ┌──────────────────────┐
  │ Waste Classes (10)   │ ← TRAINABLE in both phases
  └──────────────────────┘
```

**Code:**

```python
# Phase 1: ALL base_model frozen
base_model.trainable = False

# Phase 2: Partially frozen
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False  # Layers 0-99: FROZEN
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True   # Layers 100-154: TRAINABLE
```

---

### **C. Learning Rate Schedule**

```
Phase 1: Feature Extraction
  Initial LR: 1e-4 (0.0001)

  Epoch 1-10:  LR = 1e-4
  Epoch 11-15: LR = 5e-5  (ReduceLROnPlateau)
  Epoch 16-20: LR = 2.5e-5

Phase 2: Fine-Tuning
  Initial LR: 1e-5 (0.00001)  ← 10x smaller!

  Epoch 21-30: LR = 1e-5
  Epoch 31-35: LR = 5e-6  (ReduceLROnPlateau)
```

**Visualize:**

```
Learning Rate over Time

LR
 ↑
1e-4 ┤─────╲
     │      ╲___
     │          ╲__
     │             ╲  Phase 1
     │              ╲__
1e-5 ┤                 ─────╲  Phase 2
     │                       ╲__
     │                          ╲
1e-6 ┤                           ──
     └──────────────────────────────→ Epochs
     0     10    20    30    35
```

---

### **D. Complete Training Process**

```python
# ===== PHASE 1: FEATURE EXTRACTION (20 epochs) =====

# 1. Build model with frozen base
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze

model = keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

# 2. Compile
model.compile(
    optimizer=Adam(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train Phase 1
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[
        EarlyStopping(patience=5),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)
# Result: Val Acc = 92.78%

# ===== PHASE 2: FINE-TUNING (15 epochs) =====

# 4. Unfreeze late layers
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

# 5. Recompile with lower LR
model.compile(
    optimizer=Adam(lr=1e-5),  # 10x lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train Phase 2
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    initial_epoch=20,  # Continue from Phase 1
    callbacks=[
        EarlyStopping(patience=5),
        ReduceLROnPlateau(patience=3, factor=0.5)
    ]
)
# Result: Val Acc = 93.90% (+1.12%)
```

---

## 📈 6. RESULTS COMPARISON (15 phút)

### **A. Accuracy Comparison**

```
Train from Scratch (Baseline CNN):
  Train Acc: 81.28%
  Val Acc:   79.59%
  Test Acc:  79.51%
  Gap:       1.69%

Transfer Learning (MobileNetV2):
  Phase 1 only:
    Train Acc: 93.12%
    Val Acc:   92.78%
    Gap:       0.34%  ← Very good!

  Phase 1 + 2 (Full):
    Train Acc: 94.56%
    Val Acc:   94.00%
    Test Acc:  93.90%
    Gap:       0.56%  ← Excellent generalization!

Improvement:
  93.90% - 79.51% = +14.39 percentage points!
  (+18.1% relative improvement!)
```

---

### **B. Training Curves**

```
Baseline CNN:
Val Acc
  ↑
80% ┤                     ───────  ← Plateau at 79.5%
    │                ╱╱╱╱
70% ┤          ╱╱╱╱╱
    │     ╱╱╱╱
60% ┤╱╱╱╱
    └────────────────────────────→ Epochs
    0   5   10  15  20  25  30

Transfer Learning:
Val Acc
  ↑
94% ┤                         ────  ← Phase 2 fine-tuning
    │                    ╱╱╱╱
92% ┤              ──────           ← Phase 1 plateau
    │         ╱╱╱╱╱
88% ┤    ╱╱╱╱
    │╱╱╱╱
84% ┤
    └────────────────────────────→ Epochs
    0   5   10  15  20  25  30  35
           Phase 1      Phase 2
```

---

### **C. Per-Class Improvement**

```
Class Performance (Baseline → MobileNetV2):

Easy Classes:
  clothes:    94.10% → 96.50% (+2.40%)  ✓ Good
  shoes:      89.90% → 95.20% (+5.30%)  ✓ Great

Medium Classes:
  paper:      81.70% → 92.80% (+11.10%) ✅ Excellent!
  plastic:    78.30% → 93.40% (+15.10%) ✅ Huge!
  cardboard:  83.40% → 94.10% (+10.70%) ✅ Great

Hard Classes:
  trash:      52.11% → 82.30% (+30.19%) 🔥 MASSIVE!
  glass:      74.50% → 89.70% (+15.20%) ✅ Huge!
  metal:      76.20% → 91.50% (+15.30%) ✅ Huge!

→ Transfer Learning giúp NHIỀU NHẤT với hard classes!
```

---

## 🎓 TỔNG KẾT

### **Key Concepts:**

1. **Transfer Learning** = Dùng pretrained weights from ImageNet
2. **Two-Phase Training:**
   - Phase 1: Feature Extraction (frozen base)
   - Phase 2: Fine-Tuning (partial unfreeze)
3. **Feature Reusability:** Low-level features transferable across domains
4. **Data Efficiency:** 19K images đủ với Transfer Learning

### **Why Transfer Learning >> Baseline:**

```
✅ Pretrained on 1.2M images (vs 15K waste images)
✅ Deeper architecture (53 layers vs 8)
✅ Better features (learned from diverse data)
✅ Less overfitting (pretrained = regularization)
✅ Faster convergence (start from good features)
```

### **Results:**

```
Baseline:        79.51%
Transfer:        93.90%
Improvement:     +14.39 percentage points! ✅
```

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] Transfer Learning dùng pretrained ImageNet weights
- [ ] Two-phase training: Feature Extraction → Fine-Tuning
- [ ] Phase 1: Freeze base, train classifier (1e-4 LR)
- [ ] Phase 2: Unfreeze late layers, fine-tune (1e-5 LR)
- [ ] Low-level features transferable across domains
- [ ] Transfer Learning tốt hơn vì pretrained on 1.2M images
- [ ] MobileNetV2 đạt 93.90% (+14.39% vs Baseline)

**Nếu OK →** Tiếp tục `05_MobileNetV2_Thuc_Hanh.md` 🚀
