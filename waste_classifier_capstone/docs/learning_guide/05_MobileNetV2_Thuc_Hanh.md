# 📱 MOBILENETV2 THỰC HÀNH

**Thời gian:** 1.5 giờ
**Mục tiêu:** Hiểu MobileNetV2 architecture và implementation trong dự án

---

## 📌 1. MOBILENETV2 LÀ GÌ? (10 phút)

### **Định nghĩa:**

**MobileNetV2 = CNN được thiết kế cho MOBILE DEVICES**

```
Mục tiêu:
✅ Nhẹ (lightweight) - Ít parameters
✅ Nhanh (fast) - Inference nhanh
✅ Chính xác (accurate) - Accuracy cao
✅ Efficient - Tiết kiệm memory, power

→ Perfect cho deployment lên phone, Raspberry Pi, edge devices!
```

### **Specs:**

```
Architecture:
  - 53 layers (deep!)
  - 3.5M parameters (lightweight!)
  - ImageNet Top-1: 72.0%
  - Latency: ~75ms on mobile CPU

Trong dự án:
  - Pretrained on ImageNet
  - Fine-tuned for waste classification
  - Final accuracy: 93.90%
  - Model size: 25 MB (Keras), 9.8 MB (TFLite FP32)
```

---

## 🔍 2. KEY INNOVATIONS (20 phút)

### **A. Depthwise Separable Convolution**

**Problem với Standard Conv:**

```
Standard Convolution (Baseline CNN):
  Input: [H, W, C_in]
  Filters: F filters of size [K, K, C_in]
  Output: [H, W, C_out=F]

  Parameters: K × K × C_in × C_out

Example:
  Input: [56, 56, 64]
  Filters: 128 filters of [3, 3, 64]
  Parameters: 3 × 3 × 64 × 128 = 73,728 params

  ✗ NHIỀU parameters!
  ✗ CHẬM computation!
```

**Solution: Depthwise Separable Conv**

```
Depthwise Separable = Depthwise Conv + Pointwise Conv
```

**Step 1: Depthwise Convolution**

```
Depthwise Conv: Apply 1 filter PER INPUT CHANNEL

Input: [56, 56, 64]
Filters: 64 filters of [3, 3, 1]  ← 1 filter per channel!
Output: [56, 56, 64]  ← Same channels

Parameters: 3 × 3 × 1 × 64 = 576 params

Visual:
Channel 1 ──[3x3 filter 1]──→ Output Channel 1
Channel 2 ──[3x3 filter 2]──→ Output Channel 2
...
Channel 64──[3x3 filter 64]──→ Output Channel 64

→ Spatial filtering ONLY (không mix channels)
```

**Step 2: Pointwise Convolution**

```
Pointwise Conv: 1×1 conv to MIX CHANNELS

Input: [56, 56, 64]
Filters: 128 filters of [1, 1, 64]  ← 1×1 size!
Output: [56, 56, 128]

Parameters: 1 × 1 × 64 × 128 = 8,192 params

Visual:
[C1, C2, ..., C64] ──[1×1 Conv]──→ Output Channel 1
[C1, C2, ..., C64] ──[1×1 Conv]──→ Output Channel 2
...
[C1, C2, ..., C64] ──[1×1 Conv]──→ Output Channel 128

→ Channel mixing ONLY (không spatial filtering)
```

**Comparison:**

```
Standard Conv:
  Params: 3 × 3 × 64 × 128 = 73,728

Depthwise Separable Conv:
  Depthwise:  3 × 3 × 1 × 64  = 576
  Pointwise:  1 × 1 × 64 × 128 = 8,192
  Total:                        8,768

Reduction: 73,728 / 8,768 = 8.4x fewer params! 🔥
           (Same computation reduction!)
```

**Why it works:**

```
Key Insight: Spatial filtering và Channel mixing là INDEPENDENT!

Standard Conv mixes both:
  ✗ Inefficient (redundant computations)

Depthwise Separable separates them:
  ✓ Depthwise: Spatial filtering
  ✓ Pointwise: Channel mixing
  ✓ Much more efficient!
```

---

### **B. Inverted Residual Block**

**ResNet Residual Block (traditional):**

```
Input [256] → Bottleneck Conv [64] → Conv [64] → Expand [256] → Add
     └──────────────────────────────────────────────────────────┘
                         Skip connection

Pattern: Wide → Narrow → Wide (bottleneck in middle)
```

**MobileNetV2 Inverted Residual:**

```
Input [24] → Expand [144] → Depthwise [144] → Project [24] → Add
    └────────────────────────────────────────────────────────┘
                      Skip connection

Pattern: Narrow → Wide → Narrow (inverted!)
```

**Detailed Steps:**

```
Step 1: Expansion (1×1 Conv)
  Input: [56, 56, 24]
  Expand to: [56, 56, 144]  (6x expansion!)

  Purpose: Create high-dimensional space for feature learning

Step 2: Depthwise Conv (3×3)
  Input: [56, 56, 144]
  Depthwise Conv: [56, 56, 144]
  ReLU6 activation

  Purpose: Spatial filtering in expanded space

Step 3: Projection (1×1 Conv)
  Input: [56, 56, 144]
  Project to: [56, 56, 24]  (compress back)
  LINEAR activation (no ReLU!)

  Purpose: Compress back to low dimension

Step 4: Skip Connection
  IF input_channels == output_channels:
    output = input + projection_output
  ELSE:
    output = projection_output
```

**Why "Inverted"?**

```
Traditional Residual:
  256 → [64] → 256
  Wide → Narrow → Wide
  Bottleneck in MIDDLE

Inverted Residual:
  24 → [144] → 24
  Narrow → Wide → Narrow
  Expansion in MIDDLE (inverted!)
```

**Benefits:**

```
✅ Memory efficient:
   - Input/Output: Narrow (24 channels)
   - Intermediate: Wide (144 channels)
   - Skip connection: Only 24 channels (cheap!)

✅ Expressiveness:
   - Expansion creates rich representation
   - Depthwise in high dimension = more features

✅ Linear Bottleneck:
   - Last layer = Linear (no ReLU)
   - Preserve information (ReLU kills negatives!)
```

---

### **C. Linear Bottleneck**

**Why NO ReLU in last layer?**

```
Problem with ReLU in low dimension:

ReLU(x) = max(0, x)
  → Kills all negative values
  → In low dimension (24 channels), information LOSS!

Example:
  Before ReLU: [-0.5, 0.8, -0.2, 0.3, ...]  (24 values)
  After ReLU:  [0.0,  0.8, 0.0,  0.3, ...]  (lost 2/4 values!)

  → Information lost permanently!

In high dimension (144 channels):
  Before ReLU: 144 values
  After ReLU: ~72 values become 0

  → Still have 72 non-zero values
  → Less information loss (redundancy)

Solution: LINEAR bottleneck
  → No ReLU in projection layer
  → Preserve ALL information!
```

---

## 🏗️ 3. MOBILENETV2 ARCHITECTURE (20 phút)

### **A. Overall Structure**

```
MobileNetV2 (53 layers):

┌─────────────────────────────────────┐
│ Input: [224, 224, 3]                │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Conv2D 3×3, 32 filters, stride=2    │ ← Initial conv
│ Output: [112, 112, 32]              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 1         │
│ t=1, c=16, n=1, s=1                 │
│ Output: [112, 112, 16]              │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 2         │
│ t=6, c=24, n=2, s=2                 │
│ Output: [56, 56, 24]                │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 3         │
│ t=6, c=32, n=3, s=2                 │
│ Output: [28, 28, 32]                │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 4         │
│ t=6, c=64, n=4, s=2                 │
│ Output: [14, 14, 64]                │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 3         │
│ t=6, c=96, n=3, s=1                 │
│ Output: [14, 14, 96]                │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 3         │
│ t=6, c=160, n=3, s=2                │
│ Output: [7, 7, 160]                 │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Inverted Residual Block × 1         │
│ t=6, c=320, n=1, s=1                │
│ Output: [7, 7, 320]                 │
└─────────────────────────────────────┘
          ↓
┌─────────────────────────────────────┐
│ Conv2D 1×1, 1280 filters            │ ← Final conv
│ Output: [7, 7, 1280]                │
└─────────────────────────────────────┘

Legend:
  t = expansion factor (expansion ratio)
  c = output channels
  n = number of repeats
  s = stride (first block only)
```

**Parameters:**
```
Total: 3.5M (ImageNet version)
Trong dự án: 2.7M (custom classifier)
```

---

### **B. Receptive Field**

```
MobileNetV2 Receptive Field: ~150×150 pixels (70% of image)

Comparison:
  Baseline:    61×61   (27%)  ← Too small!
  MobileNetV2: 150×150 (70%)  ← Much better!

Visualization:
┌──────────────────────┐
│┌──────────────────┐  │
││ [Plastic Bottle] │  │ ← MobileNetV2 sees FULL object!
││ ┌──────┐         │  │
││ │ Cap  │         │  │
││ │ Body │         │  │
││ │Label │         │  │
││ └──────┘         │  │
│└──────────────────┘  │
└──────────────────────┘
```

---

## 💻 4. CODE WALKTHROUGH (30 phút)

### **File: src/models/transfer.py**

#### **Function 1: build_transfer_model**

```python
def build_transfer_model(input_shape, num_classes, freeze_base=True):
    """
    Build MobileNetV2 transfer learning model.

    Arguments:
    input_shape: (224, 224, 3)
    num_classes: 10 (waste classes)
    freeze_base: True for Phase 1, False for Phase 2

    Returns:
    model: Keras Model
    """

    # ===== STEP 1: Load Pretrained Base =====
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,      # (224, 224, 3)
        include_top=False,             # Remove ImageNet classifier
        weights='imagenet'             # Load pretrained weights
    )

    # Base model layers:
    #   - 154 layers total
    #   - Includes all inverted residual blocks
    #   - Output: [7, 7, 1280]

    # ===== STEP 2: Freeze Base Model =====
    base_model.trainable = not freeze_base

    # freeze_base=True (Phase 1):
    #   → All 154 layers FROZEN
    #   → Only train new classifier

    # freeze_base=False (Phase 2):
    #   → All 154 layers TRAINABLE
    #   → Will freeze selectively later

    # ===== STEP 3: Build Complete Model =====
    inputs = keras.Input(shape=input_shape)

    # Preprocessing for MobileNetV2
    # Converts [0, 255] → [-1, 1]
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)

    # IMPORTANT: training argument for BatchNorm
    # Phase 1 (frozen): training=False → Use pretrained BN stats
    # Phase 2 (fine-tune): training=True → Update BN stats
    x = base_model(x, training=not freeze_base)

    # Output from base: [7, 7, 1280]

    # ===== STEP 4: Classification Head =====
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)
    # [7, 7, 1280] → [1280]

    # Dense Layer 1
    x = layers.Dense(256, activation='relu', name="Dense_1")(x)
    # [1280] → [256]
    x = layers.BatchNormalization(name="BatchNorm_1")(x)
    x = layers.Dropout(0.3, name="Dropout_1")(x)

    # Dense Layer 2 (deeper head for more capacity)
    x = layers.Dense(128, activation='relu', name="Dense_2")(x)
    # [256] → [128]
    x = layers.BatchNormalization(name="BatchNorm_2")(x)
    x = layers.Dropout(0.3, name="Dropout_2")(x)

    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax', name="Classifier")(x)
    # [128] → [10]

    # ===== STEP 5: Create Model =====
    model = keras.Model(inputs, outputs, name="MobileNetV2_Transfer_Learning")

    return model
```

**Model Summary:**

```
Total parameters:    2,753,930
Trainable (Phase 1): 428,298  (15.5%) ← Only classifier
Frozen (Phase 1):    2,325,632 (84.5%) ← Base model
```

---

#### **Function 2: unfreeze_layers**

```python
def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreeze top N layers for Phase 2 fine-tuning.

    Example: unfreeze_layers(model, 54)
    → Unfreeze last 54 layers (top ~35% of 154 layers)
    """

    # Get base model (actual name from Keras)
    base_model = model.get_layer('mobilenetv2_1.00_224')
    base_model.trainable = True

    # First, freeze ALL layers
    for layer in base_model.layers:
        layer.trainable = False

    # Then, unfreeze top N layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    print(f"Unfroze {num_layers_to_unfreeze} layers from base model.")

    # Example: unfreeze_layers(model, 54)
    # Frozen: layers[0:100]   (early and middle)
    # Trainable: layers[100:154] (late layers)

    return model
```

**Layer Freezing Strategy:**

```
Total 154 layers in base_model:

Layers 0-99 (Early & Middle):
  ┌────────────────────────┐
  │ Generic features       │ ← ALWAYS FROZEN
  │ - Edges, textures      │
  │ - Basic patterns       │
  └────────────────────────┘

Layers 100-154 (Late):
  ┌────────────────────────┐
  │ High-level features    │ ← FROZEN in Phase 1
  │ - Object parts         │ ← TRAINABLE in Phase 2
  │ - Semantic patterns    │
  └────────────────────────┘
```

---

### **File: scripts/04_transfer_learning.py**

**Complete Training Flow:**

```python
# ===== PHASE 1: FEATURE EXTRACTION =====

print("PHASE 1: FEATURE EXTRACTION")

# 1. Build model with frozen base
model = build_transfer_model(
    input_shape=(224, 224, 3),
    num_classes=10,
    freeze_base=True  # ← FREEZE!
)

# 2. Compile with moderate LR
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 0.0001
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Callbacks
callbacks_phase1 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('mobilenetv2_phase1.keras', save_best_only=True)
]

# 4. Train Phase 1
history_phase1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_phase1
)

print(f"Phase 1 Result: {max(history_phase1.history['val_accuracy']):.2%}")

# ===== PHASE 2: FINE-TUNING =====

print("\nPHASE 2: FINE-TUNING")

# 5. Unfreeze top layers
model = unfreeze_layers(model, num_layers_to_unfreeze=54)

# 6. Recompile with VERY LOW LR
model.compile(
    optimizer=Adam(learning_rate=1e-5),  # 10x lower!
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Callbacks Phase 2
callbacks_phase2 = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('mobilenetv2_final.keras', save_best_only=True)
]

# 8. Train Phase 2
history_phase2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    initial_epoch=20,  # Continue from Phase 1
    callbacks=callbacks_phase2
)

print(f"Phase 2 Result: {max(history_phase2.history['val_accuracy']):.2%}")

# ===== FINAL EVALUATION =====

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {test_acc:.2%}")
```

---

## 📊 5. TRAINING RESULTS (15 phút)

### **A. Phase 1: Feature Extraction**

```
Configuration:
  - Frozen base: 2,325,632 params
  - Trainable classifier: 428,298 params
  - Learning rate: 1e-4
  - Epochs: 20

Results:
  Epoch 1:  Val Acc = 85.12%  ← Already good!
  Epoch 5:  Val Acc = 90.34%
  Epoch 10: Val Acc = 92.10%
  Epoch 15: Val Acc = 92.65%
  Epoch 20: Val Acc = 92.78%  ← Best

Training Time: ~30 minutes
```

**Why already good at Epoch 1?**

```
Pretrained features WORK WELL:
  ✓ Edges, textures learned on ImageNet
  ✓ Transferable to waste images
  ✓ Classifier just needs to learn linear combination
  ✓ Fast convergence!
```

---

### **B. Phase 2: Fine-Tuning**

```
Configuration:
  - Unfrozen late layers: 54 layers
  - Total trainable: 1,234,890 params
  - Learning rate: 1e-5 (very low!)
  - Epochs: 15

Results:
  Epoch 21: Val Acc = 92.95% (+0.17% from Phase 1)
  Epoch 25: Val Acc = 93.45% (+0.50%)
  Epoch 30: Val Acc = 93.78% (+0.33%)
  Epoch 35: Val Acc = 93.90% (+0.12%) ← BEST!

  EarlyStopping at Epoch 35 (patience=5)

Training Time: ~30 minutes
```

**Why Phase 2 improves?**

```
Fine-tuning adapts features to waste domain:
  ✓ ImageNet: cats, dogs, cars
  ✓ Waste: plastic, glass, metal
  ✓ High-level features need adaptation
  ✓ Phase 2 fine-tunes these features
  ✓ Result: +1.12% improvement!
```

---

### **C. Final Results**

```
MobileNetV2 Transfer Learning:

Train Accuracy:      94.56%
Validation Accuracy: 94.00%
Test Accuracy:       93.90%

Train-Val Gap:       0.56%  ← Excellent generalization!

Comparison vs Baseline:
  Baseline:  79.51%
  MobileNet: 93.90%

  Improvement: +14.39 percentage points! 🔥
  Relative:    +18.1%
```

---

### **D. Per-Class Performance**

```
Class Performance (sorted by accuracy):

Top 3:
  1. clothes:   96.50%  ✅
  2. shoes:     95.20%  ✅
  3. cardboard: 94.10%  ✅

Good (>90%):
  4. plastic:   93.40%
  5. paper:     92.80%
  6. biological:91.80%
  7. metal:     91.50%

Medium (85-90%):
  8. glass:     89.70%
  9. battery:   87.90%

Challenging:
  10. trash:    82.30%  ← Hardest (no clear pattern)

Average: 93.90%
```

**Why trash is hardest?**

```
Trash class = General waste
  ✗ No consistent visual pattern
  ✗ Mix of many materials
  ✗ Highly variable appearance
  ✗ Even humans struggle!

Other classes:
  ✓ Consistent appearance
  ✓ Clear material properties
  ✓ Easier to classify
```

---

## 🎓 TỔNG KẾT

### **Key Innovations:**

1. **Depthwise Separable Conv** → 8.4x fewer params than standard conv
2. **Inverted Residuals** → Narrow-Wide-Narrow pattern
3. **Linear Bottleneck** → Preserve information in low dimension
4. **Two-Phase Training** → Feature extraction + Fine-tuning

### **Why MobileNetV2 is Better:**

```
vs Baseline CNN:

Architecture:
  Baseline: 8 layers, 1.4M params
  MobileNetV2: 53 layers, 2.7M params (but efficient!)

Receptive Field:
  Baseline: 61×61 (27%)
  MobileNetV2: 150×150 (70%)  ← Sees full object!

Pre-training:
  Baseline: None (random init)
  MobileNetV2: ImageNet (1.2M images)  ← Huge advantage!

Results:
  Baseline: 79.51%
  MobileNetV2: 93.90% (+14.39%!) 🔥
```

### **Model Comparison:**

| Metric | Baseline CNN | MobileNetV2 |
|--------|--------------|-------------|
| **Accuracy** | 79.51% | 93.90% |
| **Parameters** | 1.4M | 2.7M |
| **Layers** | 8 | 53 |
| **Receptive Field** | 61×61 | 150×150 |
| **Pretrained** | ✗ | ✅ ImageNet |
| **Training Time** | ~60 mins | ~60 mins |
| **Model Size** | ~5.6 MB | ~25 MB |
| **TFLite FP32** | - | 9.8 MB |
| **TFLite INT8** | - | 2.9 MB |

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] MobileNetV2 designed for mobile/edge devices
- [ ] Depthwise Separable Conv = Depthwise + Pointwise
- [ ] 8.4x parameter reduction vs standard conv
- [ ] Inverted Residual: Narrow → Wide → Narrow
- [ ] Linear Bottleneck preserves information
- [ ] Two-phase training: frozen → partial unfreeze
- [ ] Phase 1: 92.78%, Phase 2: 93.90% (+1.12%)
- [ ] Final result: 93.90% (+14.39% vs Baseline)

**Nếu OK →** Tiếp tục `06_Optimization_Va_Deployment.md` 🚀
