# 🖼️ CNN & COMPUTER VISION

**Thời gian:** 1.5 giờ
**Mục tiêu:** Hiểu CNN extract features từ ảnh như thế nào

---

## 📌 1. TẠI SAO CẦN CNN? (10 phút)

### **Vấn đề với Neural Network thường:**

```python
# Ảnh 224x224x3
pixels = 224 * 224 * 3 = 150,528 pixels

# Fully Connected Neural Network:
Input Layer: 150,528 neurons
Hidden: 1,000 neurons

→ Weights = 150,528 * 1,000 = 150M parameters!
→ QUÁ NHIỀU! Overfitting ngay!
```

**Problems:**
- ❌ Quá nhiều parameters
- ❌ Không tận dụng spatial structure (vị trí pixels)
- ❌ Không học được local patterns (cạnh, góc...)

---

### **Giải pháp: CNN (Convolutional Neural Network)**

**Ý tưởng:** Thay vì nhìn TẤT CẢ pixels cùng lúc → Nhìn từng **vùng nhỏ**!

```
Fully Connected:        CNN:
Nhìn toàn bộ ảnh       Nhìn từng vùng 3x3
[224x224]              [3x3] → Trượt qua ảnh

🔲🔲🔲🔲🔲              🔍[3x3]
🔲🔲🔲🔲🔲              🔲🔲🔲🔲🔲
🔲🔲🔲🔲🔲   VS        🔲🔲🔲🔲🔲
🔲🔲🔲🔲🔲              🔲🔲🔲🔲🔲
🔲🔲🔲🔲🔲              🔲🔲🔲🔲🔲
```

**Lợi ích:**
- ✅ Ít parameters hơn nhiều!
- ✅ Học được local patterns (edges, corners...)
- ✅ Translation invariant (không quan tâm vị trí)

---

## 🔍 2. CONVOLUTION LÀ GÌ? (20 phút)

### **Định nghĩa:**

**Convolution = Trượt 1 filter (kernel) qua ảnh để extract features**

### **Ví dụ trực quan:**

```
Input Image (5x5):          Filter/Kernel (3x3):
┌─────────────────┐         ┌─────────┐
│ 1  1  1  0  0  │         │ 1  0  1 │
│ 0  1  1  1  0  │         │ 0  1  0 │
│ 0  0  1  1  1  │         │ 1  0  1 │
│ 0  0  1  1  0  │         └─────────┘
│ 0  1  1  0  0  │         Edge Detector
└─────────────────┘
```

**Bước 1: Filter ở góc trái trên**

```
Input:              Filter:         Computation:
┌───────┐           ┌─────────┐
│ 1  1  1│          │ 1  0  1 │     1*1 + 1*0 + 1*1 = 2
│ 0  1  1│  ✕       │ 0  1  0 │  +  0*0 + 1*1 + 1*0 = 1
│ 0  0  1│          │ 1  0  1 │  +  0*1 + 0*0 + 1*1 = 1
└───────┘           └─────────┘     ─────────────────
                                    Result = 4
```

**Bước 2: Trượt sang phải (stride=1)**

```
Input:              Filter:         Computation:
  ┌───────┐         ┌─────────┐
│  1  1  1  0│      │ 1  0  1 │     1*1 + 1*0 + 1*1 = 2
│  1  1  1  0│  ✕   │ 0  1  0 │  +  1*0 + 1*1 + 1*0 = 1
│  0  1  1  1│      │ 1  0  1 │  +  0*1 + 1*0 + 1*1 = 1
  └───────┘         └─────────┘     ─────────────────
                                    Result = 4
```

**Tiếp tục trượt... → Output Feature Map:**

```
Output (3x3):
┌──────────┐
│ 4  3  4 │
│ 2  4  3 │
│ 2  3  4 │
└──────────┘
Feature Map (detected edges!)
```

---

### **Các loại filters phổ biến:**

#### **1. Edge Detection (Vertical)**

```
Filter:                 Detects:
┌──────────┐           │ │ │ │
│ 1   0  -1│           │ │ │ │  Vertical edges
│ 1   0  -1│           │ │ │ │
│ 1   0  -1│           │ │ │ │
└──────────┘
```

#### **2. Edge Detection (Horizontal)**

```
Filter:                 Detects:
┌──────────┐           ─────────
│ 1   1   1│           ─────────  Horizontal edges
│ 0   0   0│           ─────────
│-1  -1  -1│
└──────────┘
```

#### **3. Sharpen**

```
Filter:                 Effect:
┌──────────┐           Makes image sharper
│ 0  -1   0│
│-1   5  -1│
│ 0  -1   0│
└──────────┘
```

#### **4. Blur**

```
Filter (average):       Effect:
┌───────────────┐      Smooths image
│ 1/9  1/9  1/9 │
│ 1/9  1/9  1/9 │
│ 1/9  1/9  1/9 │
└───────────────┘
```

---

### **Multiple Channels (RGB):**

```
Input: 224x224x3 (RGB)

Filter: 3x3x3 (matches input channels!)
  ┌─────────┐
  │ R filter│  3x3
  │ G filter│  3x3
  │ B filter│  3x3
  └─────────┘

Convolution:
R_out = conv(R_input, R_filter)
G_out = conv(G_input, G_filter)
B_out = conv(B_input, B_filter)

Output = R_out + G_out + B_out  → 1 feature map!
```

---

### **Multiple Filters:**

```
Input: 224x224x3

32 Filters (each 3x3x3)
→ 32 Feature Maps (each 224x224)
→ Output: 224x224x32

Conv2D(32 filters, 3x3):
Input  [224, 224, 3]
        ↓
Output [224, 224, 32]  (if padding='same')
```

---

## 📐 3. PADDING, STRIDE, POOLING (15 phút)

### **A. Padding**

**Vấn đề:** Convolution làm ảnh nhỏ đi!

```
Input: 5x5 → Conv 3x3 → Output: 3x3
Input: 224x224 → Conv 3x3 → Output: 222x222
```

**Giải pháp: Zero Padding**

```
Original (5x5):          With Padding (7x7):
┌─────────────┐         ┌───────────────────┐
│ 1  1  1  0  0│        │ 0  0  0  0  0  0  0│
│ 0  1  1  1  0│        │ 0  1  1  1  0  0  0│
│ 0  0  1  1  1│   →    │ 0  0  1  1  1  0  0│
│ 0  0  1  1  0│        │ 0  0  0  1  1  1  0│
│ 0  1  1  0  0│        │ 0  0  0  1  1  0  0│
└─────────────┘         │ 0  0  1  1  0  0  0│
                        │ 0  0  0  0  0  0  0│
                        └───────────────────┘

Conv 3x3 → Output still 5x5! ✅
```

**Trong code:**

```python
# Keras/TensorFlow
Conv2D(32, (3,3), padding='same')   # Output size = Input size
Conv2D(32, (3,3), padding='valid')  # Output size shrinks
```

---

### **B. Stride**

**Stride = Bước nhảy khi trượt filter**

```
Stride = 1 (default):     Stride = 2:
Move 1 pixel              Move 2 pixels

[███]□□□□                 [███]□□□□
□[███]□□                  □□[███]□
□□[███]□                  □□□□[███]
□□□[███]

Output: Large            Output: Half size!
```

**Ví dụ:**

```python
Input: 224x224

Conv2D(32, (3,3), stride=1) → Output: 224x224 (with padding)
Conv2D(32, (3,3), stride=2) → Output: 112x112 (downsampling!)
```

---

### **C. Pooling** ⭐ QUAN TRỌNG

**Pooling = Downsample feature maps**

#### **Max Pooling (phổ biến nhất):**

```
Input (4x4):               Max Pool 2x2:
┌──────────────┐          ┌────────┐
│ 1  3  2  4  │          │ 3  4  │  ← max of each 2x2
│ 5  6  7  8  │    →     │ 14 16 │
│ 9 10 11 12  │          └────────┘
│13 14 15 16  │
└──────────────┘

Process:
[1 3]        [2 4]
[5 6]  → 6   [7 8]  → 8

[9 10]       [11 12]
[13 14] → 14 [15 16] → 16
```

**Tại sao dùng Pooling?**
- ✅ Giảm kích thước (less computation)
- ✅ Giảm overfitting
- ✅ Translation invariance (nhận dạng object dù vị trí thay đổi)

**Trong code:**

```python
MaxPooling2D(pool_size=(2, 2))  # Giảm 50% size

Input:  224x224x32
        ↓
Output: 112x112x32  (height, width /2)
```

---

## 🏗️ 4. CNN ARCHITECTURE (30 phút)

### **Typical CNN Structure:**

```
Input Image
    ↓
[Convolutional Block] ×N
    ↓
[Flatten]
    ↓
[Dense Layers]
    ↓
Output (classes)
```

---

### **Convolutional Block:**

```
Input
    ↓
Convolution (extract features)
    ↓
Activation (ReLU)
    ↓
Pooling (downsample)
    ↓
Output
```

---

### **Ví dụ: Baseline CNN trong dự án**

```python
# src/models/baseline.py

Input: 224x224x3
    ↓
Rescaling (0-1)
    ↓
# Block 1
Conv2D(32, 3x3) → 224x224x32
ReLU
Conv2D(32, 3x3) → 224x224x32
ReLU
BatchNorm
MaxPool(2x2) → 112x112x32
    ↓
# Block 2
Conv2D(64, 3x3) → 112x112x64
ReLU
Conv2D(64, 3x3) → 112x112x64
ReLU
BatchNorm
MaxPool(2x2) → 56x56x64
    ↓
# Block 3
Conv2D(128, 3x3) → 56x56x128
ReLU
Conv2D(128, 3x3) → 56x56x128
ReLU
BatchNorm
MaxPool(2x2) → 28x28x128
    ↓
# Block 4
Conv2D(256, 3x3) → 28x28x256
ReLU
Conv2D(256, 3x3) → 28x28x256
ReLU
BatchNorm
MaxPool(2x2) → 14x14x256
    ↓
GlobalAvgPool → 256
    ↓
Dense(128) + Dropout(0.5)
    ↓
Dense(10, softmax)
    ↓
Output: [plastic prob, glass prob, ...]
```

---

### **Feature Hierarchy (Phân cấp đặc trưng):**

```
Early Layers (Block 1-2):
├─ Edges (cạnh)
├─ Corners (góc)
└─ Simple textures (vân đơn giản)

Mid Layers (Block 3):
├─ Complex textures
├─ Patterns (họa tiết)
└─ Parts (bộ phận nhỏ)

Deep Layers (Block 4):
├─ Object parts (nắp chai, nhãn)
├─ Shapes (hình dạng)
└─ High-level features

Final Layers:
└─ Complete objects (plastic bottle, glass jar...)
```

**Visualize:**

```
Layer 1: Detects       Layer 3: Detects         Layer 5: Detects
│ │ ─                 Textures                  Objects
│ │ ─                 ╱╲╱╲                      🍾 Bottle
\ / ∠                 ░▒▓█                      🥫 Can
```

---

## 📊 5. IMAGE CLASSIFICATION WORKFLOW (15 phút)

### **Complete Pipeline:**

```
1. DATA LOADING
   └─ Load images from folders
   └─ Resize to 224x224
   └─ Normalize [0, 255] → [0, 1]

2. DATA AUGMENTATION
   └─ Random flip, rotation
   └─ Zoom, brightness change
   └─ → Increase diversity!

3. MODEL DEFINITION
   └─ Define CNN architecture
   └─ Compile (loss, optimizer)

4. TRAINING
   └─ Forward prop (predict)
   └─ Calculate loss
   └─ Backprop (gradients)
   └─ Update weights
   └─ Repeat for all epochs

5. EVALUATION
   └─ Test on unseen data
   └─ Calculate accuracy
   └─ Analyze errors

6. DEPLOYMENT
   └─ Save model
   └─ Optimize (TFLite)
   └─ Deploy to production
```

---

### **Trong dự án:**

```python
# 1. Load Data
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(224, 224),
    batch_size=32
)

# 2. Data Augmentation
augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2)
])

# 3. Build Model
model = build_baseline_model(
    input_shape=(224, 224, 3),
    num_classes=10
)

# 4. Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30
)

# 6. Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")
```

---

## 🎨 6. DATA AUGMENTATION (10 phút)

**Tại sao cần?** → Tăng đa dạng dữ liệu, giảm overfitting!

### **Các kỹ thuật phổ biến:**

#### **1. Horizontal Flip**

```
Original:              Flipped:
🍾                     🍾
│  Bottle              Bottle  │
│  facing right        facing left
```

#### **2. Rotation**

```
Original:              Rotated 20°:
   🍾                    ╱🍾
   │                   ╱  │
   │                 ╱    │
```

#### **3. Zoom**

```
Original:              Zoomed In:
┌────────────┐        ┌──────┐
│            │        │ 🍾   │
│    🍾      │   →    │ │ │  │
│    │ │     │        │ │ │  │
│    │ │     │        └──────┘
└────────────┘        (closer view)
```

#### **4. Brightness**

```
Original:              Darker:        Lighter:
███████               ▓▓▓▓▓▓▓        ░░░░░░░
███████               ▓▓▓▓▓▓▓        ░░░░░░░
███████               ▓▓▓▓▓▓▓        ░░░░░░░
```

#### **5. Contrast**

```
Original:              Higher Contrast:
██▓▓▒▒░░              ████░░░░
██▓▓▒▒░░              ████░░░░
```

---

### **Trong dự án:**

```python
# src/data/preprocessing.py

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),          # ±20%
    layers.RandomZoom(0.2),              # ±20%
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1)   # Shift 10%
])
```

**Kết quả:**

```
1 ảnh gốc → 100+ variations khác nhau!

Original plastic bottle
→ Flipped
→ Rotated
→ Zoomed
→ Brightened
...

Effectively: 15,777 → 1,000,000+ training samples!
```

---

## 🎓 TỔNG KẾT

### **CNN vs Fully Connected:**

| Aspect | Fully Connected | CNN |
|--------|----------------|-----|
| **Parameters** | ~150M | ~1.4M |
| **Local patterns** | ❌ No | ✅ Yes (filters) |
| **Spatial info** | ❌ Lost | ✅ Preserved |
| **Translation invariant** | ❌ No | ✅ Yes (pooling) |

---

### **Key Concepts:**

1. **Convolution** = Trượt filter qua ảnh
2. **Filter** = Học detect features (edges, textures...)
3. **Pooling** = Downsample, giảm size
4. **Feature Hierarchy** = Low → Mid → High level
5. **Data Augmentation** = Tăng diversity

---

### **CNN Architecture Pattern:**

```
[Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] Tại sao CNN tốt hơn Fully Connected
- [ ] Convolution extract features bằng filters
- [ ] Padding, Stride, Pooling làm gì
- [ ] CNN architecture: Conv blocks → Dense
- [ ] Data Augmentation tăng data diversity

**Nếu OK →** Tiếp tục `03_Baseline_CNN_Chi_Tiet.md` 🚀

**Nếu chưa hiểu →** Đọc lại phần Convolution (quan trọng nhất!)
