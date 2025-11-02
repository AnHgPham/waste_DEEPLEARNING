# 🧠 DEEP LEARNING CƠ BẢN

**Thời gian:** 1 giờ
**Mục tiêu:** Hiểu Neural Network hoạt động như thế nào

---

## 📌 1. NEURAL NETWORK LÀ GÌ? (10 phút)

### **Định nghĩa đơn giản:**

**Neural Network (Mạng thần kinh) = Thuật toán học từ data**

```
Input (ảnh rác)  →  Neural Network  →  Output (loại rác)
[224x224x3]      →  [magic box]     →  [plastic/glass/metal...]
```

### **Ví dụ dễ hiểu:**

Giống như **não người học nhận dạng:**

```
Em bé nhìn mèo:
Lần 1: "Đây là gì?" → Mẹ: "Mèo!"
Lần 2: "Có tai nhọn, râu" → "Mèo!"
Lần 3: "Kêu meo meo" → "Mèo!"
...
Sau 100 lần → Em bé TỰ ĐỘNG nhận ra mèo!

Neural Network:
Lần 1: Nhìn plastic bottle → Label: "plastic"
Lần 2: "Trong suốt, hình trụ" → "plastic"
Lần 3: "Có nắp vặn" → "plastic"
...
Sau 15,777 ảnh → Model TỰ ĐỘNG phân loại!
```

---

## 📊 2. CẤU TRÚC NEURAL NETWORK (15 phút)

### **A. Neuron (Nơ-ron)**

**Neuron = 1 đơn vị tính toán**

```python
# Công thức 1 neuron:
output = activation(weight * input + bias)

# Ví dụ:
input = [0.5, 0.8, 0.3]  # 3 features
weight = [0.2, 0.5, -0.3] # Học được
bias = 0.1                # Học được

z = (0.5*0.2) + (0.8*0.5) + (0.3*-0.3) + 0.1
  = 0.1 + 0.4 - 0.09 + 0.1
  = 0.51

output = ReLU(0.51) = 0.51 (nếu > 0)
```

**Visualize:**
```
Input 1 (0.5) ──[w=0.2]──┐
                          ├─→ Σ + bias ──→ ReLU ──→ Output (0.51)
Input 2 (0.8) ──[w=0.5]──┤
                          │
Input 3 (0.3) ──[w=-0.3]─┘
```

---

### **B. Layer (Lớp)**

**Layer = Nhiều neurons cùng nhau**

```
Input Layer:     3 neurons (input features)
Hidden Layer 1:  10 neurons
Hidden Layer 2:  10 neurons
Output Layer:    10 neurons (10 waste classes)
```

**Visualize:**
```
Input (3)    Hidden 1 (10)    Hidden 2 (10)    Output (10)
   ●               ●                ●              ● plastic
   ●──┐         ●  ●              ●  ●            ● glass
   ●  ├─────→  ●  ●  ●  ─────→  ●  ●  ●  ─────→  ● metal
      └───→    ●  ●  ●          ●  ●  ●          ● paper
               ●  ●              ●  ●            ● ...
               ●                 ●
```

---

### **C. Activation Functions (Hàm kích hoạt)**

**Tại sao cần?** → Để model học được non-linear patterns!

#### **1. ReLU (Rectified Linear Unit)** ⭐ PHỔ BIẾN NHẤT

```python
def ReLU(x):
    return max(0, x)

# Ví dụ:
ReLU(5) = 5
ReLU(-3) = 0
ReLU(0) = 0
```

**Graph:**
```
y
│     ╱
│    ╱
│   ╱
│  ╱
│ ╱
└─────────→ x
  0
```

**Dùng khi:** Hidden layers (hầu hết trường hợp)

---

#### **2. Softmax** ⭐ OUTPUT LAYER

```python
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

# Ví dụ:
logits = [2.0, 1.0, 0.1]  # raw outputs
softmax([2.0, 1.0, 0.1]) = [0.659, 0.242, 0.099]
                            ↑
                        Tổng = 1.0 (100%)
```

**Dùng khi:** Multi-class classification (plastic/glass/metal...)

---

## 🔄 3. FORWARD PROPAGATION (15 phút)

**Forward Prop = Tính output từ input**

### **Ví dụ cụ thể:**

```python
# Giả sử classify ảnh 28x28 (mnist digit)
input = [pixel_1, pixel_2, ..., pixel_784]  # 784 pixels

# Layer 1: 784 → 128 neurons
hidden1 = ReLU(W1 @ input + b1)  # [128,]

# Layer 2: 128 → 64 neurons
hidden2 = ReLU(W2 @ hidden1 + b2)  # [64,]

# Output: 64 → 10 classes (0-9 digits)
output = Softmax(W3 @ hidden2 + b3)  # [10,]

# Kết quả:
output = [0.01, 0.02, 0.05, 0.80, 0.03, ...]
                            ↑
                    Class 3 có prob cao nhất
                    → Predict: "3"
```

### **Trong dự án Waste Classification:**

```python
# Input: Ảnh waste 224x224x3
x = load_image("plastic_bottle.jpg")  # [224, 224, 3]

# Forward qua Baseline CNN:
x = Rescaling(x)                      # [0, 255] → [0, 1]
x = Conv2D_32(x)                      # → [112, 112, 32]
x = MaxPool(x)                        # → [56, 56, 32]
x = Conv2D_64(x)                      # → [56, 56, 64]
... (nhiều layers)
x = Dense_128(x)                      # → [128,]
output = Dense_10(x)                  # → [10,]

# Softmax:
probabilities = [
    0.02,  # battery
    0.01,  # biological
    0.05,  # cardboard
    0.02,  # clothes
    0.03,  # glass
    0.01,  # metal
    0.02,  # paper
    0.82,  # plastic ← HIGHEST!
    0.01,  # shoes
    0.01   # trash
]

Prediction: "plastic" ✅
```

---

## 🔙 4. BACKPROPAGATION (15 phút)

**Backprop = Học từ sai lầm**

### **Quy trình:**

```
1. Forward Prop → Predict
2. Tính Loss (sai bao nhiêu?)
3. Backward Prop → Tính gradient
4. Update weights → Model học!
```

### **Ví dụ đơn giản:**

```python
# Ground truth: "plastic"
true_label = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                                   ↑ plastic = 1

# Prediction:
prediction = [0.02, 0.01, 0.05, 0.02, 0.03, 0.01, 0.02, 0.82, 0.01, 0.01]

# Loss (얼마나 sai):
loss = -log(0.82) = 0.198  # Càng gần 1 càng tốt!

# Backprop:
# "plastic probability quá thấp (0.82), cần tăng lên!"
# → Adjust weights để lần sau predict plastic = 0.95

# Update:
weights_new = weights_old - learning_rate * gradient
```

### **Trong thực tế:**

```
Epoch 1:
  Image 1 (plastic) → Predict plastic (0.6) → Loss = 0.51
  → Backprop → Update weights

  Image 2 (glass) → Predict glass (0.7) → Loss = 0.36
  → Backprop → Update weights

  ... (15,777 images)

Epoch 2:
  Image 1 (plastic) → Predict plastic (0.75) ↑ Better!
  ... Model đang học!

Epoch 30:
  Image 1 (plastic) → Predict plastic (0.95) ✅ Very good!
```

---

## 📉 5. LOSS FUNCTION (10 phút)

**Loss = Độ đo sai lầm**

### **A. Categorical Cross-Entropy** ⭐ Dùng trong dự án

```python
loss = -Σ (y_true * log(y_pred))

# Ví dụ:
y_true = [0, 0, 1, 0]  # Class 2 (plastic)
y_pred = [0.1, 0.2, 0.6, 0.1]

loss = -(0*log(0.1) + 0*log(0.2) + 1*log(0.6) + 0*log(0.1))
     = -log(0.6)
     = 0.51

# Nếu predict tốt hơn:
y_pred = [0.05, 0.05, 0.85, 0.05]
loss = -log(0.85) = 0.16  ← Thấp hơn = Tốt hơn!
```

**Mục tiêu training:** Minimize loss!

---

## 🎯 6. OPTIMIZER (10 phút)

**Optimizer = Thuật toán update weights**

### **A. SGD (Stochastic Gradient Descent)**

```python
weights_new = weights_old - learning_rate * gradient

# Ví dụ:
weight = 0.5
gradient = 0.2  # Direction to move
learning_rate = 0.01

weight_new = 0.5 - 0.01 * 0.2
           = 0.498
```

---

### **B. Adam** ⭐ PHỔ BIẾN NHẤT (dùng trong dự án)

**Adam = SGD + Momentum + Adaptive LR**

```python
# Adam tự động adjust learning rate cho từng parameter
# → Học nhanh hơn, ổn định hơn SGD

# Config trong dự án:
optimizer = Adam(
    learning_rate=0.001,  # Initial LR
    beta_1=0.9,           # Momentum
    beta_2=0.999          # RMSprop
)
```

**Tại sao dùng Adam?**
- ✅ Tự động adjust LR
- ✅ Faster convergence
- ✅ Work well với CNN

---

## 📊 7. OVERFITTING VS UNDERFITTING (10 phút)

### **A. Underfitting (Học chưa đủ)**

```
Train Acc: 60%
Val Acc:   58%

→ Model quá đơn giản, chưa học đủ pattern
```

**Giải pháp:**
- ✅ Tăng model capacity (more layers/neurons)
- ✅ Train lâu hơn (more epochs)

---

### **B. Overfitting (Học quá kỹ)**

```
Train Acc: 95%
Val Acc:   70%  ← GAP lớn!

→ Model nhớ training data, không generalize
```

**Giải pháp:**
- ✅ Data Augmentation
- ✅ Dropout
- ✅ Early Stopping
- ✅ Regularization

**Trong dự án:**
```python
# Baseline CNN sử dụng:
model.add(Dropout(0.5))  # Dropout 50%
model.add(BatchNormalization())

# Callbacks:
EarlyStopping(patience=5)  # Stop nếu val_loss không giảm
```

---

### **C. Good Fit (Vừa đủ)** ⭐ MỤC TIÊU

```
Train Acc: 94%
Val Acc:   93%  ← Gap nhỏ!

→ Model generalize tốt!
```

**Trong dự án:**
```
MobileNetV2:
  Train Acc: ~95%
  Val Acc: 94.00%
  Test Acc: 93.90%

→ EXCELLENT FIT! ✅
```

---

## 🎓 TỔNG KẾT

### **Concepts quan trọng:**

1. **Neural Network** = Layers of neurons
2. **Forward Prop** = Input → Output
3. **Loss** = Độ đo sai lầm
4. **Backprop** = Học từ loss
5. **Optimizer** = Update weights (Adam)
6. **Overfitting** = Train tốt, Val kém

### **Workflow:**

```
1. Load Data (images + labels)
   ↓
2. Forward Propagation (predict)
   ↓
3. Calculate Loss
   ↓
4. Backpropagation (gradients)
   ↓
5. Update Weights (optimizer)
   ↓
6. Repeat for all data (1 epoch)
   ↓
7. Repeat epochs until converge
```

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] Neural Network có layers, neurons
- [ ] Forward Prop tính output từ input
- [ ] Loss đo độ sai lầm
- [ ] Backprop cập nhật weights để giảm loss
- [ ] Adam optimizer tốt hơn SGD
- [ ] Overfitting vs Underfitting

**Nếu OK →** Tiếp tục `02_CNN_Va_Computer_Vision.md` 🚀
