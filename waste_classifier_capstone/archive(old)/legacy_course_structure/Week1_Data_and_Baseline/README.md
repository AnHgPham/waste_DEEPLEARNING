# Week 1: Data Preparation and Baseline CNN

## 📚 Mục tiêu học tập

Tuần này bạn sẽ học:
1. **Khám phá dữ liệu (Data Exploration)** - Phân tích và visualize dataset
2. **Tiền xử lý dữ liệu (Data Preprocessing)** - Chia train/val/test, augmentation
3. **Baseline CNN** - Xây dựng mạng CNN từ đầu

---

## 🎯 Lý thuyết cơ bản

### 1. Convolutional Neural Networks (CNN)

CNN là kiến trúc deep learning đặc biệt hiệu quả cho xử lý ảnh.

**Các thành phần chính:**

#### a) Convolutional Layer
- **Công thức:** `Output = (Input * Kernel) + Bias`
- **Mục đích:** Trích xuất features từ ảnh (edges, textures, patterns)
- **Parameters:** Filters (số lượng kernels), kernel size (thường 3×3 hoặc 5×5)

#### b) Activation Function (ReLU)
- **Công thức:** `f(x) = max(0, x)`
- **Mục đích:** Thêm tính phi tuyến vào mô hình
- **Lợi ích:** Nhanh, tránh vanishing gradient

#### c) Pooling Layer
- **Max Pooling:** Lấy giá trị lớn nhất trong vùng
- **Average Pooling:** Lấy trung bình
- **Mục đích:** Giảm kích thước, tăng tính bất biến vị trí

#### d) Batch Normalization
- **Công thức:** `y = γ * (x - μ) / σ + β`
- **Mục đích:** Ổn định quá trình training, tăng tốc độ học

#### e) Dropout
- **Cơ chế:** Randomly "tắt" một phần neurons trong training
- **Mục đích:** Regularization, tránh overfitting
- **Rate:** Thường dùng 0.5 (tắt 50% neurons)

### 2. Data Augmentation

Tăng cường dữ liệu để model học tốt hơn và tránh overfitting.

**Các kỹ thuật:**
- **Horizontal Flip:** Lật ảnh ngang
- **Rotation:** Xoay ảnh ±10-20°
- **Zoom:** Phóng to/thu nhỏ
- **Contrast:** Điều chỉnh độ tương phản

**Lưu ý:** Chỉ áp dụng cho training set, KHÔNG cho val/test set.

### 3. Training Process

**Loss Function:** Categorical Cross-Entropy
```
L = -Σ y_true * log(y_pred)
```

**Optimizer:** Adam (Adaptive Moment Estimation)
- Kết hợp momentum và RMSprop
- Learning rate tự động điều chỉnh
- Default: lr=0.001, β₁=0.9, β₂=0.999

**Callbacks:**
- **EarlyStopping:** Dừng khi val_loss không cải thiện
- **ReduceLROnPlateau:** Giảm learning rate khi stuck
- **ModelCheckpoint:** Lưu best model

---

## 📂 Cấu trúc

```
Week1_Data_and_Baseline/
├── README.md                    # File này - lý thuyết
├── assignments/                 # Notebooks (học tập, tương tác)
│   ├── W1_Data_Exploration.ipynb
│   ├── W1_Preprocessing.ipynb
│   └── W1_Baseline_CNN.ipynb
├── data_exploration.py          # Script (production, chạy nhanh)
├── preprocessing.py
├── baseline_training.py
└── utils/                       # Helper functions
    ├── data_utils.py
    └── model_utils.py
```

---

## 🚀 Cách sử dụng

### Option 1: Python Scripts (Nhanh, Production)

```bash
# 1. Khám phá dữ liệu
python Week1_Data_and_Baseline/data_exploration.py

# 2. Tiền xử lý
python Week1_Data_and_Baseline/preprocessing.py

# 3. Train baseline model
python Week1_Data_and_Baseline/baseline_training.py

# Hoặc dùng main.py
python main.py --week 1
```

### Option 2: Jupyter Notebooks (Học tập, Tương tác)

```bash
cd Week1_Data_and_Baseline/assignments
jupyter notebook W1_Data_Exploration.ipynb
```

**Khi nào dùng gì?**
- 📓 **Notebooks:** Khi bạn muốn học, thử nghiệm, xem output từng bước
- 🐍 **Scripts:** Khi bạn muốn chạy full pipeline nhanh chóng

---

## 📊 Kiến trúc Baseline CNN

```
Input (224×224×3)
    ↓
[Conv Block 1: 32 filters]
    Conv2D(32, 3×3, ReLU) → Conv2D(32, 3×3, ReLU)
    → BatchNorm → MaxPool(2×2)
    ↓
[Conv Block 2: 64 filters]
    Conv2D(64, 3×3, ReLU) → Conv2D(64, 3×3, ReLU)
    → BatchNorm → MaxPool(2×2)
    ↓
[Conv Block 3: 128 filters]
    Conv2D(128, 3×3, ReLU) → Conv2D(128, 3×3, ReLU)
    → BatchNorm → MaxPool(2×2)
    ↓
[Conv Block 4: 256 filters]
    Conv2D(256, 3×3, ReLU) → Conv2D(256, 3×3, ReLU)
    → BatchNorm → MaxPool(2×2)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU) → Dropout(0.5)
    ↓
Dense(10, Softmax)
```

**Số parameters:** ~10M  
**Expected accuracy:** 75-80%

---

## 📈 Kết quả mong đợi

Sau khi hoàn thành Week 1:

✅ Hiểu được cấu trúc dataset (10 classes, ~20k images)  
✅ Biết cách split data và augmentation  
✅ Xây dựng được CNN từ đầu  
✅ Đạt ~75-80% accuracy trên test set  
✅ Hiểu overfitting và cách khắc phục  

---

## 🔗 Tài liệu tham khảo

1. **CNN Architecture:**
   - CS231n: Convolutional Neural Networks - Stanford
   - Deep Learning Specialization - Andrew Ng (Coursera)

2. **Data Augmentation:**
   - "The Effectiveness of Data Augmentation in Image Classification" (2017)
   - Keras Documentation: ImageDataGenerator

3. **Training Techniques:**
   - "Batch Normalization: Accelerating Deep Network Training" (2015)
   - "Dropout: A Simple Way to Prevent Overfitting" (2014)

---

## 💡 Tips học tập

1. **Chạy notebooks trước** để hiểu từng bước
2. **Xem visualization** của filters và feature maps
3. **Thử nghiệm** với các hyperparameters khác nhau
4. **So sánh** model có/không có augmentation
5. **Ghi chú** những observations quan trọng

---

**Next:** [Week 2 - Transfer Learning](../Week2_Transfer_Learning/README.md)

