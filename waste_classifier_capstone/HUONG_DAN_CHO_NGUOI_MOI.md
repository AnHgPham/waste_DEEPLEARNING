# 🎓 HƯỚNG DẪN PHÂN LOẠI RÁC THẢI BẰNG AI - CHO NGƯỜI MỚI HỌC

> **Dự án Capstone: Hệ thống phân loại rác thải tự động sử dụng Deep Learning**
>
> 📌 **Dành cho người mới bắt đầu học Machine Learning/Deep Learning**

---

## 📖 MỤC LỤC

1. [Dự án này làm gì?](#1-dự-án-này-làm-gì)
2. [Cài đặt và chuẩn bị](#2-cài-đặt-và-chuẩn-bị)
3. [Cấu trúc dự án](#3-cấu-trúc-dự-án)
4. [Hướng dẫn chạy từng bước](#4-hướng-dẫn-chạy-từng-bước)
5. [Giải thích cách hoạt động](#5-giải-thích-cách-hoạt-động)
6. [Kết quả mong đợi](#6-kết-quả-mong-đợi)
7. [Xử lý lỗi thường gặp](#7-xử-lý-lỗi-thường-gặp)
8. [Tìm hiểu thêm](#8-tìm-hiểu-thêm)

---

## 1. DỰ ÁN NÀY LÀM GÌ?

### 🎯 Mục tiêu
Xây dựng một hệ thống AI có thể **tự động nhận diện và phân loại rác thải** thành 10 loại:

```
📦 10 loại rác:
   1. 🔋 Pin (battery)
   2. 🍎 Rác hữu cơ (biological)
   3. 📦 Bìa carton (cardboard)
   4. 👕 Quần áo (clothes)
   5. 🍾 Thủy tinh (glass)
   6. 🔩 Kim loại (metal)
   7. 📄 Giấy (paper)
   8. 🥤 Nhựa (plastic)
   9. 👟 Giày dép (shoes)
   10. 🗑️ Rác thải chung (trash)
```

### 💡 Ứng dụng thực tế
- **Nhà máy tái chế**: Tự động phân loại rác
- **Thùng rác thông minh**: Nhận diện và hướng dẫn bỏ đúng thùng
- **Giáo dục môi trường**: Dạy trẻ em phân loại rác
- **Giám sát môi trường**: Thống kê lượng rác từng loại

---

## 2. CÀI ĐẶT VÀ CHUẨN BỊ

### Bước 1: Cài đặt Python (nếu chưa có)

**Windows:**
```bash
# Tải Python 3.8+ từ python.org
# Chọn "Add Python to PATH" khi cài đặt
```

**macOS/Linux:**
```bash
# Python thường đã có sẵn
python3 --version
```

### Bước 2: Cài đặt môi trường ảo (Virtual Environment)

```bash
# Di chuyển vào thư mục dự án
cd D:\Downloads\waste_classifier_capstone\waste_classifier_capstone

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

**💡 Tại sao cần môi trường ảo?**
- Tách riêng các thư viện của dự án này với hệ thống
- Tránh xung đột phiên bản
- Dễ dàng chia sẻ dự án

### Bước 3: Cài đặt thư viện cần thiết

```bash
# Cài tất cả thư viện trong requirements.txt
pip install -r requirements.txt

# Hoặc cài từng thư viện chính:
pip install tensorflow>=2.13.0
pip install opencv-python>=4.8.0
pip install matplotlib seaborn
pip install pandas numpy
pip install ultralytics  # YOLOv8
pip install jupyter
```

**⏱️ Thời gian**: Khoảng 5-10 phút (tùy tốc độ internet)

### Bước 4: Tải dữ liệu

**Dataset**: Khoảng 19,760 ảnh (1.6 GB)

```bash
# Tải từ Kaggle (cần tài khoản Kaggle):
# https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

# Giải nén vào thư mục data/raw/
# Cấu trúc:
data/raw/
  ├── battery/
  ├── biological/
  ├── cardboard/
  ├── clothes/
  ├── glass/
  ├── metal/
  ├── paper/
  ├── plastic/
  ├── shoes/
  └── trash/
```

---

## 3. CẤU TRÚC DỰ ÁN

```
waste_classifier_capstone/
│
├── 📁 src/                      ← Mã nguồn chính (modules)
│   ├── config.py                ← Cấu hình toàn bộ dự án
│   ├── data/                    ← Xử lý dữ liệu
│   ├── models/                  ← Kiến trúc mô hình AI
│   └── deployment/              ← Tối ưu hóa model
│
├── 📁 scripts/                  ← Scripts chạy từng bước
│   ├── 01_data_exploration.py   ← Khám phá dữ liệu
│   ├── 02_preprocessing.py      ← Chia dữ liệu train/val/test
│   ├── 03_baseline_training.py  ← Train mô hình cơ bản
│   ├── 04_transfer_learning.py  ← Train mô hình nâng cao
│   ├── 05_realtime_detection.py ← Nhận diện real-time
│   └── 06_model_optimization.py ← Tối ưu model
│
├── 📁 notebooks/                ← Jupyter notebooks (học tập)
│   ├── W1_Data_Exploration.ipynb
│   ├── W1_Preprocessing.ipynb
│   ├── W1_Baseline_CNN.ipynb
│   ├── W2_Feature_Extraction.ipynb
│   ├── W2_Fine_Tuning.ipynb
│   ├── W3_Integration.ipynb
│   └── W4_Model_Optimization.ipynb
│
├── 📁 data/                     ← Dữ liệu
│   ├── raw/                     ← Ảnh gốc (19,760 ảnh)
│   └── processed/               ← Ảnh đã chia (train/val/test)
│
├── 📁 outputs/                  ← Kết quả
│   ├── models/                  ← Models đã train
│   ├── reports/                 ← Biểu đồ, báo cáo
│   └── logs/                    ← Logs training
│
└── 📄 main.py                   ← Chạy mọi thứ từ đây!
```

---

## 4. HƯỚNG DẪN CHẠY TỪNG BƯỚC

### 🚀 CÁCH 1: CHẠY NHANH (Khuyến nghị cho người mới)

```bash
# Chạy toàn bộ pipeline (cách đơn giản nhất)
python main.py --quick

# ⏱️ Thời gian: 5-15 phút (epochs giảm để test nhanh)
```

**Script này sẽ tự động:**
1. ✅ Khám phá dữ liệu
2. ✅ Chia dữ liệu train/val/test
3. ✅ Train baseline CNN (5 epochs)
4. ✅ Train MobileNetV2 (3+3 epochs)
5. ✅ Đánh giá kết quả

---

### 🎓 CÁCH 2: HỌC TỪNG TUẦN (Khuyến nghị để hiểu sâu)

#### **TUẦN 1: Dữ liệu và Baseline CNN** ⏱️ ~1-2 giờ

```bash
# Chạy cả tuần 1
python main.py --week 1

# Hoặc chạy từng bước:
# Bước 1: Khám phá dữ liệu
python main.py --explore
# → Xem phân bố classes, sample images
# → Hiểu dataset có bao nhiêu ảnh mỗi loại

# Bước 2: Chia dữ liệu
python main.py --preprocess
# → Chia 80% train / 10% validation / 10% test
# → Tạo thư mục data/processed/

# Bước 3: Train baseline CNN
python main.py --train-baseline --epochs 30
# → Train mô hình CNN từ đầu (from scratch)
# → Kết quả: ~85% accuracy
# ⏱️ Thời gian: 30-60 phút (tùy GPU/CPU)

# Bước 4: Đánh giá baseline
python main.py --evaluate --model baseline
# → Xem confusion matrix
# → Xem accuracy từng class
```

**📚 Học gì ở tuần 1?**
- Cách khám phá và chuẩn bị dữ liệu
- Kiến trúc CNN cơ bản
- Data augmentation (tăng cường dữ liệu)
- Overfitting và cách khắc phục

---

#### **TUẦN 2: Transfer Learning** ⏱️ ~1-2 giờ

```bash
# Chạy cả tuần 2
python main.py --week 2

# Hoặc chạy từng phase:
python main.py --train-transfer --phase1-epochs 20 --phase2-epochs 15
# → Phase 1: Feature extraction (20 epochs)
# → Phase 2: Fine-tuning (15 epochs)
# → Kết quả: ~95% accuracy
# ⏱️ Thời gian: 40-90 phút

# Đánh giá transfer learning model
python main.py --evaluate --model mobilenetv2
```

**📚 Học gì ở tuần 2?**
- Transfer learning là gì?
- MobileNetV2 architecture
- Feature extraction vs Fine-tuning
- Tại sao accuracy tăng từ 85% → 95%?

---

#### **TUẦN 3: Real-time Detection** ⏱️ ~30 phút

```bash
# Chạy real-time detection với webcam
python main.py --realtime --model mobilenetv2

# Hoặc dùng video file:
python scripts/05_realtime_detection.py --video test.mp4

# ⏱️ Real-time: 30+ FPS trên CPU
```

**📚 Học gì ở tuần 3?**
- YOLOv8 object detection
- Pipeline: Detect → Crop → Classify
- Real-time inference

---

#### **TUẦN 4: Model Optimization** ⏱️ ~20 phút

```bash
# Tối ưu model cho mobile/edge devices
python main.py --optimize --model mobilenetv2

# Kết quả:
# → mobilenetv2_final.keras: 9.2 MB
# → mobilenetv2_fp32.tflite: 10.3 MB
# → mobilenetv2_int8.tflite: 3.1 MB (giảm 74%!)
```

**📚 Học gì ở tuần 4?**
- TensorFlow Lite conversion
- INT8 quantization
- Trade-off: Size vs Accuracy
- Deploy lên Raspberry Pi, Android

---

### 📓 CÁCH 3: HỌC BẰNG JUPYTER NOTEBOOKS (Tương tác)

```bash
# Khởi động Jupyter
jupyter notebook

# Mở notebooks/ và chạy từng cell:
# 1. W1_Data_Exploration.ipynb
# 2. W1_Preprocessing.ipynb
# 3. W1_Baseline_CNN.ipynb
# 4. W2_Feature_Extraction.ipynb
# 5. W2_Fine_Tuning.ipynb
# 6. W3_Integration.ipynb
# 7. W4_Model_Optimization.ipynb
```

**💡 Lợi ích:**
- Chạy từng bước, xem kết quả ngay
- Sửa code và thử nghiệm
- Visualize dữ liệu và kết quả

---

## 5. GIẢI THÍCH CÁCH HOẠT ĐỘNG

### 🧠 A. Baseline CNN (Mô hình cơ bản)

**Kiến trúc:**
```
Input Image (224×224×3)
    ↓
Conv Block 1: 32 filters
    ↓
Conv Block 2: 64 filters
    ↓
Conv Block 3: 128 filters
    ↓
Conv Block 4: 256 filters
    ↓
Global Average Pooling
    ↓
Dense Layer (128 units)
    ↓
Output (10 classes)
```

**Giải thích:**
- **Conv Blocks**: Trích xuất features (đường nét, hình dạng, texture)
- **Filters tăng dần**: 32→64→128→256 (học features từ đơn giản → phức tạp)
- **MaxPooling**: Giảm kích thước, giữ thông tin quan trọng
- **Dropout**: Ngăn overfitting (học thuộc lòng)
- **Kết quả**: ~85% accuracy với ~1.2M parameters

**Code chính**: `src/models/baseline.py`

---

### 🚀 B. Transfer Learning (Mô hình nâng cao)

**MobileNetV2 - Pretrained trên ImageNet**

```
MobileNetV2 Base (Frozen)  ← Đã học 1000 classes từ ImageNet
    ↓                        (14 triệu ảnh!)
Custom Classification Head
    ↓
Dense(256) + BatchNorm + Dropout
    ↓
Dense(128) + BatchNorm + Dropout
    ↓
Output (10 classes - waste)
```

**Two-Phase Training:**

**Phase 1: Feature Extraction (20 epochs)**
```python
# Đóng băng base model → CHỈ train classification head
freeze_base = True
learning_rate = 0.0001  # LR thấp
# → Base model extract features
# → Classification head học phân loại rác
```

**Phase 2: Fine-Tuning (15 epochs)**
```python
# Mở khóa top 30 layers → Fine-tune cho waste domain
unfreeze_top_30_layers()
learning_rate = 0.00001  # LR rất thấp
# → Adapt features cho waste images
# → Cải thiện accuracy
```

**Tại sao tốt hơn?**
- ✅ MobileNetV2 đã học **general features** (cạnh, texture, hình dạng)
- ✅ Ta chỉ cần dạy nó **phân biệt rác** → học nhanh hơn
- ✅ Ít dữ liệu hơn, accuracy cao hơn: 85% → **95%**

**Code chính**: `src/models/transfer.py`

---

### 📹 C. Real-time Detection

**Pipeline:**

```
1. Webcam/Video Frame
    ↓
2. YOLOv8 Detect Objects
    ↓ (tìm vị trí objects: bounding boxes)
3. Crop Each Object
    ↓
4. MobileNetV2 Classify
    ↓ (cardboard? plastic? glass?)
5. Draw Boxes + Labels
    ↓
6. Display Real-time
```

**Ví dụ:**
```
Frame → YOLO detect 3 objects
  ↓
Object 1 (box [100,50,200,150]) → Classify → "plastic" 95%
Object 2 (box [250,80,350,180]) → Classify → "paper" 87%
Object 3 (box [400,100,500,200]) → Classify → "cardboard" 92%
  ↓
Draw boxes with labels → Display
```

**Code chính**: `scripts/05_realtime_detection.py`

---

### ⚡ D. Model Optimization

**Mục tiêu**: Deploy lên thiết bị edge (mobile, Raspberry Pi)

**Quy trình:**

```
1. Keras Model (9.2 MB, FP32)
    ↓ TFLite Conversion
2. TFLite FP32 Model (10.3 MB)
    ↓ INT8 Quantization
3. TFLite INT8 Model (3.1 MB)  ← Giảm 74% size!
```

**INT8 Quantization:**
- **FP32**: 32-bit floating point (chính xác cao, nặng)
- **INT8**: 8-bit integer (nhẹ hơn, nhanh hơn)
- **Trade-off**: Size giảm 74%, accuracy chỉ giảm 1% (95%→94%)

**Code chính**: `src/deployment/optimize.py`

---

## 6. KẾT QUẢ MONG ĐỢI

### 📊 Model Performance

| Model | Accuracy | Size | Inference Time |
|-------|----------|------|----------------|
| Baseline CNN | ~85% | 4.8 MB | 15 ms |
| MobileNetV2 | ~95% | 9.2 MB | 20 ms |
| MobileNetV2 (INT8) | ~94% | **2.4 MB** | **8 ms** |

### 📁 Files được tạo ra

```
outputs/
├── models/
│   ├── baseline_final.keras            (4.8 MB)
│   ├── mobilenetv2_phase1.keras        (14 MB)
│   ├── mobilenetv2_final.keras         (9.2 MB)
│   ├── mobilenetv2_fp32.tflite         (10.3 MB)
│   └── mobilenetv2_int8.tflite         (2.4 MB) ← BEST!
│
├── reports/
│   ├── class_distribution.png          (biểu đồ phân bố classes)
│   ├── baseline_training_history.png   (loss/accuracy curves)
│   ├── mobilenetv2_phase1_history.png
│   ├── mobilenetv2_phase2_history.png
│   ├── baseline_confusion_matrix.png   (ma trận nhầm lẫn)
│   └── mobilenetv2_confusion_matrix.png
│
└── logs/
    └── training_*.log
```

### 🎯 Accuracy per Class

**Ví dụ kết quả MobileNetV2:**
```
Class          Precision  Recall  F1-Score
─────────────────────────────────────────
battery           96%      94%     95%
biological        93%      91%     92%
cardboard         97%      96%     96%
clothes           94%      93%     93%
glass             96%      97%     96%
metal             95%      94%     94%
paper             96%      95%     95%
plastic           97%      98%     97%
shoes             92%      91%     91%
trash             89%      90%     89%
─────────────────────────────────────────
Overall           95%      95%     95%
```

---

## 7. XỬ LÝ LỖI THƯỜNG GẶP

### ❌ Lỗi 1: "No module named 'src'"

**Nguyên nhân**: Chạy script từ sai thư mục

**Cách sửa:**
```bash
# Đảm bảo đang ở thư mục gốc dự án
cd D:\Downloads\waste_classifier_capstone\waste_classifier_capstone
pwd  # Kiểm tra đường dẫn

# Chạy lại
python main.py --quick
```

---

### ❌ Lỗi 2: "Raw data directory not found"

**Nguyên nhân**: Chưa tải dataset

**Cách sửa:**
```bash
# 1. Tải dataset từ Kaggle
# 2. Giải nén vào data/raw/
# 3. Kiểm tra cấu trúc:
ls data/raw/
# Phải thấy: battery/, biological/, cardboard/, ...
```

---

### ❌ Lỗi 3: "Out of Memory" (GPU/CPU)

**Nguyên nhân**: Batch size quá lớn

**Cách sửa:**
```python
# Edit src/config.py
BATCH_SIZE = 16  # Giảm từ 32 xuống 16
```

---

### ❌ Lỗi 4: Webcam không mở được (Real-time)

**Cách sửa:**
```bash
# Thử camera index khác
python scripts/05_realtime_detection.py --camera 1

# Hoặc dùng video file
python scripts/05_realtime_detection.py --video test.mp4
```

---

### ❌ Lỗi 5: "Weights only load failed" (YOLOv8)

**Nguyên nhân**: PyTorch 2.6+ security update

**Cách sửa**: ✅ **ĐÃ TỰ ĐỘNG SỬA** trong code!
- File `src/detection/detection_utils.py` đã handle
- Không cần làm gì thêm

---

### ❌ Lỗi 6: Training quá chậm

**Tối ưu hóa:**

```python
# 1. Giảm epochs để test
python main.py --train-baseline --epochs 10  # Thay vì 30

# 2. Giảm batch size
BATCH_SIZE = 16  # trong config.py

# 3. Dùng GPU nếu có
# TensorFlow tự động detect GPU, chỉ cần:
pip install tensorflow-gpu  # Nếu có GPU NVIDIA
```

---

## 8. TÌM HIỂU THÊM

### 📖 Đọc lý thuyết

```
docs/theory/
├── Week1_Data_and_Baseline.md      ← CNN basics
├── Week2_Transfer_Learning.md       ← Transfer learning
├── Week3_Realtime_Detection.md      ← YOLOv8
└── Week4_Deployment.md              ← TFLite optimization
```

### 🎥 Khái niệm cần học

**Tuần 1:**
- [ ] Convolutional Neural Networks (CNN)
- [ ] Convolutional layers, pooling layers
- [ ] Activation functions (ReLU, Softmax)
- [ ] Overfitting và regularization
- [ ] Data augmentation

**Tuần 2:**
- [ ] Transfer learning
- [ ] Pretrained models (ImageNet)
- [ ] Feature extraction vs Fine-tuning
- [ ] Learning rate scheduling
- [ ] Freezing/unfreezing layers

**Tuần 3:**
- [ ] Object detection
- [ ] YOLO (You Only Look Once)
- [ ] Bounding boxes và confidence scores
- [ ] Real-time inference

**Tuần 4:**
- [ ] Model optimization
- [ ] TensorFlow Lite
- [ ] Quantization (FP32, INT8)
- [ ] Edge deployment

### 🔗 Resources hữu ích

**TensorFlow & Keras:**
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras Guide](https://keras.io/guides/)

**Transfer Learning:**
- [CS231n Transfer Learning](http://cs231n.github.io/transfer-learning/)

**YOLOv8:**
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

**TensorFlow Lite:**
- [TFLite Guide](https://www.tensorflow.org/lite/guide)

---

## 9. CÁC LỆNH THƯỜNG DÙNG

### Xem cấu hình hiện tại
```bash
python main.py --config
```

### Train với custom epochs
```bash
# Baseline với 50 epochs
python main.py --train-baseline --epochs 50

# Transfer learning với custom epochs cho cả 2 phases
python main.py --train-transfer --phase1-epochs 25 --phase2-epochs 20
```

### Evaluate models
```bash
# Đánh giá baseline
python main.py --evaluate --model baseline

# Đánh giá MobileNetV2
python main.py --evaluate --model mobilenetv2
```

### Chạy toàn bộ pipeline (full)
```bash
# Full pipeline với epochs đầy đủ (lâu!)
python main.py --all
# ⏱️ Thời gian: 2-4 giờ
```

---

## 10. TIPS VÀ TRICKS

### 💡 Tip 1: Bắt đầu với --quick
```bash
# Test mọi thứ hoạt động trước khi train lâu
python main.py --quick
```

### 💡 Tip 2: Dùng Jupyter notebooks để học
```bash
# Tương tác, dễ hiểu hơn
jupyter notebook
# Mở notebooks/W1_Data_Exploration.ipynb
```

### 💡 Tip 3: Monitor training
```bash
# Xem logs trong outputs/logs/
# Hoặc dùng TensorBoard:
tensorboard --logdir=outputs/logs/
```

### 💡 Tip 4: Backup models
```bash
# Copy models quan trọng ra ngoài
cp outputs/models/mobilenetv2_final.keras ~/backup/
```

### 💡 Tip 5: Thử nghiệm hyperparameters
```python
# Edit src/config.py:
LEARNING_RATE_BASELINE = 0.01  # Thử LR khác
BATCH_SIZE = 64  # Thử batch size khác
DROPOUT_RATE = 0.3  # Thử dropout rate khác

# Rồi train lại:
python main.py --train-baseline
```

---

## 🎉 CHÚC MỪNG!

Bạn đã biết cách:
- ✅ Chuẩn bị dữ liệu cho deep learning
- ✅ Train CNN từ đầu
- ✅ Áp dụng transfer learning
- ✅ Real-time object detection
- ✅ Tối ưu model cho edge devices

**Next steps:**
- 📚 Đọc papers về MobileNetV2, YOLOv8
- 🔬 Thử nghiệm các architectures khác (ResNet, EfficientNet)
- 🚀 Deploy lên Raspberry Pi hoặc Android app
- 📊 Thử dataset khác
- 💡 Áp dụng cho bài toán thực tế

---

## 📞 HỖ TRỢ

**Nếu gặp vấn đề:**
1. Đọc lại phần "Xử lý lỗi thường gặp"
2. Check logs trong `outputs/logs/`
3. Đọc README.md chính
4. Mở issue trên GitHub (nếu có)

---

**💪 Chúc bạn học tốt và thành công với dự án Deep Learning!**

> *"The best way to learn deep learning is by doing projects!"*

---

**📅 Version:** 2.0.0
**📝 Last Updated:** 2024
**👤 Author:** Pham An
