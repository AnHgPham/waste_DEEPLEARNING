# 🚀 Quick Start Guide

**Chạy dự án trong 3 bước!**

---

## 📍 Bước 1: Vào đúng thư mục

```powershell
# Mở PowerShell/Terminal
cd D:\Downloads\waste_classifier_capstone\waste_classifier_capstone

# Kiểm tra (phải thấy main.py)
ls main.py
```

**✅ Output mong đợi:**
```
-a----        10/29/2025   5:22 PM          12345 main.py
```

**❌ Nếu lỗi "Cannot find path":**
- Bạn đang ở sai thư mục
- Chạy: `cd waste_classifier_capstone`

---

## 🔧 Bước 2: Kiểm tra cấu hình

```powershell
python main.py --config
```

**✅ Nếu OK, bạn sẽ thấy:**
```
======================================================================
               WASTE CLASSIFICATION SYSTEM v2.0
======================================================================

📋 Current Configuration:
   - Image size: (224, 224)
   - Batch size: 32
   - Number of classes: 10
   ...
```

---

## 🎯 Bước 3: Chọn cách chạy

### Option A: Quick Test (Khuyến nghị lần đầu)

**Chạy nhanh để test (~15-30 phút):**

```powershell
python main.py --quick
```

**Sẽ chạy:**
- ✅ Data exploration
- ✅ Preprocessing  
- ✅ Baseline CNN (5 epochs)
- ✅ Transfer Learning (3+3 epochs)
- ✅ Evaluation

---

### Option B: Full Pipeline (Chất lượng cao)

**Chạy đầy đủ với epochs chuẩn (~2-3 giờ):**

```powershell
python main.py --all
```

**Sẽ chạy:**
- ✅ Week 1: Data + Baseline CNN (30 epochs)
- ✅ Week 2: Transfer Learning (20+15 epochs)
- ✅ Week 4: Model Optimization

---

### Option C: Từng bước (Recommended for learning)

**Chạy từng task riêng lẻ:**

```powershell
# Bước 1: Khám phá dữ liệu
python main.py --explore

# Bước 2: Tiền xử lý
python main.py --preprocess

# Bước 3: Train Baseline CNN
python main.py --train-baseline --epochs 30

# Bước 4: Train Transfer Learning
python main.py --train-transfer --phase1-epochs 20 --phase2-epochs 15

# Bước 5: Đánh giá model
python main.py --evaluate --model mobilenetv2

# Bước 6: Tối ưu hóa
python main.py --optimize --model mobilenetv2
```

---

### Option D: Real-time Detection 📹

**Phát hiện và phân loại rác thời gian thực:**

```powershell
# Sử dụng webcam (mặc định)
python scripts/05_realtime_detection.py

# Sử dụng webcam với model cụ thể
python scripts/05_realtime_detection.py --model mobilenetv2 --camera 0

# Sử dụng video file
python scripts/05_realtime_detection.py --video path/to/video.mp4
```

**Controls khi chạy:**
- `Q` - Thoát
- `S` - Lưu screenshot
- `P` - Tạm dừng/Tiếp tục

**Yêu cầu:**
- ✅ Model đã train (mobilenetv2 hoặc baseline)
- ✅ YOLOv8 sẽ tự động download lần đầu
- ✅ Webcam hoặc video file

---

### Option E: Chạy scripts trực tiếp

**Nếu muốn control chi tiết hơn:**

```powershell
# Week 1
python scripts/01_data_exploration.py
python scripts/02_preprocessing.py
python scripts/03_baseline_training.py --epochs 30

# Week 2
python scripts/04_transfer_learning.py --phase1-epochs 20 --phase2-epochs 15

# Week 3 - Real-time Detection
python scripts/05_realtime_detection.py

# Evaluation
python scripts/99_evaluate_model.py --model mobilenetv2

# Week 4
python scripts/06_model_optimization.py --model mobilenetv2
```

---

## 📊 Kiểm tra kết quả

Sau khi chạy xong, kiểm tra outputs:

```powershell
# Xem models đã train
ls outputs/models/

# Xem reports (plots, metrics)
ls outputs/reports/

# Mở report images
start outputs/reports/class_distribution.png
start outputs/reports/baseline_training_history.png
start outputs/reports/confusion_matrix_mobilenetv2.png
```

---

## 🐛 Troubleshooting

### Lỗi: "can't open file main.py"

**Nguyên nhân:** Sai thư mục

**Giải pháp:**
```powershell
cd D:\Downloads\waste_classifier_capstone\waste_classifier_capstone
ls main.py  # Phải thấy file này
```

---

### Lỗi: "No module named 'src'"

**Nguyên nhân:** Chưa vào đúng project root

**Giải pháp:**
```powershell
# Phải chạy từ thư mục có main.py
pwd  # Xem current directory
cd D:\Downloads\waste_classifier_capstone\waste_classifier_capstone
```

---

### Lỗi: "Raw data directory not found"

**Nguyên nhân:** Chưa có dataset

**Giải pháp:**
```powershell
# Kiểm tra data
ls data/raw/

# Nếu không có, download dataset về
# Giải nén vào: data/raw/
# Cấu trúc: data/raw/cardboard/, data/raw/glass/, ...
```

---

### Lỗi: Out of memory (GPU/CPU)

**Giải pháp:** Giảm batch size trong `src/config.py`:
```python
BATCH_SIZE = 16  # Giảm từ 32
```

---

### Lỗi: YOLOv8 "Weights only load failed" (PyTorch 2.6+)

**Triệu chứng:**
```
_pickle.UnpicklingError: Weights only load failed...
GLOBAL torch.nn.modules.container.Sequential was not an allowed global
```

**Nguyên nhân:** PyTorch 2.6+ thay đổi cơ chế bảo mật khi load models

**Giải pháp:** ✅ **ĐÃ FIX** - Code đã được cập nhật tự động xử lý
- File `src/detection/detection_utils.py` đã được cập nhật
- Sử dụng `weights_only=False` cho YOLOv8 (trusted source)
- Không cần thao tác gì thêm

**Nếu vẫn lỗi:**
```powershell
# Downgrade PyTorch (không khuyến nghị)
pip install torch==2.5.0

# Hoặc kiểm tra version
python -c "import torch; print(torch.__version__)"
```

---

### Lỗi: Webcam không mở được

**Giải pháp:**
```powershell
# Thử camera index khác
python scripts/05_realtime_detection.py --camera 1

# Hoặc dùng video file thay vì webcam
python scripts/05_realtime_detection.py --video test_video.mp4
```

---

## 📖 Tài liệu chi tiết

- **Tổng quan**: `README.md`
- **Cấu trúc**: `STRUCTURE.md`
- **Hướng dẫn setup**: `docs/guides/GETTING_STARTED.md`
- **Lý thuyết**: `docs/theory/Week*.md`

---

## 🎯 Workflow đề xuất cho lần đầu

1. **Kiểm tra config**: `python main.py --config`
2. **Quick test**: `python main.py --quick` (15-30 phút)
3. **Xem kết quả**: Mở files trong `outputs/reports/`
4. **Nếu OK**: Chạy full pipeline `python main.py --all`

---

## ⏱️ Thời gian ước tính

| Mode | Thời gian | Mục đích |
|------|-----------|----------|
| `--quick` | 15-30 phút | Test nhanh |
| `--week 1` | 30-60 phút | Baseline CNN |
| `--week 2` | 1-2 giờ | Transfer Learning |
| `--all` | 2-3 giờ | Full pipeline |

*(Phụ thuộc vào CPU/GPU)*

---

## 🎉 Hoàn thành!

Sau khi chạy xong, bạn sẽ có:
- ✅ Trained models in `outputs/models/`
- ✅ Evaluation reports in `outputs/reports/`
- ✅ Training plots and metrics
- ✅ Optimized TFLite models (nếu chạy Week 4)

**Happy coding! 🚀**

