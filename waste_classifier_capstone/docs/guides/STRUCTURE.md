# Cấu trúc Dự án - Waste Classification Capstone

## 🎓 Triết lý Học Thuật

Dự án này được tổ chức theo **3-tier approach** để đáp ứng cả nhu cầu học tập và production:

### 📖 Tier 1: Theory (README.md)
- Lý thuyết toán học và concepts
- Giải thích architecture
- Best practices
- Tài liệu tham khảo

### 🐍 Tier 2: Production Scripts (.py)
- Ready-to-run Python files
- Command-line interface
- Automated pipelines
- Error handling

### 📓 Tier 3: Interactive Learning (notebooks - optional)
- Step-by-step execution
- Visualizations
- Experiments
- Output cells

---

## 📁 Cấu trúc Chi tiết

```
waste_classifier_capstone/
│
├── 📄 main.py                       # CLI chính - chạy toàn bộ pipeline
├── 📄 evaluate_model.py             # Đánh giá models
├── 📄 config.py                     # Tập trung tất cả config
├── 📄 requirements.txt              # Dependencies
├── 📄 .gitignore                    # Git rules
│
├── 📖 README.md                     # Project overview
├── 📖 PROJECT_SUMMARY.md            # Technical summary
├── 📖 GETTING_STARTED.md            # Setup guide
└── 📖 STRUCTURE.md                  # File này
│
├── 📂 Week1_Data_and_Baseline/      # TUẦN 1: Dữ liệu & Baseline CNN
│   │
│   ├── 📖 README.md                 # ✅ LÝ THUYẾT: CNN, Augmentation, Training
│   │                                #    - Convolution layers
│   │                                #    - Batch Normalization
│   │                                #    - Dropout regularization
│   │                                #    - Data augmentation techniques
│   │
│   ├── 🐍 data_exploration.py       # ✅ SCRIPT: Khám phá dataset
│   │                                #    Usage: python Week1_Data_and_Baseline/data_exploration.py
│   │
│   ├── 🐍 preprocessing.py          # ✅ SCRIPT: Tiền xử lý, split data
│   │                                #    Usage: python Week1_Data_and_Baseline/preprocessing.py
│   │
│   ├── 🐍 baseline_training.py      # ✅ SCRIPT: Train baseline CNN
│   │                                #    Usage: python Week1_Data_and_Baseline/baseline_training.py --epochs 30
│   │
│   ├── 📂 assignments/              # ✅ NOTEBOOKS (optional cho học tập)
│   │   ├── W1_Data_Exploration.ipynb
│   │   ├── W1_Preprocessing.ipynb
│   │   └── W1_Baseline_CNN.ipynb
│   │
│   └── 📂 utils/                    # Helper functions
│       ├── data_utils.py            # split_data(), create_data_generators()
│       └── model_utils.py           # build_baseline_model()
│
├── 📂 Week2_Transfer_Learning/      # TUẦN 2: Transfer Learning
│   │
│   ├── 📖 README.md                 # ✅ LÝ THUYẾT: Transfer Learning, MobileNetV2
│   │                                #    - Why transfer learning works
│   │                                #    - MobileNetV2 architecture
│   │                                #    - Depthwise separable convolutions
│   │                                #    - Two-phase training strategy
│   │                                #    - Fine-tuning best practices
│   │
│   ├── 🐍 transfer_learning.py      # ✅ SCRIPT: Phase 1 + Phase 2 training
│   │                                #    Usage: python Week2_Transfer_Learning/transfer_learning.py
│   │                                #           --phase1-epochs 15 --phase2-epochs 10
│   │
│   ├── 📂 assignments/              # ✅ NOTEBOOKS (optional)
│   │   ├── W2_Feature_Extraction.ipynb
│   │   └── W2_Fine_Tuning.ipynb
│   │
│   └── 📂 utils/
│       └── model_utils.py           # build_transfer_model(), unfreeze_layers()
│
├── 📂 Week3_Realtime_Detection/     # TUẦN 3: Real-time Detection
│   │
│   ├── 📖 README.md                 # ✅ LÝ THUYẾT: YOLO, Object Detection
│   │                                #    - Object detection vs classification
│   │                                #    - YOLO architecture (YOLOv8)
│   │                                #    - IoU, NMS, confidence scores
│   │                                #    - Real-time optimization
│   │
│   ├── 🐍 realtime_detection.py     # ✅ SCRIPT: Webcam detection
│   │                                #    Usage: python Week3_Realtime_Detection/realtime_detection.py
│   │                                #           --model mobilenetv2 --camera 0
│   │
│   ├── 📂 assignments/              # ✅ NOTEBOOKS (optional)
│   │   └── W3_Integration.ipynb
│   │
│   └── 📂 utils/
│       ├── detection_utils.py       # load_yolo_model(), detect_objects()
│       └── realtime_utils.py        # classify_images(), draw_results()
│
├── 📂 Week4_Deployment/             # TUẦN 4: Optimization & Deployment
│   │
│   ├── 📖 README.md                 # ✅ LÝ THUYẾT: Quantization, TFLite
│   │                                #    - Model optimization techniques
│   │                                #    - INT8 quantization math
│   │                                #    - TensorFlow Lite conversion
│   │                                #    - Size-Accuracy trade-offs
│   │                                #    - Deployment strategies
│   │
│   ├── 🐍 model_optimization.py     # ✅ SCRIPT: Convert & quantize
│   │                                #    Usage: python Week4_Deployment/model_optimization.py
│   │                                #           --model mobilenetv2
│   │
│   ├── 📂 assignments/              # ✅ NOTEBOOKS (optional)
│   │   └── W4_Model_Optimization.ipynb
│   │
│   └── 📂 utils/
│       └── optimization_utils.py    # convert_to_tflite(), quantize_model()
│
├── 📂 data/
│   ├── raw/                         # Dataset gốc (download từ Kaggle)
│   │   ├── battery/
│   │   ├── biological/
│   │   ├── cardboard/
│   │   ├── clothes/
│   │   ├── glass/
│   │   ├── metal/
│   │   ├── paper/
│   │   ├── plastic/
│   │   ├── shoes/
│   │   └── trash/
│   │
│   └── processed/                   # Auto-generated sau preprocessing
│       ├── train/
│       ├── val/
│       └── test/
│
└── 📂 outputs/                      # Auto-generated outputs
    ├── models/                      # Trained models
    │   ├── baseline_final.keras
    │   ├── mobilenetv2_phase1.keras
    │   ├── mobilenetv2_final.keras
    │   ├── mobilenetv2_fp32.tflite
    │   └── mobilenetv2_int8.tflite
    │
    ├── reports/                     # Evaluation results
    │   ├── class_distribution.png
    │   ├── sample_images.png
    │   ├── baseline_confusion_matrix.png
    │   ├── baseline_training_history.png
    │   ├── mobilenetv2_confusion_matrix.png
    │   └── classification_report.txt
    │
    ├── logs/                        # Training logs
    └── screenshots/                 # Real-time detection screenshots
```

---

## 🚀 Cách Sử Dụng

### 1. Quick Start (Recommended)

```bash
# Chạy full pipeline
python main.py --quick
```

**Thực hiện:**
1. ✅ Data exploration
2. ✅ Preprocessing
3. ✅ Train MobileNetV2 (transfer learning)
4. ✅ Evaluate model

### 2. Run Từng Week

```bash
# Week 1: Data + Baseline
python main.py --week 1

# Week 2: Transfer Learning
python main.py --week 2

# Week 3: Real-time Detection
python main.py --week 3

# Week 4: Optimization
python main.py --week 4
```

### 3. Run Individual Scripts

```bash
# Khám phá dữ liệu
python Week1_Data_and_Baseline/data_exploration.py

# Train transfer learning
python Week2_Transfer_Learning/transfer_learning.py

# Real-time detection
python Week3_Realtime_Detection/realtime_detection.py --camera 0

# Optimize model
python Week4_Deployment/model_optimization.py --model mobilenetv2
```

### 4. Học Từ Lý Thuyết

```bash
# Đọc lý thuyết Week 1
cat Week1_Data_and_Baseline/README.md

# Hoặc dùng notebook (tương tác)
cd Week1_Data_and_Baseline/assignments
jupyter notebook W1_Data_Exploration.ipynb
```

---

## 📚 Quy trình Học Tập Đề xuất

### 🎓 **Approach 1: Academic (Deep Learning)**

Nếu bạn muốn **hiểu sâu lý thuyết**:

```
1. 📖 Đọc README.md của Week
   → Hiểu concepts, toán học, architecture
   
2. 📓 Chạy Jupyter Notebook
   → Thực hành từng bước, xem visualizations
   
3. 🐍 Chạy Python Script
   → Verify kết quả, chạy full pipeline
   
4. 🔬 Thử nghiệm
   → Modify hyperparameters, compare results
```

**Timeline:** 1-2 tuần/week

### ⚡ **Approach 2: Production (Fast Track)**

Nếu bạn muốn **kết quả nhanh**:

```
1. 📖 Skim README.md
   → Hiểu overview và key concepts
   
2. 🐍 Chạy Python Scripts
   → Run full pipeline
   
3. 📊 Review Outputs
   → Check accuracy, confusion matrix
   
4. 📓 (Optional) Notebooks
   → Chỉ xem khi cần debug hoặc hiểu sâu
```

**Timeline:** 2-3 ngày/week

---

## 🎯 Sự Khác Biệt

### README.md (Lý thuyết)

**Nội dung:**
- Mathematical formulations
- Architecture diagrams
- Conceptual explanations
- Best practices
- References

**Khi nào đọc:**
- Trước khi code
- Khi gặp khó khăn
- Để hiểu "WHY"

### Python Scripts (Production)

**Đặc điểm:**
- Hoàn chỉnh, ready-to-run
- Error handling
- Progress logging
- Argument parsing
- Modular, reusable

**Khi nào dùng:**
- Train models
- Quick experiments
- Production deployment
- CI/CD pipelines

### Jupyter Notebooks (Learning)

**Đặc điểm:**
- Step-by-step execution
- Inline visualizations
- Markdown explanations
- Interactive output
- Easy debugging

**Khi nào dùng:**
- First time learning
- Experiments
- Teaching/presentations
- Debugging

---

## 💡 Tips

### 1. Đối với Sinh viên

✅ **Đọc README.md trước** → Hiểu lý thuyết  
✅ **Chạy notebooks** → Thực hành và visualize  
✅ **Đọc Python scripts** → Xem production code  
✅ **Ghi chú observations** → Cho báo cáo  

### 2. Đối với Developers

✅ **Skim README.md** → Hiểu concepts  
✅ **Chạy Python scripts** → Fast results  
✅ **Review outputs** → Verify quality  
✅ **Modify scripts** → Adapt cho use case  

### 3. Đối với Giảng viên

✅ **README.md** → Teaching materials  
✅ **Notebooks** → In-class demos  
✅ **Scripts** → Homework grading  
✅ **config.py** → Easy assignments modification  

---

## 🔍 File Lookup Quick Reference

| Task | File |
|------|------|
| **Lý thuyết CNN** | `Week1_Data_and_Baseline/README.md` |
| **Train baseline** | `python Week1_Data_and_Baseline/baseline_training.py` |
| **Lý thuyết Transfer Learning** | `Week2_Transfer_Learning/README.md` |
| **Train MobileNetV2** | `python Week2_Transfer_Learning/transfer_learning.py` |
| **Lý thuyết YOLO** | `Week3_Realtime_Detection/README.md` |
| **Real-time demo** | `python Week3_Realtime_Detection/realtime_detection.py` |
| **Lý thuyết Quantization** | `Week4_Deployment/README.md` |
| **Optimize model** | `python Week4_Deployment/model_optimization.py` |
| **Config tất cả** | `config.py` |
| **Evaluate model** | `python evaluate_model.py --model mobilenetv2` |

---

## ✨ Kết luận

Cấu trúc này đảm bảo:

✅ **Học thuật:** READMEs với lý thuyết đầy đủ  
✅ **Production:** Python scripts ready-to-run  
✅ **Tương tác:** Notebooks cho learning  
✅ **Modular:** Utils reusable  
✅ **Reproducible:** config.py centralized  
✅ **Documented:** Docstrings chuẩn  

**→ Phù hợp cho cả đồ án học thuật VÀ ứng dụng thực tế!**

