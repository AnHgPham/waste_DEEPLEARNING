# Week 3: Real-time Detection with YOLOv8

## 📚 Mục tiêu học tập

Tuần này bạn sẽ học:
1. **Object Detection** - Phát hiện và localize objects
2. **YOLOv8 Architecture** - State-of-the-art detector
3. **Integration** - Kết hợp detection + classification
4. **Real-time Processing** - Webcam inference

---

## 🎯 Lý thuyết Object Detection

### 1. Object Detection vs Classification

**Image Classification:**
```
Input: Ảnh → Output: Class label
"This is a plastic bottle"
```

**Object Detection:**
```
Input: Ảnh → Output: [Bounding boxes + Class labels]
"There are 2 objects:
  - plastic bottle at (x1, y1, x2, y2)
  - paper cup at (x3, y3, x4, y4)"
```

**Sự khác biệt:**
- 🎯 Detection: Tìm **WHERE** và **WHAT**
- 📍 Localization: Bounding box coordinates
- 🔢 Multiple objects trong 1 ảnh

### 2. YOLO (You Only Look Once)

**Ý tưởng chính:**
- Single-stage detector (chỉ 1 lần forward pass)
- Nhanh hơn two-stage (R-CNN, Faster R-CNN)
- Real-time: 30-60 FPS

**Evolution:**
```
YOLOv1 (2016) → YOLOv3 (2018) → YOLOv5 (2020) 
→ YOLOv8 (2023) ← We use this!
```

#### YOLOv8 Architecture

**Backbone:** CSPDarknet (feature extraction)
```
Input (640×640)
  ↓
Conv layers + CSP blocks
  ↓
Feature maps at 3 scales:
  - P3: 80×80 (small objects)
  - P4: 40×40 (medium objects)
  - P5: 20×20 (large objects)
```

**Neck:** PAN (Path Aggregation Network)
- Kết hợp multi-scale features
- Bottom-up + Top-down paths

**Head:** Detection head
- Phát hiện objects ở mỗi scale
- Output: [x, y, w, h, confidence, class_probs]

### 3. Key Concepts

#### a) Bounding Box Format

**YOLO format: (x_center, y_center, width, height)**
```python
# Normalized [0, 1]
bbox = [0.5, 0.5, 0.3, 0.4]
# → Object ở giữa ảnh, width=30%, height=40%
```

**Chuyển sang (x1, y1, x2, y2):**
```python
x1 = x_center - width/2
y1 = y_center - height/2
x2 = x_center + width/2
y2 = y_center + height/2
```

#### b) Confidence Score

**Công thức:**
```
Confidence = P(object) × IoU(pred, truth)
```

- `P(object)`: Xác suất có object trong box
- `IoU`: Intersection over Union với ground truth

**Threshold:** 
- Confidence > 0.5 → Keep detection
- Confidence < 0.5 → Discard

#### c) IoU (Intersection over Union)

**Công thức:**
```
IoU = Area(Intersection) / Area(Union)
```

**Ý nghĩa:**
- IoU = 1.0: Perfect overlap
- IoU > 0.5: Good detection
- IoU < 0.3: Poor detection

**Sử dụng:**
- Đánh giá quality của bounding box
- Non-Maximum Suppression (NMS)

#### d) Non-Maximum Suppression (NMS)

**Vấn đề:** Nhiều boxes detect cùng 1 object

**Giải pháp NMS:**
```
1. Sort boxes by confidence (high → low)
2. Keep box với confidence cao nhất
3. Remove boxes có IoU > threshold với box đã chọn
4. Repeat cho boxes còn lại
```

**Parameters:**
- `iou_threshold`: 0.45 (default)
- Cao → giữ nhiều boxes (nhiễu)
- Thấp → giữ ít boxes (miss objects)

---

## 🔗 Pipeline Integration

### Detection + Classification

**Flow:**
```
[Input: Camera frame]
        ↓
[YOLOv8 Detection]
    → Detects generic "objects"
    → Output: Bounding boxes
        ↓
[Crop detected regions]
        ↓
[MobileNetV2 Classification]
    → Classifies waste type
    → Output: Class labels
        ↓
[Draw results on frame]
    → Bounding box + Label
        ↓
[Display: Real-time video]
```

**Tại sao dùng 2 models?**

1. **YOLOv8:** Không train lại, dùng pretrained
   - Phát hiện "objects" tổng quát
   - Nhanh, real-time
   
2. **MobileNetV2:** Custom classifier đã train
   - Phân loại waste types (10 classes)
   - Accuracy cao (~90%)

**Alternative approach:**
- Train YOLOv8 trực tiếp trên waste dataset
- → Cần annotate bounding boxes (tốn thời gian)
- → Our approach nhanh hơn, leverage pretrained models

---

## 📂 Cấu trúc

```
Week3_Realtime_Detection/
├── README.md                    # File này
├── assignments/                 # Notebook
│   └── W3_Integration.ipynb
├── realtime_detection.py        # Main script
└── utils/
    ├── detection_utils.py       # YOLOv8 wrapper
    └── realtime_utils.py        # Classification + drawing
```

---

## 🚀 Cách sử dụng

### Python Script

```bash
# Dùng webcam mặc định
python Week3_Realtime_Detection/realtime_detection.py

# Dùng camera khác
python Week3_Realtime_Detection/realtime_detection.py --camera 1

# Dùng video file
python Week3_Realtime_Detection/realtime_detection.py --video path/to/video.mp4

# Chọn model classifier
python Week3_Realtime_Detection/realtime_detection.py --model mobilenetv2

# Hoặc main.py
python main.py --realtime
```

### Controls khi chạy

```
📹 Real-time Detection Window
├── 'q': Quit/Exit
├── 's': Save screenshot
└── 'p': Pause/Resume
```

---

## ⚡ Performance Optimization

### 1. Frame Skipping

**Vấn đề:** Processing mọi frame → slow

**Giải pháp:**
```python
if frame_count % 2 == 0:  # Process every 2nd frame
    detections = detect(frame)
```

**Trade-off:**
- ✅ 2x faster
- ⚠️ Less responsive

### 2. Model Optimization

**YOLOv8 variants:**
```
YOLOv8n: Nano    - 3.2M params - 80 FPS ← We use
YOLOv8s: Small   - 11M params  - 60 FPS
YOLOv8m: Medium  - 25M params  - 45 FPS
YOLOv8l: Large   - 43M params  - 30 FPS
```

**Recommendation:** YOLOv8n cho real-time

### 3. Resolution

**Input size:**
```
640×640: Best accuracy, slower
320×320: Lower accuracy, 4x faster
```

**Webcam resolution:**
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

### 4. Batching (GPU)

**Nếu có GPU:**
```python
# Process multiple crops cùng lúc
crops = [crop1, crop2, crop3, ...]
predictions = model.predict(batch(crops))
```

---

## 📊 Specs

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 hoặc equivalent
- RAM: 8GB
- Webcam: 720p

**Recommended:**
- GPU: NVIDIA GTX 1060 hoặc cao hơn
- RAM: 16GB
- Webcam: 1080p

### Performance Metrics

| Setup | FPS | Latency |
|-------|-----|---------|
| CPU only (i7) | 10-15 | ~100ms |
| GPU (GTX 1060) | 25-30 | ~40ms |
| GPU (RTX 3060) | 40-50 | ~25ms |

---

## 🔧 Troubleshooting

### Camera không mở được

```python
# Thử các camera index khác
python realtime_detection.py --camera 0  # Default
python realtime_detection.py --camera 1  # External
```

### FPS thấp

**Solutions:**
1. Giảm resolution: `--resolution 320`
2. Dùng YOLOv8n (nano)
3. Skip frames: process every 2-3 frames
4. Đóng các apps khác

### Detection không chính xác

**Adjustments:**
```python
# Tăng confidence threshold
YOLO_CONFIDENCE = 0.7  # default: 0.5

# Điều chỉnh NMS threshold
YOLO_IOU_THRESHOLD = 0.3  # default: 0.45
```

---

## 🎓 Kiến thức học được

Sau Week 3, bạn sẽ hiểu:

✅ Object detection vs classification  
✅ YOLO architecture và cách hoạt động  
✅ IoU, NMS, confidence score  
✅ Kết hợp detection + classification  
✅ Real-time processing techniques  
✅ Optimization strategies cho FPS cao  

---

## 🔗 Tài liệu tham khảo

1. **YOLO:**
   - "YOLOv8: State-of-the-art Object Detection" - Ultralytics
   - YOLOv1 paper: "You Only Look Once" (2016)

2. **Object Detection:**
   - CS231n: Detection and Segmentation
   - "Object Detection in 20 Years: A Survey" (2019)

3. **Real-time Processing:**
   - OpenCV Documentation
   - "Real-Time Object Detection with YOLO"

---

**Previous:** [Week 2 - Transfer Learning](../Week2_Transfer_Learning/README.md)  
**Next:** [Week 4 - Model Optimization](../Week4_Deployment/README.md)

