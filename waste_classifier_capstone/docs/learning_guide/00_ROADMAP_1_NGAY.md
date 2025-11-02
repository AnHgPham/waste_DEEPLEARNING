# 🎓 LỘ TRÌNH HỌC DỰ ÁN TRONG 1 NGÀY

**Mục tiêu:** Hiểu TOÀN BỘ dự án Waste Classification từ cơ bản đến nâng cao

**Thời gian:** 8-10 giờ (1 ngày học tập trung)

---

## ⏰ LỊCH HỌC CHI TIẾT

### 📚 **BUỔI SÁNG (8:00 - 12:00) - 4 giờ**

#### **8:00 - 9:00: CƠ BẢN DEEP LEARNING (1 giờ)** ⭐⭐⭐⭐⭐
**File:** `01_Deep_Learning_Co_Ban.md`

**Nội dung:**
- [ ] Neural Network là gì? (10 phút)
- [ ] Forward Propagation (15 phút)
- [ ] Backpropagation (15 phút)
- [ ] Loss Function, Optimizer (10 phút)
- [ ] Overfitting vs Underfitting (10 phút)

**Checkpoint:** Hiểu được neural network hoạt động như thế nào

---

#### **9:00 - 10:30: CNN & COMPUTER VISION (1.5 giờ)** ⭐⭐⭐⭐⭐
**File:** `02_CNN_Va_Computer_Vision.md`

**Nội dung:**
- [ ] Convolution là gì? (20 phút)
- [ ] Pooling, Stride, Padding (15 phút)
- [ ] CNN Architecture (30 phút)
- [ ] Image Classification workflow (15 phút)
- [ ] Data Augmentation (10 phút)

**Checkpoint:** Hiểu được CNN extract features từ ảnh

---

#### **10:30 - 11:00: NGHỈ GIẢI LAO** ☕
- Review lại ghi chú
- Làm bài tập nhỏ (nếu có)

---

#### **11:00 - 12:00: BASELINE CNN (1 giờ)** ⭐⭐⭐⭐
**File:** `03_Baseline_CNN_Chi_Tiet.md`

**Nội dung:**
- [ ] Baseline architecture trong dự án (15 phút)
- [ ] Code walkthrough: `src/models/baseline.py` (20 phút)
- [ ] Training process (15 phút)
- [ ] Kết quả: 79.59% accuracy (10 phút)

**Checkpoint:** Hiểu code baseline CNN trong dự án

---

### 🍜 **NGHỈ TRƯA (12:00 - 13:00)**

---

### 📘 **BUỔI CHIỀU (13:00 - 18:00) - 5 giờ**

#### **13:00 - 14:30: TRANSFER LEARNING (1.5 giờ)** ⭐⭐⭐⭐⭐
**File:** `04_Transfer_Learning_Chi_Tiet.md`

**Nội dung:**
- [ ] Transfer Learning là gì? Tại sao cần? (20 phút)
- [ ] ImageNet pre-training (15 phút)
- [ ] Feature Extraction vs Fine-Tuning (25 phút)
- [ ] MobileNetV2 architecture (30 phút)

**Checkpoint:** Hiểu tại sao Transfer Learning tốt hơn Baseline

---

#### **14:30 - 16:00: MOBILENETV2 TRONG DỰ ÁN (1.5 giờ)** ⭐⭐⭐⭐⭐
**File:** `05_MobileNetV2_Thuc_Hanh.md`

**Nội dung:**
- [ ] Code walkthrough: `src/models/transfer.py` (30 phút)
- [ ] Two-phase training (30 phút)
- [ ] Kết quả: 93.90% accuracy (15 phút)
- [ ] So sánh Baseline vs MobileNetV2 (15 phút)

**Checkpoint:** Hiểu toàn bộ quá trình training MobileNetV2

---

#### **16:00 - 16:30: NGHỈ GIẢI LAO** ☕
- Review confusion matrix
- Xem visualization kết quả

---

#### **16:30 - 17:30: OPTIMIZATION & DEPLOYMENT (1 giờ)** ⭐⭐⭐
**File:** `06_Optimization_Va_Deployment.md`

**Nội dung:**
- [ ] TensorFlow Lite là gì? (15 phút)
- [ ] Quantization (INT8 vs FP32) (20 phút)
- [ ] Model optimization process (15 phút)
- [ ] Deployment use cases (10 phút)

**Checkpoint:** Hiểu cách deploy model lên production

---

#### **17:30 - 18:00: TỔNG KẾT & ÔN TẬP (30 phút)** ⭐⭐⭐⭐⭐
**File:** `07_Tong_Ket_Va_On_Tap.md`

**Nội dung:**
- [ ] Review toàn bộ flow (10 phút)
- [ ] Cheatsheet các khái niệm quan trọng (10 phút)
- [ ] Q&A các câu hỏi khó (10 phút)

---

## 📋 CHECKLIST HỌC

### **Level 1: Cơ Bản (BẮT BUỘC)** ✅
- [ ] Hiểu Neural Network cơ bản
- [ ] Hiểu Convolution hoạt động thế nào
- [ ] Hiểu flow: Data → Model → Training → Evaluation
- [ ] Biết Baseline CNN architecture
- [ ] Biết Transfer Learning là gì

### **Level 2: Trung Bình (QUAN TRỌNG)** ⭐
- [ ] Hiểu Forward & Backward propagation
- [ ] Hiểu CNN architecture layers
- [ ] Hiểu Training callbacks (EarlyStopping, ReduceLR)
- [ ] Hiểu Two-phase training
- [ ] Hiểu Model Capacity & Ceiling

### **Level 3: Nâng Cao (NÊN HỌC)** 🚀
- [ ] Hiểu MobileNetV2 architecture chi tiết
- [ ] Hiểu Depthwise Separable Convolution
- [ ] Hiểu Quantization (INT8)
- [ ] Hiểu Receptive Field
- [ ] Hiểu Residual Connections

---

## 🎯 MỤC TIÊU ĐÁNH GIÁ

### **Sau 1 ngày học, bạn cần:**

#### **Trả lời được:**
- ✅ CNN khác Neural Network thường thế nào?
- ✅ Tại sao Transfer Learning tốt hơn train from scratch?
- ✅ MobileNetV2 đạt 93.90% còn Baseline chỉ 79.59% vì sao?
- ✅ Two-phase training là gì? Tại sao cần?
- ✅ Quantization làm model nhỏ hơn thế nào?

#### **Giải thích code được:**
- ✅ `src/models/baseline.py` - Baseline CNN
- ✅ `src/models/transfer.py` - MobileNetV2
- ✅ `scripts/04_transfer_learning.py` - Training process
- ✅ `src/deployment/optimize.py` - Optimization

#### **Vẽ được:**
- ✅ CNN architecture diagram
- ✅ Transfer Learning workflow
- ✅ Training history graphs
- ✅ Confusion matrix interpretation

---

## 💡 TIPS HỌC HIỆU QUẢ

### **1. Học Theo Thứ Tự**
```
Cơ bản → Trung bình → Nâng cao
ĐỪNG bỏ qua bước nào!
```

### **2. Kết Hợp Lý Thuyết + Code**
```
Đọc file .md → Xem code → Chạy thử script
```

### **3. Ghi Chú Tay**
```
Viết lại bằng chữ mình → Nhớ lâu hơn
```

### **4. Vẽ Diagram**
```
Vẽ architecture, flow charts → Visualize concepts
```

### **5. Giải Thích Cho Người Khác**
```
Nếu giải thích được cho bạn → Bạn đã hiểu!
```

---

## 🆘 KHI GẶP KHÓ KHĂN

### **Không hiểu khái niệm:**
→ Xem lại phần "Ví dụ đơn giản" trong file .md
→ Xem video YouTube về topic đó (5-10 phút)

### **Không hiểu code:**
→ Đọc comments trong code
→ Chạy từng dòng để xem output
→ Xem file README.md trong folder đó

### **Quá nhiều thông tin:**
→ Tập trung vào **Level 1 (Cơ Bản)** trước
→ Level 2 & 3 học sau nếu còn thời gian

---

## 📚 THỨ TỰ ĐỌC FILES

```
1. 00_ROADMAP_1_NGAY.md              (File này - 5 phút)
2. 01_Deep_Learning_Co_Ban.md        (1 giờ)
3. 02_CNN_Va_Computer_Vision.md      (1.5 giờ)
4. 03_Baseline_CNN_Chi_Tiet.md       (1 giờ)
5. 04_Transfer_Learning_Chi_Tiet.md  (1.5 giờ)
6. 05_MobileNetV2_Thuc_Hanh.md       (1.5 giờ)
7. 06_Optimization_Va_Deployment.md  (1 giờ)
8. 07_Tong_Ket_Va_On_Tap.md          (30 phút)
9. 08_Cau_Hoi_Thuong_Gap.md          (Tham khảo)
10. 09_Cheatsheet.md                 (Ôn nhanh)
```

---

## ✅ HOÀN THÀNH

Sau khi học xong tất cả:

- [ ] Làm bài tập cuối (nếu có)
- [ ] Review Cheatsheet 1 lần nữa
- [ ] Giải thích dự án cho bạn/gia đình
- [ ] Ngủ đủ giấc trước ngày thi! 😴

---

## 🎉 TIN TƯỞNG VÀO BẢN THÂN!

**Bạn CÓ THỂ làm được!**

Dự án này có hệ thống tài liệu đầy đủ. Chỉ cần:
- ✅ Học theo roadmap
- ✅ Kiên nhẫn
- ✅ Thực hành code
- ✅ Hỏi khi không hiểu

**→ Bạn sẽ HIỂU và QUA MÔN! 💪**

---

**BẮT ĐẦU NGAY:** Mở file `01_Deep_Learning_Co_Ban.md`

**Chúc may mắn! 🍀**
