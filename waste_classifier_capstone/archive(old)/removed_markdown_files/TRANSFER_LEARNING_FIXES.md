# Transfer Learning Performance Fixes

## 🔴 VẤN ĐỀ BAN ĐẦU (ORIGINAL PROBLEMS)

### 1. **Double Preprocessing Bug** ❌
**Lỗi nghiêm trọng nhất!** Dữ liệu bị chuẩn hóa 2 lần:

- `preprocessing.py` line 126: `Rescaling(1./255)` chuyển ảnh từ [0, 255] → [0, 1]
- `transfer.py` line 44: `mobilenet_v2.preprocess_input()` mong đợi [0, 255] và chuyển thành [-1, 1]
- **Kết quả**: MobileNetV2 nhận input trong range [0, 1] thay vì [0, 255], dẫn đến normalization sai hoàn toàn!

**Hậu quả**:
- Validation loss cực cao (9+)
- Accuracy rất thấp (~27-44%)
- Model không thể học được pattern đúng

### 2. **BatchNormalization Always Frozen** ❌
`transfer.py` line 47: `base_model(x, training=False)` được hardcode

**Hậu quả**:
- Ngay cả khi fine-tuning (phase 2), BatchNorm layers trong base model vẫn dùng statistics từ ImageNet
- Không adapt được với waste classification dataset
- Mất đi lợi ích của fine-tuning

### 3. **Suboptimal Hyperparameters** ⚠️
- Learning rate quá cao cho transfer learning
- Augmentation quá yếu
- Classification head đơn giản
- Dropout rate hơi cao

---

## ✅ CÁC THAY ĐỔI ĐÃ THỰC HIỆN (FIXES APPLIED)

### 1. **Fixed Double Preprocessing** ✓

**`src/data/preprocessing.py`**:
```python
# REMOVED: normalization_layer = layers.Rescaling(1./255)
# REMOVED: train_ds.map(lambda x, y: (normalization_layer(x), y))

# NOW: Keep images in [0, 255] range for MobileNetV2
# MobileNetV2's preprocess_input will handle normalization to [-1, 1]
```

**`src/data/loader.py`**:
```python
# Added 'normalize' parameter (default: True for baseline, False for transfer learning)
def load_dataset(..., normalize=True):
    if normalize:
        # Only normalize for baseline CNN models
        dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
```

**`src/models/transfer.py`**:
```python
# Input now correctly in [0, 255] range
x = keras.applications.mobilenet_v2.preprocess_input(inputs)  # Converts to [-1, 1]
```

### 2. **Fixed BatchNormalization Training Mode** ✓

**`src/models/transfer.py`**:
```python
# BEFORE:
x = base_model(x, training=False)  # ❌ Always frozen

# AFTER:
x = base_model(x, training=not freeze_base)  # ✅ Adapts during fine-tuning
# Phase 1: training=False (frozen, use ImageNet statistics)
# Phase 2: training=True (fine-tuning, update BatchNorm statistics)
```

### 3. **Improved Hyperparameters** ✓

**`src/config.py`**:

**Learning Rates** (giảm để stable hơn):
```python
# BEFORE:
LEARNING_RATE_TRANSFER_PHASE1 = 1e-3   # 0.001 (quá cao!)
LEARNING_RATE_TRANSFER_PHASE2 = 1e-4   # 0.0001

# AFTER:
LEARNING_RATE_TRANSFER_PHASE1 = 1e-4   # 0.0001 (stable)
LEARNING_RATE_TRANSFER_PHASE2 = 1e-5   # 0.00001 (very gentle fine-tuning)
```

**Epochs** (tăng để học tốt hơn):
```python
# BEFORE:
EPOCHS_TRANSFER_PHASE1 = 15
EPOCHS_TRANSFER_PHASE2 = 10

# AFTER:
EPOCHS_TRANSFER_PHASE1 = 20  # +5 epochs
EPOCHS_TRANSFER_PHASE2 = 15  # +5 epochs
```

**Classification Head** (tăng capacity):
```python
# BEFORE:
TRANSFER_DENSE_UNITS = 128     # Nhỏ
TRANSFER_DROPOUT_RATE = 0.5    # Hơi cao

# AFTER:
TRANSFER_DENSE_UNITS = 256     # Tăng gấp đôi
TRANSFER_DROPOUT_RATE = 0.3    # Giảm để model học tốt hơn
```

**Data Augmentation** (mạnh hơn):
```python
# BEFORE:
'rotation_factor': 0.1,     # ±36 degrees
'zoom_factor': 0.1,
'contrast_factor': 0.1,
'brightness_factor': 0.0,   # Disabled
'width_shift_factor': 0.0,  # Disabled
'height_shift_factor': 0.0, # Disabled

# AFTER:
'rotation_factor': 0.2,      # ±72 degrees
'zoom_factor': 0.2,
'contrast_factor': 0.2,
'brightness_factor': 0.1,    # ✅ Enabled
'width_shift_factor': 0.1,   # ✅ Enabled
'height_shift_factor': 0.1,  # ✅ Enabled
```

### 4. **Deeper Classification Head** ✓

**`src/models/transfer.py`**:
```python
# BEFORE: 1 dense layer
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

# AFTER: 2 dense layers (more capacity)
x = layers.Dense(256, activation='relu', name="Dense_1")(x)
x = layers.BatchNormalization(name="BatchNorm_1")(x)
x = layers.Dropout(0.3, name="Dropout_1")(x)

x = layers.Dense(128, activation='relu', name="Dense_2")(x)
x = layers.BatchNormalization(name="BatchNorm_2")(x)
x = layers.Dropout(0.3, name="Dropout_2")(x)

outputs = layers.Dense(num_classes, activation='softmax', name="Classifier")(x)
```

### 5. **Enhanced Data Augmentation Pipeline** ✓

**`src/data/preprocessing.py`**:
- Thêm `RandomBrightness` layer
- Thêm `RandomTranslation` layer
- Augmentation mạnh hơn để model generalize tốt hơn

---

## 🚀 CÁCH SỬ DỤNG (HOW TO USE)

### Xóa models cũ và train lại:
```bash
# Delete old models (they were trained with wrong preprocessing!)
rm outputs/models/mobilenetv2_phase1.keras
rm outputs/models/mobilenetv2_final.keras

# Train again with fixed code
python scripts/04_transfer_learning.py
```

### Hoặc với custom parameters:
```bash
python scripts/04_transfer_learning.py --phase1-epochs 25 --phase2-epochs 20 --unfreeze-layers 50
```

---

## 📊 KẾT QUẢ DỰ KIẾN (EXPECTED RESULTS)

### Trước khi fix:
- ❌ Phase 1 Val Accuracy: ~27-30%
- ❌ Phase 2 Val Accuracy: ~10-40% (rất không stable)
- ❌ Val Loss: 4-9 (cực cao)

### Sau khi fix:
- ✅ Phase 1 Val Accuracy: **~75-85%** (feature extraction)
- ✅ Phase 2 Val Accuracy: **~85-92%** (fine-tuning)
- ✅ Val Loss: **<1.0** (normal range)
- ✅ Training stable, không còn spike lớn

---

## 🔍 TÓM TẮT TECHNICAL

### Root Cause:
**Data preprocessing pipeline incompatible với MobileNetV2's expected input range**

### Solution:
1. Remove redundant `Rescaling(1./255)` từ data pipeline
2. Keep images in [0, 255] range
3. Let `mobilenet_v2.preprocess_input()` handle normalization to [-1, 1]
4. Fix BatchNorm training mode
5. Optimize hyperparameters và architecture

### Key Lesson:
**Khi dùng pretrained models, PHẢI kiểm tra input preprocessing requirements!**
- MobileNetV2: expects [0, 255] → normalizes to [-1, 1]
- ResNet/VGG: expects [0, 255] → normalizes with mean subtraction
- EfficientNet: expects [0, 255] → normalizes to [0, 1]
- Inception: expects [-1, 1]

**NEVER mix preprocessing methods!**

---

## 📝 NOTES

- Baseline CNN model vẫn cần `Rescaling(1./255)` vì nó được train từ scratch
- Chỉ transfer learning models mới bỏ rescaling và dùng model-specific preprocessing
- Luôn đọc documentation của pretrained model để hiểu input requirements!

---

**Fixed by:** AI Assistant  
**Date:** October 30, 2025  
**Impact:** Critical - fixes major bug causing complete model failure

