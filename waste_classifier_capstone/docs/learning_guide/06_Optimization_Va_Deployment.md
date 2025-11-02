# 🚀 OPTIMIZATION VÀ DEPLOYMENT

**Thời gian:** 1 giờ
**Mục tiêu:** Hiểu cách optimize model và deploy lên production

---

## 📌 1. TẠI SAO CẦN OPTIMIZATION? (10 phút)

### **Problem:**

```
Trained Model (Keras):
  File: mobilenetv2_final.keras
  Size: 25.0 MB
  Format: Keras SavedModel
  Precision: FP32 (32-bit floating point)

Deployment Targets:
  📱 Smartphone (Android/iOS)
  🥧 Raspberry Pi
  🖥️ Edge devices (limited resources)

Challenges:
  ✗ 25 MB quá lớn cho mobile apps
  ✗ FP32 inference chậm trên mobile CPU
  ✗ Tốn battery, memory
  ✗ Không phù hợp real-time applications
```

### **Solution: Model Optimization**

```
Original Model (Keras):
  Size: 25.0 MB
  Precision: FP32
  Inference: ~200ms (mobile CPU)

Optimized Model (TFLite):
  FP32:
    Size: 9.84 MB (-60.7%)  ✅
    Accuracy: 93.90% (no loss!)
    Inference: ~100ms (-50%)

  INT8 (Quantized):
    Size: 2.94 MB (-88.3%!)  🔥
    Accuracy: 93.20% (-0.70%)  ← Acceptable!
    Inference: ~50ms (-75%!)  ← VERY FAST!
```

---

## 🔧 2. TENSORFLOW LITE (TFLite) (15 phút)

### **A. TFLite là gì?**

**TensorFlow Lite = Framework cho mobile & edge deployment**

```
TensorFlow:
  ✓ Training models (powerful)
  ✗ Large size
  ✗ Desktop/server only

TensorFlow Lite:
  ✓ Inference only (lightweight)
  ✓ Optimized for mobile/edge
  ✓ Small size, fast inference
  ✗ Cannot train models
```

**Platforms supported:**

```
Mobile:
  📱 Android (Java/Kotlin)
  📱 iOS (Swift/Obj-C)

Edge:
  🥧 Raspberry Pi (Python)
  🔌 Coral Edge TPU
  🎮 Jetson Nano

Web:
  🌐 TensorFlow.js
```

---

### **B. Conversion Process**

```
Keras Model (.keras)
    ↓
[TFLite Converter]
    ↓
TFLite Model (.tflite)
    ↓
[Deploy]
    ↓
Mobile/Edge Device
```

**Code:**

```python
import tensorflow as tf

# 1. Load Keras model
model = tf.keras.models.load_model('mobilenetv2_final.keras')

# 2. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save TFLite model
with open('mobilenetv2_fp32.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

---

### **C. TFLite Benefits**

```
1. Size Reduction:
   Keras: 25.0 MB
   TFLite: 9.84 MB
   → 60.7% smaller!

   Why?
   ✓ Remove training ops (backprop, etc.)
   ✓ Optimize graph (fuse ops)
   ✓ Compress weights

2. Speed Improvement:
   FP32 Keras: ~200ms
   FP32 TFLite: ~100ms
   → 2x faster!

   Why?
   ✓ Optimized kernels for mobile
   ✓ Graph optimization
   ✓ Hardware acceleration (NNAPI, GPU)

3. Memory Efficiency:
   ✓ Smaller model → Less RAM
   ✓ Optimized inference → Less peak memory
   ✓ Better for resource-constrained devices
```

---

## ⚡ 3. QUANTIZATION (20 phút)

### **A. Quantization là gì?**

**Quantization = Giảm precision của weights và activations**

```
Standard Model (FP32):
  Weights: 32-bit floating point
  Range: ±3.4 × 10^38
  Precision: ~7 decimal digits

  Example weight: 0.123456789 (32 bits)

Quantized Model (INT8):
  Weights: 8-bit integer
  Range: -128 to 127
  Precision: 256 values

  Example weight: 31 (8 bits)
  → Maps to ~0.123 after dequantization

Reduction: 32 bits → 8 bits = 4x smaller!
```

---

### **B. How Quantization Works**

**Linear Quantization Formula:**

```
Quantization (FP32 → INT8):
  Q = round((F - zero_point) / scale)

  Where:
    F = FP32 value
    Q = INT8 value
    scale = (F_max - F_min) / 255
    zero_point = INT8 value for F = 0

Example:
  FP32 range: [-1.0, 1.0]
  scale = (1.0 - (-1.0)) / 255 = 0.00784
  zero_point = 0

  Quantize 0.5:
    Q = round(0.5 / 0.00784) = round(63.8) = 64

  Quantize -0.3:
    Q = round(-0.3 / 0.00784) = round(-38.3) = -38
```

**Dequantization (INT8 → FP32):**

```
Dequantization:
  F = scale * (Q - zero_point)

Example:
  Q = 64
  F = 0.00784 * 64 = 0.5  ✓

  Q = -38
  F = 0.00784 * -38 = -0.298 ≈ -0.3  ✓ (small error)
```

---

### **C. Types of Quantization**

#### **1. Post-Training Quantization (PTQ)**

```
Post-Training Quantization = Quantize SAU KHI train xong

Process:
  1. Train model với FP32 (normal)
  2. Model trained → 93.90% accuracy
  3. Convert to INT8 using TFLite converter
  4. TFLite model → 93.20% accuracy (-0.70%)

Pros:
  ✅ Easy (no retraining)
  ✅ Fast (minutes to convert)
  ✅ Good accuracy (usually <1% loss)

Cons:
  ⚠ Slight accuracy drop
  ⚠ May need calibration data
```

**Code:**

```python
# Post-Training INT8 Quantization

# 1. Load model
model = tf.keras.models.load_model('mobilenetv2_final.keras')

# 2. Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Enable INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Provide representative dataset for calibration
def representative_dataset():
    for _ in range(100):
        # Sample from training data
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        yield [data]

converter.representative_dataset = representative_dataset

# 5. Set INT8 input/output (full integer model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# 6. Convert
tflite_model = converter.convert()

# 7. Save
with open('mobilenetv2_int8.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

#### **2. Quantization-Aware Training (QAT)**

```
Quantization-Aware Training = Train model VỚI quantization simulation

Process:
  1. Insert fake quantization ops during training
  2. Model learns to work with quantized weights
  3. Convert to INT8 after training
  4. Minimal accuracy loss (<0.5%)

Pros:
  ✅ Better accuracy than PTQ
  ✅ Model adapts to quantization

Cons:
  ⚠ Longer training time
  ⚠ More complex implementation
```

**Dự án này dùng PTQ (đơn giản hơn, đủ tốt!)**

---

### **D. Calibration Dataset**

**Tại sao cần calibration?**

```
Problem:
  Quantization cần biết FP32 range để compute scale

  Example:
    Layer 1 weights: [-0.5, 0.8]
    Layer 2 weights: [-2.3, 1.9]

    → Different ranges!
    → Need to measure ACTUAL ranges during inference

Solution: Calibration
  1. Run representative data through model
  2. Measure min/max values at each layer
  3. Compute optimal scale/zero_point
  4. Quantize with these parameters
```

**Code:**

```python
def representative_dataset():
    """
    Generate calibration data from training set.

    Should cover:
    - All classes
    - Various lighting conditions
    - Different object sizes
    """
    for images, _ in train_ds.take(100):  # 100 batches
        for img in images:
            # Yield single image
            yield [np.expand_dims(img, axis=0)]

converter.representative_dataset = representative_dataset
```

---

## 📊 4. OPTIMIZATION RESULTS (10 phút)

### **A. Size Comparison**

```
Original Keras Model:
  mobilenetv2_final.keras
  Size: 25.0 MB
  Precision: FP32

TFLite FP32:
  mobilenetv2_fp32.tflite
  Size: 9.84 MB
  Reduction: 60.7%
  Accuracy: 93.90% (NO LOSS!)

TFLite INT8:
  mobilenetv2_int8.tflite
  Size: 2.94 MB
  Reduction: 88.3%!
  Accuracy: 93.20% (-0.70%)

Visualization:
Keras    ████████████████████████░ 25.0 MB
TFLite   █████████░░░░░░░░░░░░░░░░  9.8 MB
INT8     ██░░░░░░░░░░░░░░░░░░░░░░░  2.9 MB
```

---

### **B. Accuracy Comparison**

```
Test Set Evaluation (1,974 images):

Keras FP32:     93.90% ← Baseline
TFLite FP32:    93.90% ← NO LOSS!
TFLite INT8:    93.20% ← -0.70%

Per-Class Comparison (INT8 vs Keras):

clothes:   96.50% → 96.30% (-0.20%)
shoes:     95.20% → 95.00% (-0.20%)
cardboard: 94.10% → 93.80% (-0.30%)
plastic:   93.40% → 92.90% (-0.50%)
paper:     92.80% → 92.50% (-0.30%)
biological:91.80% → 91.20% (-0.60%)
metal:     91.50% → 91.00% (-0.50%)
glass:     89.70% → 88.90% (-0.80%)  ← Largest drop
battery:   87.90% → 87.10% (-0.80%)
trash:     82.30% → 81.50% (-0.80%)

→ Accuracy drop CONSISTENT across classes
→ No catastrophic failures
→ Trade-off: 88.3% size reduction for 0.70% accuracy
```

---

### **C. Inference Speed**

```
Platform: Raspberry Pi 4 (4GB RAM)

Keras FP32:
  Inference: ~200ms/image
  FPS: 5

TFLite FP32:
  Inference: ~100ms/image
  FPS: 10  (2x faster!)

TFLite INT8:
  Inference: ~50ms/image
  FPS: 20  (4x faster!)

Platform: Android Phone (mid-range)

Keras FP32:
  Not supported (too large)

TFLite FP32:
  Inference: ~80ms/image
  FPS: 12

TFLite INT8 (with NNAPI):
  Inference: ~30ms/image
  FPS: 33  (real-time!)
```

---

## 🎯 5. DEPLOYMENT SCENARIOS (15 phút)

### **A. Raspberry Pi Deployment**

**Use case:** Waste sorting machine

```python
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# 1. Load TFLite model
interpreter = tflite.Interpreter(
    model_path='mobilenetv2_int8.tflite'
)
interpreter.allocate_tensors()

# 2. Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Load and preprocess image
image = Image.open('waste.jpg').resize((224, 224))
input_data = np.array(image, dtype=np.uint8)  # INT8 model expects uint8
input_data = np.expand_dims(input_data, axis=0)

# 4. Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# 5. Dequantize output (INT8 → probabilities)
scale, zero_point = output_details[0]['quantization']
output = scale * (output.astype(np.float32) - zero_point)

# 6. Get prediction
class_idx = np.argmax(output)
confidence = output[class_idx]

print(f"Prediction: {CLASS_NAMES[class_idx]}")
print(f"Confidence: {confidence:.2%}")
```

**Hardware requirements:**

```
Raspberry Pi 4:
  CPU: Quad-core ARM Cortex-A72
  RAM: 2GB+ recommended
  Storage: 8GB+ SD card
  Camera: Raspberry Pi Camera Module v2

Performance:
  INT8 model: ~50ms/image
  FPS: 20 (sufficient for sorting!)
```

---

### **B. Mobile App Deployment**

**Android Example:**

```kotlin
// 1. Load TFLite model from assets
private val tflite = Interpreter(loadModelFile())

// 2. Prepare input
val inputBuffer = ByteBuffer.allocateDirect(1 * 224 * 224 * 3)
inputBuffer.order(ByteOrder.nativeOrder())

// Fill buffer with image pixels (uint8)
bitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
for (pixel in pixels) {
    inputBuffer.put((pixel shr 16 and 0xFF).toByte())  // R
    inputBuffer.put((pixel shr 8 and 0xFF).toByte())   // G
    inputBuffer.put((pixel and 0xFF).toByte())         // B
}

// 3. Prepare output
val outputBuffer = ByteBuffer.allocateDirect(10)  // 10 classes

// 4. Run inference
tflite.run(inputBuffer, outputBuffer)

// 5. Get prediction
val probabilities = FloatArray(10)
outputBuffer.rewind()
for (i in 0..9) {
    // Dequantize
    val q = outputBuffer.get().toInt() and 0xFF
    probabilities[i] = scale * (q - zeroPoint)
}

val prediction = probabilities.indices.maxByOrNull { probabilities[it] }!!
```

**App size impact:**

```
Without TFLite model:
  APK size: ~10 MB

With Keras model (not possible):
  APK size: ~35 MB (too large!)

With TFLite FP32:
  APK size: ~20 MB (acceptable)

With TFLite INT8:
  APK size: ~13 MB (excellent!)
```

---

### **C. Cloud vs Edge Deployment**

**Comparison:**

| Aspect | Cloud | Edge (TFLite) |
|--------|-------|---------------|
| **Latency** | 100-500ms | 30-100ms |
| **Internet** | Required | Not required |
| **Privacy** | Data sent to cloud | Data stays local |
| **Cost** | API calls fee | One-time hardware |
| **Scalability** | Easy | Limited by device |
| **Offline** | ✗ | ✅ |
| **Real-time** | ⚠ Depends | ✅ |

**When to use Edge (TFLite):**

```
✅ Real-time requirements (sorting machine)
✅ Privacy concerns (medical, personal)
✅ Offline environments (remote areas)
✅ Low latency critical (<100ms)
✅ Cost-sensitive (avoid API fees)

Example: Waste sorting kiosk
  - Users drop waste
  - Camera captures image
  - TFLite model classifies (50ms)
  - Display result immediately
  - No internet needed!
```

**When to use Cloud:**

```
✅ Complex models (too large for mobile)
✅ Frequent updates needed
✅ High accuracy critical
✅ Centralized data collection

Example: Waste analytics platform
  - Users upload photos
  - Cloud processes with large model
  - Store results in database
  - Generate analytics reports
```

---

## 🎓 TỔNG KẾT

### **Key Concepts:**

1. **TensorFlow Lite** = Framework for mobile/edge deployment
2. **Quantization** = FP32 → INT8 (4x smaller, 4x faster)
3. **Post-Training Quantization** = Quantize after training
4. **Calibration** = Measure activation ranges for optimal quantization

### **Optimization Results:**

```
Keras Model:
  Size: 25.0 MB
  Accuracy: 93.90%
  Inference: ~200ms

TFLite INT8:
  Size: 2.94 MB (-88.3%!)
  Accuracy: 93.20% (-0.70%)
  Inference: ~50ms (-75%!)

Trade-off: Excellent!
  → Huge size/speed gain
  → Minimal accuracy loss
```

### **Deployment:**

```
Best for Mobile/Edge:
  ✅ TFLite INT8 (2.94 MB)
  ✅ 20 FPS on Raspberry Pi
  ✅ 33 FPS on Android (NNAPI)
  ✅ Offline capable
  ✅ Privacy-preserving
```

---

## ✅ CHECKPOINT

**Bạn cần hiểu được:**

- [ ] TFLite là framework cho mobile/edge deployment
- [ ] Quantization giảm FP32 → INT8 (4x smaller)
- [ ] Post-Training Quantization quantize sau khi train
- [ ] Calibration dataset cần để compute scale
- [ ] TFLite INT8: 2.94 MB, 93.20% accuracy
- [ ] 88.3% size reduction, 0.70% accuracy loss
- [ ] Inference 4x nhanh hơn trên mobile
- [ ] Suitable cho Raspberry Pi, Android, iOS

**Nếu OK →** Tiếp tục `07_Tong_Ket_Va_On_Tap.md` 🚀
