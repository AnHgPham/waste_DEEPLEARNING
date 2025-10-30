# Week 4: Model Optimization and Deployment

## 📚 Mục tiêu học tập

Tuần này bạn sẽ học:
1. **Model Optimization** - Giảm kích thước, tăng tốc độ
2. **Quantization** - INT8, FP16
3. **TensorFlow Lite** - Deployment cho mobile/edge
4. **Performance Trade-offs** - Size vs Accuracy

---

## 🎯 Lý thuyết Model Optimization

### 1. Tại sao cần Optimization?

**Vấn đề với models gốc:**
- 📦 Kích thước lớn (MobileNetV2: ~15 MB)
- 🐌 Inference chậm trên edge devices
- 💰 Memory footprint cao
- 🔋 Tiêu thụ năng lượng nhiều

**Target devices:**
- 📱 Smartphones (Android/iOS)
- 🥧 Raspberry Pi
- 🤖 Edge AI devices (Jetson Nano)
- 🌐 Web browsers (TensorFlow.js)

### 2. TensorFlow Lite

**Định nghĩa:** Framework để run TensorFlow models trên mobile/embedded devices.

**Đặc điểm:**
- ✅ Lightweight runtime
- ✅ Low latency
- ✅ Nhỏ gọn (< 1MB)
- ✅ Hardware acceleration support

**Conversion process:**
```
TensorFlow/Keras Model (.keras)
        ↓
TFLite Converter
        ↓
TFLite Model (.tflite)
```

### 3. Quantization

#### a) Floating Point (FP32)

**Standard format:**
- 32 bits per weight
- Range: ±3.4 × 10³⁸
- Precision: 7 decimal digits

**Example:**
```
Weight = 0.12345678 → Stored as FP32
```

#### b) INT8 Quantization

**Công thức chuyển đổi:**
```
Quantized = round(FP32_value / scale) + zero_point

Where:
- scale = (max - min) / 255
- zero_point: offset value
```

**Example:**
```
FP32 range: [-1.0, 1.0]
→ scale = 2.0 / 255 ≈ 0.0078
→ INT8 range: [-128, 127]

FP32 value: 0.5
→ INT8: round(0.5 / 0.0078) = 64
```

**Benefits:**
- 📦 4× smaller (32 bits → 8 bits)
- ⚡ 2-4× faster inference
- 💾 Less memory bandwidth
- 🔋 Lower power consumption

**Trade-off:**
- ⚠️ Accuracy drop: 1-3% (acceptable)

#### c) Post-Training Quantization

**3 types:**

**1. Dynamic Range Quantization**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```
- ✅ Dễ nhất, không cần data
- ⚠️ Activation vẫn FP32

**2. Full Integer Quantization**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = gen_func
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
```
- ✅ Weights + Activations đều INT8
- ✅ Maximum speed-up
- ⚠️ Cần representative dataset

**3. Float16 Quantization**
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```
- ✅ 2× smaller
- ✅ Minimal accuracy loss
- ⚠️ Chỉ trên GPU hỗ trợ FP16

#### Representative Dataset

**Mục đích:** Calibrate quantization ranges

**Yêu cầu:**
- ~100-500 samples từ training set
- Đại diện cho distribution of data
- Chạy qua model để capture activation ranges

**Code:**
```python
def representative_data_gen():
    for input_value in dataset.take(100):
        yield [input_value]

converter.representative_dataset = representative_data_gen
```

---

## 📊 Optimization Pipeline

### Step 1: Baseline (Keras Model)

```
mobilenetv2_final.keras
- Format: FP32
- Size: ~15 MB
- Accuracy: 90%
```

### Step 2: Convert to TFLite FP32

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

**Result:**
```
mobilenetv2_fp32.tflite
- Format: FP32
- Size: ~14 MB (slight reduction)
- Accuracy: ~90% (same)
- Speed: 1.2× faster
```

### Step 3: Quantize to INT8

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = gen_func
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
```

**Result:**
```
mobilenetv2_int8.tflite
- Format: INT8
- Size: ~900 KB (16× reduction!)
- Accuracy: ~87-88% (2-3% drop)
- Speed: 3-4× faster
```

---

## 📂 Cấu trúc

```
Week4_Deployment/
├── README.md                    # File này
├── assignments/                 # Notebook
│   └── W4_Model_Optimization.ipynb
├── model_optimization.py        # Main script
└── utils/
    └── optimization_utils.py    # Convert, quantize, evaluate
```

---

## 🚀 Cách sử dụng

### Python Script

```bash
# Optimize model
python Week4_Deployment/model_optimization.py

# Chọn model cụ thể
python Week4_Deployment/model_optimization.py --model mobilenetv2

# Hoặc main.py
python main.py --optimize --model mobilenetv2
```

### Output

Script sẽ tạo:
```
outputs/models/
├── mobilenetv2_fp32.tflite     # FP32 TFLite
└── mobilenetv2_int8.tflite     # INT8 quantized
```

Và in comparison:
```
Optimization Summary:
  Model Type          Size (MB)    Accuracy    Size Reduction
  ----------------------------------------------------------------
  Original Keras         14.50      0.9012      baseline
  TFLite FP32            13.80      0.9010      4.8%
  TFLite INT8             0.90      0.8788      93.8%
```

---

## ⚖️ Trade-offs Analysis

### Size vs Accuracy

| Model | Size | Accuracy | Use Case |
|-------|------|----------|----------|
| Keras FP32 | 15 MB | 90% | Server, High-end GPU |
| TFLite FP32 | 14 MB | 90% | Desktop, Laptop |
| TFLite INT8 | 900 KB | 87-88% | Mobile, Raspberry Pi |

**Recommendation:**
- 🖥️ **Server/Cloud:** Keras FP32
- 💻 **Desktop app:** TFLite FP32
- 📱 **Mobile/Edge:** TFLite INT8

### Speed vs Accuracy

**Inference Time (1 image):**

| Device | Keras FP32 | TFLite FP32 | TFLite INT8 |
|--------|------------|-------------|-------------|
| Desktop GPU | 5 ms | 4 ms | 2 ms |
| Laptop CPU | 50 ms | 40 ms | 15 ms |
| Raspberry Pi 4 | 500 ms | 400 ms | 100 ms |
| Smartphone | 200 ms | 150 ms | 50 ms |

**→ INT8 là 3-4× nhanh hơn trên edge devices!**

---

## 🔧 Deployment Strategies

### 1. Raspberry Pi

```python
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="mobilenetv2_int8.tflite")
interpreter.allocate_tensors()

# Inference
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

interpreter.set_tensor(input_details['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details['index'])
```

**Install:**
```bash
pip install tflite-runtime
```

### 2. Android (Kotlin/Java)

```kotlin
// Load model
val interpreter = Interpreter(loadModelFile())

// Prepare input
val inputArray = arrayOf(imageData)
val outputArray = arrayOf(FloatArray(10))

// Inference
interpreter.run(inputArray, outputArray)
val predictions = outputArray[0]
```

### 3. iOS (Swift)

```swift
// Load model
let model = try VNCoreMLModel(for: WasteClassifier().model)

// Create request
let request = VNCoreMLRequest(model: model) { request, error in
    guard let results = request.results as? [VNClassificationObservation]
    else { return }
    // Process results
}
```

### 4. Web (TensorFlow.js)

```javascript
// Load model
const model = await tf.loadGraphModel('model.json');

// Preprocess
const tensor = tf.browser.fromPixels(imageElement)
    .resizeBilinear([224, 224])
    .expandDims(0)
    .div(255.0);

// Inference
const predictions = await model.predict(tensor).data();
```

---

## 📈 Benchmarking

### Metrics to measure

**1. Model Size**
```python
size_mb = os.path.getsize(model_path) / (1024 * 1024)
```

**2. Inference Time**
```python
import time
start = time.time()
predictions = model.predict(image)
latency = time.time() - start
```

**3. Accuracy**
```python
test_loss, test_acc = model.evaluate(test_dataset)
```

**4. Memory Usage**
```python
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / (1024 * 1024)
```

---

## 💡 Best Practices

### 1. Always benchmark

**Before deployment:**
```
✓ Test on target device
✓ Measure actual latency
✓ Check memory consumption
✓ Verify accuracy on test set
```

### 2. Calibration data

**Representative dataset:**
- Dùng 100-500 samples từ training set
- Shuffle để diverse
- Không dùng augmentation

### 3. Quantization-aware training

**For best INT8 accuracy:**
```python
import tensorflow_model_optimization as tfmot

# Quantization-aware training
quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# Train
q_aware_model.fit(...)

# Convert to TFLite INT8
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
```

→ Accuracy drop < 1%!

### 4. Fallback strategies

```python
# Try INT8 first
if accuracy_int8 > threshold:
    use_int8_model()
else:
    # Fallback to FP16
    use_fp16_model()
```

---

## 🎓 Kiến thức học được

Sau Week 4, bạn sẽ hiểu:

✅ Model optimization techniques  
✅ Quantization (INT8, FP16)  
✅ TensorFlow Lite conversion  
✅ Size-Accuracy-Speed trade-offs  
✅ Deployment strategies cho mobile/edge  
✅ Benchmarking và profiling  

---

## 🔗 Tài liệu tham khảo

1. **TensorFlow Lite:**
   - Official TFLite Documentation
   - "TensorFlow Lite Guide" - Google

2. **Quantization:**
   - "Quantization and Training of Neural Networks" (2018)
   - "Integer Quantization for Deep Learning Inference" (2020)

3. **Deployment:**
   - "On-Device Machine Learning" - Google I/O
   - TFLite Performance Best Practices

---

**Previous:** [Week 3 - Real-time Detection](../Week3_Realtime_Detection/README.md)  
**Home:** [Project Root](../README.md)

