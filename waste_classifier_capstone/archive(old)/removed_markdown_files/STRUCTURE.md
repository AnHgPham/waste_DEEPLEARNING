# 🏗️ Project Structure Guide

**Version 2.0 - Reorganized Production Structure**

---

## 📋 Overview

This document explains the new, streamlined project structure designed for:
- **Easy development**: Clear separation of concerns
- **Easy debugging**: Modular, testable code
- **Academic rigor**: Well-documented, reproducible
- **Production ready**: Deployable modules and scripts

---

## 🎯 Design Philosophy

### Three-Layer Architecture

```
┌─────────────────────────────────────────────┐
│         USER INTERFACE LAYER                │
│  (main.py, scripts/, notebooks/)           │
├─────────────────────────────────────────────┤
│         BUSINESS LOGIC LAYER                │
│  (src/ - reusable modules)                 │
├─────────────────────────────────────────────┤
│         DATA LAYER                          │
│  (data/, outputs/)                         │
└─────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **DRY (Don't Repeat Yourself)**: Common code in `src/`, used by scripts
3. **Testability**: Modules can be imported and tested independently
4. **Documentation**: Theory in `docs/`, code comments, docstrings

---

## 📁 Detailed Structure

### 🔹 `src/` - Production Source Code

**Purpose**: Reusable, production-ready modules

```
src/
├── __init__.py              # Package initialization
├── config.py                # 🔧 SINGLE SOURCE OF TRUTH
│                            # All hyperparameters, paths, constants
│
├── data/                    # Data processing
│   ├── __init__.py
│   ├── preprocessing.py     # split_data(), create_data_generators()
│   └── loader.py            # load_dataset()
│
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── baseline.py          # build_baseline_model()
│   └── transfer.py          # build_transfer_model(), unfreeze_layers()
│
├── training/                # Training utilities (future)
│   └── __init__.py
│
├── evaluation/              # Evaluation utilities (future)
│   └── __init__.py
│
└── deployment/              # Deployment & optimization
    ├── __init__.py
    └── optimize.py          # convert_to_tflite(), quantize_model()
```

**Usage Example**:

```python
from src.config import *
from src.data import split_data, create_data_generators
from src.models import build_baseline_model, build_transfer_model
from src.deployment import quantize_model

# All hyperparameters come from config
model = build_baseline_model(INPUT_SHAPE, NUM_CLASSES)
train_ds, val_ds = create_data_generators(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE, RANDOM_SEED)
```

---

### 🔹 `scripts/` - Executable Scripts

**Purpose**: Numbered workflow scripts for running the pipeline

```
scripts/
├── 01_data_exploration.py    # 📊 Visualize dataset
├── 02_preprocessing.py        # 🔧 Split & prepare data
├── 03_baseline_training.py    # 🏋️ Train baseline CNN
├── 04_transfer_learning.py    # 🚀 Train MobileNetV2
├── 05_realtime_detection.py   # 📹 YOLO + classifier
├── 06_model_optimization.py   # ⚡ TFLite + quantization
└── 99_evaluate_model.py       # 📈 Evaluate any model
```

**Features**:
- ✅ **Self-contained**: Each script can run independently
- ✅ **CLI arguments**: Customizable parameters
- ✅ **Progress output**: Clear console messages
- ✅ **Error handling**: Graceful failures

**Usage**:

```bash
# Run directly
python scripts/01_data_exploration.py
python scripts/03_baseline_training.py --epochs 30

# Or via main.py
python main.py --train-baseline --epochs 30
```

---

### 🔹 `notebooks/` - Learning & Experimentation

**Purpose**: Jupyter notebooks for step-by-step learning

```
notebooks/
├── W1_Data_Exploration.ipynb      # 📊 Interactive EDA
├── W1_Preprocessing.ipynb         # 🔧 Data prep walkthrough
├── W1_Baseline_CNN.ipynb          # 🧠 Build & train CNN
├── W2_Feature_Extraction.ipynb    # 🎨 Transfer learning phase 1
├── W2_Fine_Tuning.ipynb           # 🎯 Transfer learning phase 2
├── W3_Integration.ipynb           # 🔗 YOLO + classifier
└── W4_Model_Optimization.ipynb    # ⚡ TFLite conversion
```

**When to use**:
- 📚 **Learning**: Understand concepts step-by-step
- 🔬 **Experimenting**: Try different hyperparameters
- 📊 **Visualizing**: Interactive plots and analysis
- 🐛 **Debugging**: Test individual components

**Workflow**:
1. Read theory from `docs/theory/Week*.md`
2. Open corresponding notebook
3. Run cells, experiment, learn
4. For production, use `scripts/` instead

---

### 🔹 `docs/` - Documentation

**Purpose**: Centralized documentation

```
docs/
├── theory/                          # 📖 Theoretical background
│   ├── Week1_Data_and_Baseline.md   # CNNs, convolution, backprop
│   ├── Week2_Transfer_Learning.md   # MobileNetV2, fine-tuning
│   ├── Week3_Realtime_Detection.md  # YOLO, object detection
│   └── Week4_Deployment.md          # TFLite, quantization
│
└── guides/                          # 📚 User guides
    ├── GETTING_STARTED.md           # Setup & installation
    ├── PROJECT_SUMMARY.md           # Technical overview
    ├── STRUCTURE.md                 # This file
    └── CHANGELOG.md                 # Version history
```

**Reading Order**:
1. `GETTING_STARTED.md` - Setup environment
2. `PROJECT_SUMMARY.md` - Understand project goals
3. `STRUCTURE.md` - Learn project organization
4. `theory/Week*.md` - Deep dive into concepts

---

### 🔹 `data/` - Datasets

```
data/
├── raw/                   # Original dataset (download here)
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
│
└── processed/             # Split dataset (generated)
    ├── train/
    ├── val/
    └── test/
```

**⚠️ Important**: `data/` is in `.gitignore` - not committed to Git

---

### 🔹 `outputs/` - Generated Files

```
outputs/
├── models/               # Trained models
│   ├── baseline_final.keras
│   ├── mobilenetv2_phase1.keras
│   ├── mobilenetv2_phase2.keras
│   ├── mobilenetv2_optimized.tflite
│   └── mobilenetv2_quantized.tflite
│
├── reports/              # Evaluation reports & plots
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── baseline_training_history.png
│   ├── confusion_matrix_baseline.png
│   └── classification_report_mobilenetv2.txt
│
├── logs/                 # Training logs
│   └── tensorboard/
│
└── screenshots/          # Real-time detection screenshots
    └── detection_*.jpg
```

**⚠️ Important**: `outputs/` is in `.gitignore` - not committed to Git

---

### 🔹 `Week[1-4]_*/` - Legacy Course Structure

**Purpose**: Original course materials (kept for reference)

```
Week1_Data_and_Baseline/
├── assignments/          # Original notebooks (BACKUP)
│   ├── W1_Data_Exploration.ipynb
│   ├── W1_Preprocessing.ipynb
│   └── W1_Baseline_CNN.ipynb
│
├── utils/                # Week-specific utilities (LEGACY)
│   ├── data_utils.py     # ⚠️ Moved to src/data/
│   └── model_utils.py    # ⚠️ Moved to src/models/
│
├── slides/               # Course slides
│   └── W1_Slides.md
│
└── datasets/             # (empty, unused)
```

**Status**: 
- ✅ **Keep**: For academic reference
- ⚠️ **Don't modify**: Use `src/` and `scripts/` for development
- 📚 **Notebooks**: Available in `notebooks/` directory
- 🔧 **Utils**: Consolidated in `src/` modules

---

## 🚀 Development Workflows

### Workflow 1: Quick Testing (Scripts)

```bash
# Best for: Running complete pipeline
python main.py --quick

# Or step-by-step
python scripts/01_data_exploration.py
python scripts/02_preprocessing.py
python scripts/03_baseline_training.py
```

### Workflow 2: Learning & Experimentation (Notebooks)

```bash
# Best for: Understanding concepts, trying ideas
jupyter notebook

# Open: notebooks/W1_Data_Exploration.ipynb
# Read first: docs/theory/Week1_Data_and_Baseline.md
```

### Workflow 3: Development (Modules)

```bash
# Best for: Adding features, fixing bugs

# 1. Modify source
vim src/models/baseline.py

# 2. Test with script
python scripts/03_baseline_training.py

# 3. Or test directly
python -c "from src.models import build_baseline_model; print('OK')"
```

### Workflow 4: Production Deployment

```python
# In your production code
from src.config import *
from src.models import build_transfer_model
from src.deployment import quantize_model

# Load and optimize
model = tf.keras.models.load_model(get_model_path('mobilenetv2', 'phase2'))
quantize_model(
    model_path=get_model_path('mobilenetv2', 'phase2'),
    output_path=MODELS_DIR / 'optimized.tflite',
    data_dir=TRAIN_DIR
)
```

---

## 🎯 Comparison: Old vs New Structure

### Old Structure (❌ Problems)

```
Week1_Data_and_Baseline/
  ├── data_exploration.py         # Script in week folder
  ├── utils/data_utils.py         # Utils scattered
Week2_Transfer_Learning/
  ├── transfer_learning.py        # Similar script duplicated
  ├── utils/model_utils.py        # Duplicate imports, hard to reuse
```

**Issues**:
- 😕 Confusing: Where is the main entry point?
- 🔄 Duplication: Similar code in multiple places
- 🐛 Hard to debug: Imports scattered across week folders
- 📚 Mixed purposes: Scripts, utils, notebooks all together

### New Structure (✅ Solutions)

```
src/                    # All reusable modules
scripts/                # All executable scripts (numbered)
notebooks/              # All notebooks (for learning)
docs/                   # All documentation
main.py                 # Single entry point
```

**Benefits**:
- 😊 Clear: `scripts/01_*.py` shows workflow order
- 🔧 Modular: `src/` modules are reusable
- 🐛 Debuggable: Simple imports, easy testing
- 📚 Organized: Each directory has one purpose

---

## 🔍 Finding Things

### "Where should I...?"

| Task | Location | Example |
|------|----------|---------|
| **Change hyperparameters** | `src/config.py` | `BATCH_SIZE = 64` |
| **Add a new model** | `src/models/` | `src/models/resnet.py` |
| **Modify data processing** | `src/data/` | `src/data/augmentation.py` |
| **Run the pipeline** | `main.py` | `python main.py --all` |
| **Run one step** | `scripts/` | `python scripts/03_*.py` |
| **Learn a concept** | `docs/theory/` | `docs/theory/Week2_*.md` |
| **Experiment** | `notebooks/` | `W1_Baseline_CNN.ipynb` |
| **Find a trained model** | `outputs/models/` | `mobilenetv2_phase2.keras` |
| **See evaluation results** | `outputs/reports/` | `confusion_matrix_*.png` |

---

## 📦 Import Patterns

### ✅ Correct Imports (New Structure)

```python
# In any script or notebook
from src.config import *
from src.data import split_data, create_data_generators
from src.models import build_baseline_model, build_transfer_model
from src.deployment import quantize_model
```

### ❌ Old Imports (Don't Use)

```python
# These are DEPRECATED
from Week1_Data_and_Baseline.utils.data_utils import split_data
from Week2_Transfer_Learning.utils.model_utils import build_transfer_model
from config import *  # Use src.config instead
```

---

## 🧪 Testing Structure

```bash
# Test config
python -c "from src.config import *; print(f'Classes: {NUM_CLASSES}')"

# Test data module
python -c "from src.data import split_data; print('Data module OK')"

# Test models
python -c "from src.models import build_baseline_model; print('Models OK')"

# Test script
python scripts/01_data_exploration.py

# Test full pipeline
python main.py --quick
```

---

## 📝 Adding New Features

### Example: Add a New Model (ResNet)

1. **Create module**: `src/models/resnet.py`

```python
from ..config import *
import tensorflow as tf

def build_resnet_model(input_shape, num_classes):
    # Implementation
    pass
```

2. **Update `__init__.py`**: `src/models/__init__.py`

```python
from .baseline import build_baseline_model
from .transfer import build_transfer_model
from .resnet import build_resnet_model  # NEW

__all__ = ['build_baseline_model', 'build_transfer_model', 'build_resnet_model']
```

3. **Create script**: `scripts/07_resnet_training.py`

```python
from src.config import *
from src.data import create_data_generators
from src.models import build_resnet_model  # Import new model

model = build_resnet_model(INPUT_SHAPE, NUM_CLASSES)
# Training code...
```

4. **Update main.py**: Add CLI option

```python
parser.add_argument('--train-resnet', action='store_true')
# Implementation...
```

---

## 🎓 For Academic Submission

### What to Submit

```
waste_classifier_capstone/
├── src/                    # ✅ Source code
├── scripts/                # ✅ Executable scripts
├── notebooks/              # ✅ Jupyter notebooks (with outputs)
├── docs/                   # ✅ Documentation
├── main.py                 # ✅ Entry point
├── setup.py                # ✅ Package setup
├── requirements.txt        # ✅ Dependencies
├── README.md               # ✅ Project overview
├── .gitignore              # ✅ Git ignore
│
├── outputs/reports/        # ✅ INCLUDE evaluation reports & plots
│   ├── *.png
│   └── *.txt
│
├── data/                   # ❌ EXCLUDE (too large, provide download link)
├── outputs/models/         # ❌ EXCLUDE (too large, provide download link)
└── Week*/                  # ✅ INCLUDE (for reference, show original work)
```

### Reproducibility Checklist

- ✅ `requirements.txt` with pinned versions
- ✅ `src/config.py` with all hyperparameters documented
- ✅ Fixed `RANDOM_SEED` for reproducibility
- ✅ Clear README with setup instructions
- ✅ Evaluation reports in `outputs/reports/`
- ✅ Git history showing development process

---

## 🆘 Troubleshooting

### Import Error: `ModuleNotFoundError: No module named 'src'`

**Solution**: Make sure you're running from project root:

```bash
cd waste_classifier_capstone/
python scripts/01_data_exploration.py
```

### FileNotFoundError: Can't find data

**Solution**: Check `src/config.py` paths and download dataset:

```bash
# Check config
python main.py --config

# Download dataset to data/raw/
```

### Model not found

**Solution**: Train the model first:

```bash
# Train baseline
python main.py --train-baseline

# Train transfer learning
python main.py --train-transfer
```

---

## 🎯 Summary

### Structure Benefits

| Aspect | Old | New |
|--------|-----|-----|
| **Entry point** | Unclear | `main.py` |
| **Workflow order** | Not obvious | Numbered `scripts/` |
| **Reusable code** | Scattered in `Week*/utils/` | Consolidated in `src/` |
| **Learning** | Mixed with scripts | Separate `notebooks/` |
| **Documentation** | Scattered READMEs | Organized `docs/` |
| **Production** | Hard to deploy | Import from `src/` |
| **Debugging** | Complex imports | Simple, clean imports |

---

**Happy developing with the new structure! 🚀**

For questions, see `docs/guides/GETTING_STARTED.md` or open an issue.
