# Continue Training Guide

## Overview

This guide explains how to continue training your baseline model when the initial accuracy is not satisfactory.

## When to Continue Training?

Continue training when:
- ✅ Initial validation accuracy < 85%
- ✅ Training curves show the model can still improve
- ✅ Validation loss is still decreasing
- ✅ You want to squeeze out more performance

Do NOT continue training when:
- ❌ Model is clearly overfitting (train acc >> val acc)
- ❌ Validation loss plateaued or increasing
- ❌ Already achieved satisfactory accuracy
- ❌ Better to switch to transfer learning

## How It Works

### Automatic Version Management

The script automatically manages model versions:

```
Initial training:
  baseline_final.keras  (v0)

First continuation:
  baseline_v1.keras

Second continuation:
  baseline_v2.keras

... and so on
```

### Learning Rate Strategy

**Default behavior:** Reduces learning rate by 10x

```python
Initial training: LR = 0.001
First continue:   LR = 0.0001  (10x lower)
Second continue:  LR = 0.00001 (10x lower again)
```

**Why reduce LR?**
- Model already found good weights
- Need smaller updates to fine-tune
- Prevents overshooting the optimum

## Usage

### Basic Usage

```bash
# Continue with default settings (20 epochs, LR * 0.1)
python main.py --continue-baseline

# Or directly:
python scripts/07_continue_baseline_training.py
```

### Custom Epochs

```bash
# Train 30 more epochs
python main.py --continue-baseline --epochs 30

# Train 50 more epochs
python scripts/07_continue_baseline_training.py --epochs 50
```

### Custom Learning Rate

```bash
# Use specific learning rate
python scripts/07_continue_baseline_training.py --epochs 20 --lr 0.0001

# Keep original learning rate (not recommended)
python scripts/07_continue_baseline_training.py --epochs 20 --keep-lr
```

## Example Workflow

### Scenario 1: Low Initial Accuracy

```bash
# Initial training
python main.py --train-baseline --epochs 30
# Result: 75% validation accuracy (too low!)

# Continue training
python main.py --continue-baseline --epochs 20
# Result: 82% validation accuracy (better!)

# Continue again if needed
python main.py --continue-baseline --epochs 20
# Result: 85% validation accuracy (good!)
```

### Scenario 2: Model Not Converged

```bash
# Initial training stopped early
python main.py --train-baseline --epochs 30
# EarlyStopping triggered at epoch 25
# Result: 83% validation accuracy

# Continue to see if it can improve
python main.py --continue-baseline --epochs 15
# Result: 86% validation accuracy
```

### Scenario 3: Comparing with Transfer Learning

```bash
# Train baseline to convergence
python main.py --train-baseline --epochs 30
python main.py --continue-baseline --epochs 20
python main.py --continue-baseline --epochs 20
# Best baseline: 87% validation accuracy

# Now try transfer learning
python main.py --train-transfer
# Result: 95% validation accuracy
# → Transfer learning is much better!
```

## Output Files

Each continuation creates:

```
outputs/models/
  ├── baseline_final.keras      # Original (v0)
  ├── baseline_v1.keras          # First continuation
  ├── baseline_v2.keras          # Second continuation
  └── ...

outputs/reports/
  ├── baseline_v1_history.pkl    # Training history
  ├── baseline_v1_comparison.png # Comparison plot
  ├── baseline_v2_history.pkl
  ├── baseline_v2_comparison.png
  └── ...
```

## Understanding the Output

### Console Output

```
============================================================
CONTINUE TRAINING BASELINE CNN
============================================================

============================================================
STEP 1: LOADING EXISTING MODEL
============================================================

[FOUND] Latest baseline model:
  Path: outputs/models/baseline_final.keras
  Version: final
  Size: 4.73 MB

[MODEL INFO]
  Total parameters: 1,234,567
  Trainable parameters: 1,234,567
  Current learning rate: 0.001000

[CURRENT] Validation Accuracy: 0.8234 (82.34%)
[CURRENT] Validation Loss: 0.4521

============================================================
STEP 2: CONFIGURING CONTINUED TRAINING
============================================================

[LEARNING RATE] Reducing 10x: 0.001000 -> 0.000100

============================================================
STEP 4: CONTINUING TRAINING
============================================================

Epoch 1/20
500/500 [======] - 45s - loss: 0.4201 - accuracy: 0.8534 - val_loss: 0.4321 - val_accuracy: 0.8456
...

============================================================
STEP 5: EVALUATING IMPROVEMENT
============================================================

[RESULTS]
  Final Validation Accuracy: 0.8567 (85.67%)

[IMPROVEMENT]
  Before: 0.8234 (82.34%)
  After:  0.8567 (85.67%)
  Gain:   +0.0333 (+4.04%)

  [SUCCESS] Model improved! Accuracy increased by 0.0333
```

### Comparison Plot

The script generates a 4-panel plot:

```
┌─────────────────┬─────────────────┐
│ Accuracy        │ Loss            │
│ (Old vs New)    │ (Old vs New)    │
├─────────────────┼─────────────────┤
│ Val Acc         │ Training        │
│ Per Epoch       │ Summary         │
└─────────────────┴─────────────────┘
```

## Best Practices

### 1. Monitor for Overfitting

```python
# Good signs:
train_acc ≈ val_acc  (within 5%)
val_loss decreasing

# Warning signs:
train_acc >> val_acc  (difference > 10%)
val_loss increasing
```

### 2. Learning Rate Guidelines

| Situation | Recommended LR |
|-----------|----------------|
| First continuation | Current LR * 0.1 |
| Second continuation | Current LR * 0.1 again |
| Model oscillating | Current LR * 0.5 |
| Very close to convergence | Current LR * 0.01 |

### 3. When to Stop

Stop continuing training when:
- Validation accuracy improvement < 0.5% per continuation
- Training time exceeds benefit
- Baseline reaches ~85-90% (switch to transfer learning)
- Clear overfitting observed

## Comparison with Transfer Learning

| Aspect | Continue Baseline | Transfer Learning |
|--------|-------------------|-------------------|
| **Time to implement** | Instant | Need to set up |
| **Training time** | +30-60 min | ~60-90 min |
| **Final accuracy** | ~85-90% | ~95% |
| **When to use** | Quick improvement | Best performance |

**Recommendation:**
1. Train baseline 30 epochs
2. If accuracy < 85%, continue 1-2 times
3. If still < 90%, switch to transfer learning

## Troubleshooting

### Issue 1: No Model Found

```
[ERROR] No trained baseline model found!
```

**Solution:** Train baseline first
```bash
python main.py --train-baseline
```

### Issue 2: Model Getting Worse

```
[WARNING] Model got worse! Accuracy decreased by 0.0234
```

**Solutions:**
- Use lower learning rate: `--lr 0.00001`
- Train fewer epochs: `--epochs 10`
- Check for data issues
- May have already converged

### Issue 3: Minimal Improvement

```
[INFO] Minimal change. Model may have converged.
```

**Solutions:**
- Model has converged, stop training
- Switch to transfer learning
- Try different architecture

## Advanced Usage

### Manual Version Selection

```python
from tensorflow import keras
from src.config import *

# Load specific version
model = keras.models.load_model(MODELS_DIR / 'baseline_v2.keras')

# Continue from that version
# (modify script to accept --from-version argument)
```

### Custom Callbacks

Modify script to add custom callbacks:

```python
custom_callbacks = [
    keras.callbacks.TensorBoard(log_dir='./logs'),
    keras.callbacks.CSVLogger('training.csv'),
    # ... your custom callbacks
]
```

### Learning Rate Scheduling

```python
# Cosine annealing
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    ...
)
```

## Summary

The continue training script provides:
- ✅ Easy resumption from checkpoints
- ✅ Automatic version management
- ✅ Smart learning rate reduction
- ✅ Performance comparison
- ✅ Visual progress tracking

**Use it when:**
- Initial baseline accuracy is low
- Model hasn't converged yet
- You want to maximize baseline performance

**Don't use it when:**
- Model is overfitting
- Already achieved good accuracy
- Better to switch to transfer learning

---

**Next Steps:**
- [Train Transfer Learning Model](../theory/Week2_Transfer_Learning.md)
- [Evaluate Model Performance](GETTING_STARTED.md#evaluation)
- [Optimize for Deployment](../theory/Week4_Deployment.md)
