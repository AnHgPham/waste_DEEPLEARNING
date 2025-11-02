# Receptive Field Visualization

## Baseline CNN Receptive Field

```
Input Image: 224x224

After Block 1 (MaxPool): 112x112
  Receptive field: 5x5 pixels

After Block 2 (MaxPool): 56x56
  Receptive field: 13x13 pixels

After Block 3 (MaxPool): 28x28
  Receptive field: 29x29 pixels

After Block 4 (MaxPool): 14x14
  Receptive field: 61x61 pixels
```

**Final receptive field:** ~61x61 pixels (27% of image)

## Comparison:

```
Baseline CNN:      61x61 pixels  (27% coverage)
MobileNetV2:       ~150x150      (70% coverage)
ResNet50:          ~200x200      (95% coverage)
```

## Why This Matters:

Waste objects cần nhìn TOÀN BỘ object để classify đúng:

```
Plastic Bottle:
  - Need to see: Shape + Transparency + Cap + Label
  - Required view: ~80-120 pixels minimum

With 61x61 receptive field:
  ✗ Cannot see full context
  ✗ Misses important details
  ✓ Only sees local patches
```

---

## Visual Example:

```
Original Image (224x224):
┌──────────────────────┐
│                      │
│    [Plastic Bottle]  │
│    ┌──────┐          │
│    │ Cap  │          │
│    │      │          │
│    │Label │          │
│    │      │          │
│    │      │          │
│    └──────┘          │
│                      │
└──────────────────────┘

Baseline Sees (61x61):
┌──────────────────────┐
│                      │
│    [????]            │
│    ┌───┐             │ <- Only sees part
│    │ Ca│             │
│    │   │             │
│    └───┘             │
│                      │
│                      │
│                      │
│                      │
└──────────────────────┘

MobileNetV2 Sees (150x150):
┌──────────────────────┐
│ ┌──────────────────┐ │
│ │[Plastic Bottle]  │ │ <- Sees full object
│ │┌──────┐          │ │
│ ││ Cap  │          │ │
│ ││      │          │ │
│ ││Label │          │ │
│ ││      │          │ │
│ ││      │          │ │
│ │└──────┘          │ │
│ └──────────────────┘ │
└──────────────────────┘
```

This is why deeper networks with larger receptive fields perform better!
