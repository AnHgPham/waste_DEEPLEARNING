"""
Generate All Diagrams for Waste Classification Project

This script creates visual diagrams to help understand the project:
1. CNN Architecture Diagram
2. Training Pipeline Flowchart
3. Transfer Learning Two-Phase Diagram
4. Real-time Detection Flow
5. Model Comparison Chart
6. Complete System Architecture

Usage:
    python scripts/generate_diagrams.py

Output:
    All diagrams saved to docs/diagrams/
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# Create output directory
DIAGRAMS_DIR = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)

def set_style():
    """Set consistent style for all diagrams."""
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

# =============================================================================
# DIAGRAM 1: CNN Architecture (Baseline Model)
# =============================================================================

def create_cnn_architecture_diagram():
    """Create CNN architecture visualization."""
    print("Creating Diagram 1: CNN Architecture...")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Baseline CNN Architecture',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Layers
    layers = [
        {"name": "Input\n224×224×3", "x": 1, "y": 4, "w": 1.2, "h": 3, "color": "#FFE6E6"},
        {"name": "Conv\n32", "x": 3, "y": 4, "w": 1.2, "h": 2.8, "color": "#FFD4D4"},
        {"name": "Conv\n64", "x": 5, "y": 4, "w": 1.2, "h": 2.5, "color": "#FFC2C2"},
        {"name": "Conv\n128", "x": 7, "y": 4, "w": 1.2, "h": 2.2, "color": "#FFB0B0"},
        {"name": "Conv\n256", "x": 9, "y": 4, "w": 1.2, "h": 1.8, "color": "#FF9E9E"},
        {"name": "GAP", "x": 11, "y": 4, "w": 0.8, "h": 1.2, "color": "#E6F3FF"},
        {"name": "Dense\n128", "x": 12.5, "y": 4, "w": 0.8, "h": 1.2, "color": "#D4E8FF"},
    ]

    # Draw layers
    for i, layer in enumerate(layers):
        rect = FancyBboxPatch(
            (layer["x"], layer["y"]), layer["w"], layer["h"],
            boxstyle="round,pad=0.05",
            facecolor=layer["color"],
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(layer["x"] + layer["w"]/2, layer["y"] + layer["h"]/2,
                layer["name"], ha='center', va='center', fontsize=9, fontweight='bold')

        # Add arrows
        if i < len(layers) - 1:
            arrow = FancyArrowPatch(
                (layer["x"] + layer["w"], layer["y"] + layer["h"]/2),
                (layers[i+1]["x"], layers[i+1]["y"] + layers[i+1]["h"]/2),
                arrowstyle='->', mutation_scale=20, linewidth=2, color='blue'
            )
            ax.add_patch(arrow)

    # Output
    output_rect = FancyBboxPatch(
        (12.5, 2), 0.8, 1,
        boxstyle="round,pad=0.05",
        facecolor='#C2FFD4', edgecolor='black', linewidth=2
    )
    ax.add_patch(output_rect)
    ax.text(12.9, 2.5, 'Output\n10 classes', ha='center', va='center', fontsize=9, fontweight='bold')

    # Final arrow
    arrow = FancyArrowPatch(
        (12.9, 4), (12.9, 3),
        arrowstyle='->', mutation_scale=20, linewidth=2, color='blue'
    )
    ax.add_patch(arrow)

    # Annotations
    ax.text(1.6, 2.5, 'Input Layer', ha='center', fontsize=8, style='italic')
    ax.text(6, 2.5, 'Convolutional Blocks\n(BatchNorm + MaxPool)', ha='center', fontsize=8, style='italic')
    ax.text(12, 2.5, 'Classification\nHead', ha='center', fontsize=8, style='italic')

    # Stats
    stats_text = """
    Total Parameters: ~1.2M
    Training: From scratch
    Expected Accuracy: ~85%
    """
    ax.text(0.5, 1, stats_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '01_cnn_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# DIAGRAM 2: Training Pipeline Flowchart
# =============================================================================

def create_training_pipeline_diagram():
    """Create complete training pipeline flowchart."""
    print("Creating Diagram 2: Training Pipeline...")

    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(6, 13.5, 'Complete Training Pipeline',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Pipeline steps
    steps = [
        {"y": 12.5, "text": "Raw Dataset\n19,760 images", "color": "#FFE6E6"},
        {"y": 11.2, "text": "Data Exploration\n(Class distribution, sample images)", "color": "#FFF4E6"},
        {"y": 9.9, "text": "Data Preprocessing\nSplit: 80% train / 10% val / 10% test", "color": "#FFF4E6"},
        {"y": 8.6, "text": "Data Augmentation\n(Rotation, flip, zoom, contrast)", "color": "#FFF4E6"},
        {"y": 7.3, "text": "Baseline CNN Training\n30 epochs, ~85% accuracy", "color": "#E6F3FF"},
        {"y": 6.0, "text": "Transfer Learning Phase 1\nFeature Extraction (20 epochs)", "color": "#D4E8FF"},
        {"y": 4.7, "text": "Transfer Learning Phase 2\nFine-Tuning (15 epochs), ~95% accuracy", "color": "#D4E8FF"},
        {"y": 3.4, "text": "Model Evaluation\nConfusion matrix, per-class metrics", "color": "#FFE6F0"},
        {"y": 2.1, "text": "Model Optimization\nTFLite + INT8 Quantization", "color": "#E6FFE6"},
        {"y": 0.8, "text": "Deployment\nMobile/Edge Devices", "color": "#C2FFD4"},
    ]

    for i, step in enumerate(steps):
        # Draw box
        rect = FancyBboxPatch(
            (2, step["y"]), 8, 0.9,
            boxstyle="round,pad=0.1",
            facecolor=step["color"],
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(6, step["y"] + 0.45, step["text"],
                ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw arrow to next step
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (6, step["y"]), (6, steps[i+1]["y"] + 0.9),
                arrowstyle='->', mutation_scale=25, linewidth=2.5, color='blue'
            )
            ax.add_patch(arrow)

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '02_training_pipeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# DIAGRAM 3: Transfer Learning Two-Phase Strategy
# =============================================================================

def create_transfer_learning_diagram():
    """Create transfer learning two-phase visualization."""
    print("Creating Diagram 3: Transfer Learning Strategy...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Phase 1: Feature Extraction
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Phase 1: Feature Extraction', fontsize=16, fontweight='bold', pad=20)

    # MobileNetV2 Base (frozen)
    frozen_base = FancyBboxPatch(
        (2, 3), 6, 4,
        boxstyle="round,pad=0.1",
        facecolor='#E0E0E0', edgecolor='red', linewidth=3, linestyle='--'
    )
    ax1.add_patch(frozen_base)
    ax1.text(5, 5, 'MobileNetV2\n(ImageNet Pretrained)\n\n🔒 FROZEN',
            ha='center', va='center', fontsize=12, fontweight='bold', color='red')

    # Classification head
    head_rect = FancyBboxPatch(
        (3, 1), 4, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='#90EE90', edgecolor='green', linewidth=3
    )
    ax1.add_patch(head_rect)
    ax1.text(5, 1.6, 'Classification Head\n✓ TRAINABLE',
            ha='center', va='center', fontsize=11, fontweight='bold', color='green')

    # Arrow
    arrow1 = FancyArrowPatch((5, 3), (5, 2.2), arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
    ax1.add_patch(arrow1)

    # Details
    details1 = """
    Learning Rate: 0.0001
    Epochs: 20
    Strategy: Extract features
    Trainable: Head only
    """
    ax1.text(0.5, 8.5, details1, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Phase 2: Fine-Tuning
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Phase 2: Fine-Tuning', fontsize=16, fontweight='bold', pad=20)

    # MobileNetV2 Base (partially unfrozen)
    frozen_part = FancyBboxPatch(
        (2, 5), 6, 2,
        boxstyle="round,pad=0.1",
        facecolor='#E0E0E0', edgecolor='red', linewidth=3, linestyle='--'
    )
    ax2.add_patch(frozen_part)
    ax2.text(5, 6, 'MobileNetV2\n(Bottom Layers)\n🔒 FROZEN',
            ha='center', va='center', fontsize=11, fontweight='bold', color='red')

    unfrozen_part = FancyBboxPatch(
        (2, 3), 6, 1.8,
        boxstyle="round,pad=0.1",
        facecolor='#FFD700', edgecolor='orange', linewidth=3
    )
    ax2.add_patch(unfrozen_part)
    ax2.text(5, 3.9, 'Top 30 Layers\n✓ UNFROZEN',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkorange')

    # Classification head
    head_rect2 = FancyBboxPatch(
        (3, 1), 4, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='#90EE90', edgecolor='green', linewidth=3
    )
    ax2.add_patch(head_rect2)
    ax2.text(5, 1.6, 'Classification Head\n✓ TRAINABLE',
            ha='center', va='center', fontsize=11, fontweight='bold', color='green')

    # Arrows
    arrow2 = FancyArrowPatch((5, 3), (5, 2.2), arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
    ax2.add_patch(arrow2)

    # Details
    details2 = """
    Learning Rate: 0.00001 (lower!)
    Epochs: 15
    Strategy: Adapt features
    Trainable: Head + Top 30 layers
    Result: ~95% accuracy
    """
    ax2.text(0.5, 8.5, details2, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '03_transfer_learning_phases.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# DIAGRAM 4: Real-time Detection Flow
# =============================================================================

def create_realtime_detection_diagram():
    """Create real-time detection pipeline visualization."""
    print("Creating Diagram 4: Real-time Detection Flow...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'Real-time Waste Detection Pipeline',
            ha='center', va='top', fontsize=18, fontweight='bold')

    # Pipeline components
    components = [
        {"x": 1, "y": 7, "w": 2.5, "h": 1.5, "name": "Video Input\n(Webcam/File)", "color": "#FFE6E6"},
        {"x": 5, "y": 7, "w": 2.5, "h": 1.5, "name": "YOLOv8\nObject Detection", "color": "#FFF4E6"},
        {"x": 9, "y": 7, "w": 2.5, "h": 1.5, "name": "Crop Objects\n(Bounding Boxes)", "color": "#E6F3FF"},
        {"x": 5, "y": 4.5, "w": 2.5, "h": 1.5, "name": "MobileNetV2\nClassification", "color": "#D4E8FF"},
        {"x": 9, "y": 4.5, "w": 2.5, "h": 1.5, "name": "Draw Results\n(Boxes + Labels)", "color": "#E6FFE6"},
        {"x": 5, "y": 2, "w": 2.5, "h": 1.5, "name": "Display\nReal-time", "color": "#C2FFD4"},
    ]

    # Draw components
    for comp in components:
        rect = FancyBboxPatch(
            (comp["x"], comp["y"]), comp["w"], comp["h"],
            boxstyle="round,pad=0.1",
            facecolor=comp["color"],
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(comp["x"] + comp["w"]/2, comp["y"] + comp["h"]/2,
                comp["name"], ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows
    arrows = [
        ((3.5, 7.75), (5, 7.75)),  # Input -> YOLO
        ((7.5, 7.75), (9, 7.75)),  # YOLO -> Crop
        ((10.25, 7), (6.5, 6)),    # Crop -> Classify
        ((6.25, 4.5), (9.5, 6)),   # Classify -> Draw
        ((10.25, 4.5), (6.5, 3.5)), # Draw -> Display
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=25,
                               linewidth=3, color='blue')
        ax.add_patch(arrow)

    # Example output
    ax.text(1.5, 5, 'Example Frame:', fontsize=11, fontweight='bold')
    ax.text(1.5, 4.3, '📹 Frame captured', fontsize=9)
    ax.text(1.5, 3.9, '🔍 3 objects detected', fontsize=9)
    ax.text(1.5, 3.5, '✓ Object 1: plastic (95%)', fontsize=9, color='green')
    ax.text(1.5, 3.1, '✓ Object 2: paper (87%)', fontsize=9, color='green')
    ax.text(1.5, 2.7, '✓ Object 3: cardboard (92%)', fontsize=9, color='green')

    # Performance stats
    stats = """
    Performance:
    • YOLOv8n: ~50 FPS
    • Classification: ~100 FPS
    • Overall: 30+ FPS
    • Latency: <33ms
    """
    ax.text(11, 2, stats, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '04_realtime_detection_flow.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# DIAGRAM 5: Model Comparison Chart
# =============================================================================

def create_model_comparison_chart():
    """Create model performance comparison chart."""
    print("Creating Diagram 5: Model Comparison...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=18, fontweight='bold')

    models = ['Baseline\nCNN', 'MobileNetV2', 'MobileNetV2\n(INT8)']

    # 1. Accuracy comparison
    accuracies = [85, 95, 94]
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Model size comparison
    sizes = [4.8, 9.2, 2.4]
    bars2 = ax2.bar(models, sizes, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('Model Size', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} MB', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Inference time comparison
    inference_times = [15, 20, 8]
    bars3 = ax3.bar(models, inference_times, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('Inference Speed (CPU)', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 25)
    ax3.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Parameters comparison
    params = [1.2, 2.3, 2.3]
    bars4 = ax4.bar(models, params, color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax4.set_title('Model Parameters', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 3)
    ax4.grid(axis='y', alpha=0.3)

    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '05_model_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# DIAGRAM 6: Complete System Architecture
# =============================================================================

def create_system_architecture_diagram():
    """Create complete system architecture overview."""
    print("Creating Diagram 6: System Architecture...")

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'Complete System Architecture',
            ha='center', va='top', fontsize=20, fontweight='bold')

    # Data Layer
    data_box = FancyBboxPatch(
        (1, 9), 14, 1.8,
        boxstyle="round,pad=0.1",
        facecolor='#FFE6E6', edgecolor='black', linewidth=2
    )
    ax.add_patch(data_box)
    ax.text(8, 10.3, 'DATA LAYER', ha='center', fontsize=14, fontweight='bold')
    ax.text(3, 9.7, 'Raw Images\n(19,760)', ha='center', fontsize=10)
    ax.text(8, 9.7, 'Preprocessing\n(Split + Augment)', ha='center', fontsize=10)
    ax.text(13, 9.7, 'Train/Val/Test\n(80/10/10)', ha='center', fontsize=10)

    # Model Layer
    model_box = FancyBboxPatch(
        (1, 6.5), 14, 2,
        boxstyle="round,pad=0.1",
        facecolor='#E6F3FF', edgecolor='black', linewidth=2
    )
    ax.add_patch(model_box)
    ax.text(8, 8.2, 'MODEL LAYER', ha='center', fontsize=14, fontweight='bold')

    # Individual models
    baseline_box = FancyBboxPatch((1.5, 6.8), 3.5, 1.2, boxstyle="round",
                                  facecolor='#FFD4D4', edgecolor='black', linewidth=1.5)
    ax.add_patch(baseline_box)
    ax.text(3.25, 7.4, 'Baseline CNN\n1.2M params, 85%', ha='center', fontsize=9)

    transfer_box = FancyBboxPatch((6, 6.8), 4, 1.2, boxstyle="round",
                                  facecolor='#D4E8FF', edgecolor='black', linewidth=1.5)
    ax.add_patch(transfer_box)
    ax.text(8, 7.4, 'MobileNetV2 Transfer\n2.3M params, 95%', ha='center', fontsize=9)

    optimized_box = FancyBboxPatch((11, 6.8), 3.5, 1.2, boxstyle="round",
                                   facecolor='#C2FFD4', edgecolor='black', linewidth=1.5)
    ax.add_patch(optimized_box)
    ax.text(12.75, 7.4, 'TFLite INT8\n2.4MB, 94%', ha='center', fontsize=9)

    # Detection Layer
    detection_box = FancyBboxPatch(
        (1, 4), 14, 1.8,
        boxstyle="round,pad=0.1",
        facecolor='#FFF4E6', edgecolor='black', linewidth=2
    )
    ax.add_patch(detection_box)
    ax.text(8, 5.5, 'DETECTION LAYER', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 4.7, 'YOLOv8\n(Object Detection)', ha='center', fontsize=10)
    ax.text(8, 4.7, '+', ha='center', fontsize=16, fontweight='bold')
    ax.text(11, 4.7, 'MobileNetV2\n(Classification)', ha='center', fontsize=10)

    # Deployment Layer
    deploy_box = FancyBboxPatch(
        (1, 1.5), 14, 2,
        boxstyle="round,pad=0.1",
        facecolor='#E6FFE6', edgecolor='black', linewidth=2
    )
    ax.add_patch(deploy_box)
    ax.text(8, 3.2, 'DEPLOYMENT LAYER', ha='center', fontsize=14, fontweight='bold')

    # Deployment targets
    targets = ['Web App\n(FastAPI)', 'Mobile\n(Android/iOS)', 'Edge Device\n(Raspberry Pi)', 'Docker\nContainer']
    target_x = [2.5, 5.5, 9, 12.5]
    for i, (target, x) in enumerate(zip(targets, target_x)):
        target_box = FancyBboxPatch((x-0.8, 1.7), 1.8, 1.2, boxstyle="round",
                                    facecolor='#90EE90', edgecolor='black', linewidth=1.5)
        ax.add_patch(target_box)
        ax.text(x, 2.3, target, ha='center', fontsize=9)

    # Arrows connecting layers
    for y_start, y_end in [(9, 8.5), (6.5, 5.8), (4, 3.5)]:
        arrow = FancyArrowPatch((8, y_start), (8, y_end),
                               arrowstyle='->', mutation_scale=30, linewidth=3, color='blue')
        ax.add_patch(arrow)

    # Legend
    legend_text = """
    Key Features:
    ✓ End-to-end ML pipeline
    ✓ Two model approaches
    ✓ Real-time detection
    ✓ Multi-platform deployment
    ✓ Edge-optimized models
    """
    ax.text(0.5, 0.5, legend_text, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    save_path = DIAGRAMS_DIR / '06_system_architecture.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Generate all diagrams."""
    print("=" * 70)
    print("GENERATING ALL DIAGRAMS FOR WASTE CLASSIFICATION PROJECT")
    print("=" * 70)
    print(f"\nOutput directory: {DIAGRAMS_DIR}\n")

    set_style()

    # Generate all diagrams
    create_cnn_architecture_diagram()
    create_training_pipeline_diagram()
    create_transfer_learning_diagram()
    create_realtime_detection_diagram()
    create_model_comparison_chart()
    create_system_architecture_diagram()

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL DIAGRAMS GENERATED!")
    print("=" * 70)
    print(f"\nGenerated files:")
    for i in range(1, 7):
        filename = list(DIAGRAMS_DIR.glob(f'0{i}_*.png'))[0]
        print(f"  {i}. {filename.name}")

    print(f"\nLocation: {DIAGRAMS_DIR}")
    print("\nYou can now use these diagrams in your documentation!")

if __name__ == "__main__":
    main()
