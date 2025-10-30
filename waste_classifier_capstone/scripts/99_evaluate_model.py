"""
Model Evaluation Script

This script evaluates a trained model on the test set and generates:
- Confusion matrix
- Classification report
- Per-class accuracy

Usage:
    python scripts/evaluate_model.py --model mobilenetv2
    python scripts/evaluate_model.py --model baseline

"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plots a confusion matrix.

    Arguments:
    cm -- np.ndarray, confusion matrix.
    class_names -- list, list of class names.
    save_path -- Path or None, path to save the figure.
    """
    plt.figure(figsize=CM_FIGSIZE)
    
    # Normalize confusion matrix if specified
    if CM_NORMALIZE:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f' if CM_NORMALIZE else 'd',
                cmap=CM_CMAP, xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if CM_NORMALIZE else 'Count'})
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI)
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()

def main(args):
    """Main function for model evaluation."""
    print("=" * 70)
    print("WASTE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    model_name = args.model
    model_path = get_model_path(model_name, 'final')
    
    if not model_path.exists():
        print(f"\n❌ ERROR: Model not found: {model_path}")
        print(f"\n   Please train the {model_name} model first.")
        return
    
    print(f"\n📦 Loading model: {model_name}")
    model = tf.keras.models.load_model(model_path)
    print(f"   ✅ Model loaded from {model_path}")
    
    # Load test dataset
    print(f"\n📂 Loading test dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False
    )
    
    # Apply appropriate preprocessing based on model type
    # MobileNetV2: Keep [0,255], preprocessing is built into the model
    # Baseline: Apply Rescaling(1./255) manually
    if model_name == 'baseline':
        print(f"   Applying baseline preprocessing (Rescaling 0-1)...")
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    else:
        print(f"   Using model's built-in preprocessing (MobileNetV2)...")
        # MobileNetV2 has preprocess_input built-in, no need to normalize here
        # Images remain in [0, 255] range and will be normalized to [-1, 1] by the model
    
    print(f"   ✅ Test dataset loaded")
    
    # Evaluate model
    print(f"\n🧪 Evaluating model on test set...")
    test_loss, test_acc, test_top5 = model.evaluate(test_ds, verbose=1)
    
    print(f"\n📈 Test Results:")
    print(f"   - Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   - Top-5 Accuracy: {test_top5:.4f} ({test_top5*100:.2f}%)")
    print(f"   - Test Loss: {test_loss:.4f}")
    
    # Generate predictions for detailed analysis
    print(f"\n🔮 Generating predictions...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    print(f"   ✅ {len(y_true)} predictions generated")
    
    # Generate confusion matrix
    print(f"\n📊 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_path = get_report_path(model_name, 'confusion_matrix')
    plot_confusion_matrix(cm, CLASS_NAMES, save_path=cm_path)
    
    # Generate classification report
    print(f"\n📋 Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    
    # Save classification report
    report_path = get_report_path(model_name, 'classification_report')
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"\n✅ Classification report saved to {report_path}")
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = (y_true == i)
        class_acc = (y_pred[class_mask] == i).sum() / class_mask.sum()
        print(f"   {class_name:12s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Find most confused classes
    print(f"\n🔍 Most Confused Classes (Top 5):")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(cm_normalized, 0)  # Ignore diagonal
    
    confused_pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j:
                confused_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm_normalized[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (true_class, pred_class, confusion_rate) in enumerate(confused_pairs[:5], 1):
        print(f"   {i}. {true_class} → {pred_class}: {confusion_rate:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ Model evaluation complete!")
    print("=" * 70)
    print(f"\n📁 Reports saved to:")
    print(f"   - Confusion matrix: {cm_path}")
    print(f"   - Classification report: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Trained Model')
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['baseline', 'mobilenetv2'],
                        help='Model to evaluate (default: mobilenetv2)')
    args = parser.parse_args()
    
    main(args)

