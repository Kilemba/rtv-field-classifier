"""
evaluate.py — Task 2: Model Evaluation
────────────────────────────────────────
This script loads the saved best model and evaluates it on the held-out TEST set.

The test set is data the model has NEVER seen during training or validation.
It is the most honest measure of real-world performance.

Metrics we report:
  - Accuracy:   Overall fraction of correctly classified images
  - Precision:  Of all images predicted as class X, how many are actually class X?
  - Recall:     Of all images that ARE class X, how many did the model find?
  - F1 Score:   Harmonic mean of precision and recall — best single metric for
                imbalanced datasets
  - Confusion Matrix: Shows which classes are confused with which — the most
                      informative diagnostic tool for a classifier

Why these metrics matter for RTV:
  Accuracy alone is misleading when classes are imbalanced. If 'compost' has
  200 images and 'vsla' has 50, a model that just always predicts 'compost'
  would have 80% accuracy but 0% recall on vsla — completely useless in the field.
  F1 per class gives us the full picture.

Author: Caleb Kilemba
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders, CLASS_NAMES
from model   import load_model_for_inference


# ─── Collect all predictions on the test set ──────────────────────────────────

@torch.no_grad()
def run_inference(
    model:  torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model on every image in the DataLoader and collect results.

    Args:
        model:  Trained model in eval() mode.
        loader: Test DataLoader.
        device: cpu or cuda.

    Returns:
        all_preds:  Array of predicted class indices.
        all_labels: Array of true class indices.
        all_confs:  Array of confidence scores (max softmax probability).
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_confs  = []

    for images, labels in tqdm(loader, desc="  Running inference on test set"):
        images = images.to(device)

        logits      = model(images)
        # Softmax converts raw scores to probabilities that sum to 1.0
        probs       = torch.softmax(logits, dim=1)
        # The confidence score is the highest probability
        confidence, predictions = probs.max(dim=1)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confs.extend(confidence.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_confs),
    )


# ─── Confusion matrix plot ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    output_dir:  str,
):
    """
    Save a colour-coded confusion matrix as a PNG file.

    How to read a confusion matrix:
    - Rows = true class (what it actually is)
    - Columns = predicted class (what the model said)
    - Diagonal cells (top-left to bottom-right) = correct predictions (we want these high)
    - Off-diagonal cells = mistakes (we want these low/empty)

    A bright off-diagonal cell means the model often confuses those two categories.
    This tells us where more training data or annotation quality improvements are needed.
    """
    cm = confusion_matrix(true_labels, pred_labels)

    # Normalise by row so each cell shows % of that true class (easier to read)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Raw counts ────────────────────────────────────────────────────────────
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (raw counts)", fontsize=13, pad=12)
    axes[0].set_xlabel("Predicted", fontsize=11)
    axes[0].set_ylabel("Actual",    fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)

    # ── Normalised (%) ────────────────────────────────────────────────────────
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Confusion Matrix (normalised by row)", fontsize=13, pad=12)
    axes[1].set_xlabel("Predicted", fontsize=11)
    axes[1].set_ylabel("Actual",    fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


# ─── Print and save the full classification report ─────────────────────────────

def print_evaluation_report(
    true_labels:  np.ndarray,
    pred_labels:  np.ndarray,
    confidences:  np.ndarray,
    class_names:  list[str],
    output_dir:   str,
):
    """
    Print precision, recall, F1 per class and save the report to a file.

    Args:
        true_labels:  Ground truth class indices.
        pred_labels:  Model's predicted class indices.
        confidences:  Softmax confidence for each prediction.
        class_names:  List of class name strings.
        output_dir:   Where to save the text report.
    """
    overall_accuracy = accuracy_score(true_labels, pred_labels)

    # scikit-learn gives us per-class precision, recall, F1 in one call
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=3,
    )

    # Average confidence across all predictions
    avg_confidence = confidences.mean()

    print(f"\n{'='*60}")
    print(f"  Test Set Evaluation Results")
    print(f"{'='*60}")
    print(f"  Overall Accuracy  : {overall_accuracy*100:.1f}%")
    print(f"  Mean Confidence   : {avg_confidence*100:.1f}%")
    print(f"\n  Per-Class Report:")
    print(report)

    # ── Identify the hardest classes ─────────────────────────────────────────
    # These are the categories the model confuses most often
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    print("  Hardest classes (lowest recall):")
    for i in range(len(class_names)):
        recall = cm_norm[i, i]
        if recall < 0.70:   # flag classes below 70% recall
            print(f"    ⚠  {class_names[i]:25s} recall = {recall*100:.1f}%")

    # ── Save full report to file ──────────────────────────────────────────────
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("RTV Field Image Classifier — Test Set Evaluation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy  : {overall_accuracy*100:.1f}%\n")
        f.write(f"Mean Confidence   : {avg_confidence*100:.1f}%\n\n")
        f.write("Per-Class Report:\n")
        f.write(report)

    print(f"\n  Full report saved → {report_path}")


# ─── Main evaluation function ──────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str = "outputs/best_model.pth",
    data_dir:        str = "data",
    output_dir:      str = "outputs",
    batch_size:      int = 32,
):
    """
    Load the saved model and evaluate it on the held-out test set.

    Args:
        checkpoint_path: Path to the saved .pth model file.
        data_dir:        Path to the data/ folder.
        output_dir:      Where to save evaluation outputs.
        batch_size:      Batch size for inference.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RTV Image Classifier — Test Set Evaluation")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {checkpoint_path}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            "Please run 'python src/train.py' first."
        )

    print("\n  Loading saved model...")
    model = load_model_for_inference(checkpoint_path, device)
    print("  ✓ Model loaded successfully.")

    # ── Load test data ────────────────────────────────────────────────────────
    print("\n  Loading test data...")
    _, _, test_loader, _ = build_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        num_workers = 0,
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\n  Running model on test set...")
    all_preds, all_labels, all_confs = run_inference(model, test_loader, device)

    # ── Print results ─────────────────────────────────────────────────────────
    print_evaluation_report(all_labels, all_preds, all_confs, CLASS_NAMES, output_dir)

    # ── Save confusion matrix plot ─────────────────────────────────────────────
    print("\n  Generating confusion matrix plot...")
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, output_dir)

    print(f"\n  All evaluation outputs saved to: {output_dir}/")
    print("  ✓ Evaluation complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
