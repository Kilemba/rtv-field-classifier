"""
train.py — Task 2: Model Training
───────────────────────────────────
This script trains the EfficientNet-B0 classifier on RTV field images.

Training strategy:
  Phase 1 (Epochs 1–5):  Freeze backbone, train only the head.
                          The head learns to map RTV-specific features to labels.
                          This prevents "catastrophic forgetting" of ImageNet knowledge.

  Phase 2 (Epochs 6–20): Unfreeze all layers, train with a smaller learning rate.
                          The whole network fine-tunes together for RTV images.

This two-phase approach is the standard for transfer learning with small datasets
(< 5,000 images) and consistently outperforms training the whole network at once.

Key decisions:
  - Loss: CrossEntropyLoss with class weights — penalises errors on minority classes more
  - Optimiser: AdamW — Adam with weight decay, better than standard Adam for vision tasks
  - Scheduler: CosineAnnealingLR — smoothly reduces learning rate over training
  - Early stopping: halts training if val loss stops improving for 5 epochs (saves time)
  - Checkpoint: saves the best model based on validation accuracy

Author: Caleb Kilemba
"""

import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use("Agg")           # non-interactive backend so it works without a display
import matplotlib.pyplot as plt
from tqdm import tqdm           # progress bar

# Add src/ to path so we can import our own modules
sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders, CLASS_NAMES
from model   import build_model, freeze_backbone, unfreeze_all, count_parameters


# ─── Hyperparameters ──────────────────────────────────────────────────────────
# These are the settings that control how training behaves.
# They were chosen based on best practices for transfer learning on small datasets.

CONFIG = {
    "data_dir":          "data",
    "output_dir":        "outputs",
    "batch_size":        32,
    "num_epochs":        20,         # total training epochs
    "warmup_epochs":     5,          # epochs where backbone is frozen
    "lr_head":           1e-3,       # learning rate for the head (Phase 1)
    "lr_finetune":       1e-4,       # smaller learning rate for full fine-tuning (Phase 2)
    "weight_decay":      1e-4,       # L2 regularisation (prevents overfitting)
    "early_stop_patience": 5,        # stop if val loss doesn't improve for 5 epochs
    "num_classes":       9,
    "num_workers":       0,          # 0 = safe on all platforms (no multiprocessing issues)
    "seed":              42,         # for reproducibility
}


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix random seeds so training is reproducible across runs."""
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Class-weighted loss ───────────────────────────────────────────────────────

def compute_class_weights(train_labels: list, device: torch.device) -> torch.Tensor:
    """
    Compute loss weights inversely proportional to class frequency.

    Classes with fewer images get a higher loss weight, so the model
    is penalised more for misclassifying minority classes.

    This is a complement to the WeightedRandomSampler — together they
    give robust handling of the class imbalance in RTV's dataset.

    Args:
        train_labels: Integer class labels for the training set.
        device:       Where to put the weight tensor.

    Returns:
        Tensor of shape [num_classes] with per-class loss weights.
    """
    counts = Counter(train_labels)
    total  = sum(counts.values())

    # Weight = total / (num_classes * count) — standard inverse frequency weighting
    weights = [
        total / (CONFIG["num_classes"] * counts.get(i, 1))
        for i in range(CONFIG["num_classes"])
    ]
    return torch.FloatTensor(weights).to(device)


# ─── One epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     torch.utils.data.DataLoader,
    criterion:  nn.Module,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
    epoch:      int,
) -> tuple[float, float]:
    """
    Run one full pass over the training data.

    Args:
        model:     The neural network.
        loader:    Training DataLoader.
        criterion: Loss function.
        optimiser: The optimiser (AdamW).
        device:    'cpu' or 'cuda'.
        epoch:     Current epoch number (for display only).

    Returns:
        avg_loss:  Mean loss over all batches.
        accuracy:  Fraction of correct predictions (0.0 to 1.0).
    """
    model.train()   # activate training mode (enables dropout, BatchNorm uses batch stats)

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # tqdm wraps the DataLoader to show a progress bar
    loop = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for images, labels in loop:
        # Move data to GPU (or keep on CPU if no GPU)
        images = images.to(device)
        labels = labels.to(device)

        # ── Forward pass ──────────────────────────────────────────────────────
        optimiser.zero_grad()         # clear gradients from previous step
        logits = model(images)        # model output: raw scores for each class
        loss   = criterion(logits, labels)  # compare to true labels

        # ── Backward pass ─────────────────────────────────────────────────────
        loss.backward()               # compute gradients
        optimiser.step()              # update weights in direction that reduces loss

        # ── Track metrics ─────────────────────────────────────────────────────
        total_loss    += loss.item() * images.size(0)
        predictions    = logits.argmax(dim=1)           # predicted class = highest score
        correct       += (predictions == labels).sum().item()
        total_samples += images.size(0)

        # Update progress bar with current loss
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ─── One epoch of validation ───────────────────────────────────────────────────

@torch.no_grad()  # disables gradient computation — we don't need it for evaluation
def evaluate_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    split:     str = "Val",
) -> tuple[float, float]:
    """
    Evaluate the model on validation or test data (no weight updates).

    Args:
        model:     The neural network (in eval mode).
        loader:    Val or test DataLoader.
        criterion: Loss function.
        device:    'cpu' or 'cuda'.
        split:     Label for the progress bar ("Val" or "Test").

    Returns:
        avg_loss:  Mean loss.
        accuracy:  Fraction correct.
    """
    model.eval()  # deactivate dropout; BatchNorm uses stored running statistics

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    loop = tqdm(loader, desc=f"         [{split}]  ", leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss    += loss.item() * images.size(0)
        predictions    = logits.argmax(dim=1)
        correct       += (predictions == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ─── Plot training curves ──────────────────────────────────────────────────────

def plot_training_curves(history: dict, output_dir: str):
    """
    Save a plot of training and validation loss + accuracy over epochs.

    This is useful for diagnosing:
    - Overfitting: train accuracy >> val accuracy
    - Underfitting: both are low
    - Convergence: curves flattening out
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add vertical line where we unfreeze the backbone
    warmup = CONFIG["warmup_epochs"]
    ax1.axvline(x=warmup, color="grey", linestyle="--", alpha=0.7,
                label=f"Backbone unfrozen (epoch {warmup})")
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val Accuracy",   markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {save_path}")


# ─── Main training function ────────────────────────────────────────────────────

def train(config: dict = CONFIG):
    """
    Full training pipeline from data loading to saving the best model.

    This function orchestrates:
    1. Device setup (GPU if available, CPU otherwise)
    2. Data loading via build_dataloaders()
    3. Model construction with frozen backbone
    4. Phase 1: head-only training for warmup_epochs
    5. Phase 2: full fine-tuning for remaining epochs
    6. Early stopping based on validation loss
    7. Saving the best checkpoint
    8. Plotting training curves

    Args:
        config: Hyperparameter dictionary (uses CONFIG by default).
    """
    set_seed(config["seed"])

    # ── Device setup ──────────────────────────────────────────────────────────
    # Use GPU if available — training is ~10x faster on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  RTV Image Classifier — Model Training")
    print(f"{'='*60}")
    print(f"  Device        : {device}")
    print(f"  Epochs        : {config['num_epochs']} (warmup: {config['warmup_epochs']})")
    print(f"  Batch size    : {config['batch_size']}")
    print(f"  Learning rate : {config['lr_head']} → {config['lr_finetune']} (after warmup)")

    # ── Create output directory ────────────────────────────────────────────────
    os.makedirs(config["output_dir"], exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n── Loading data ─────────────────────────────────────────")
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir    = config["data_dir"],
        batch_size  = config["batch_size"],
        num_workers = config["num_workers"],
    )

    # We need the training labels to compute class weights
    # Re-read them from the dataset attribute
    train_labels = train_loader.dataset.labels

    # ── Model ──────────────────────────────────────────────────────────────────
    print("\n── Building model ───────────────────────────────────────")
    model = build_model(num_classes=config["num_classes"], pretrained=True)
    model = model.to(device)

    # Start with frozen backbone (Phase 1)
    model = freeze_backbone(model)

    # ── Loss function ──────────────────────────────────────────────────────────
    # We compute class weights so minority classes get higher loss
    class_weights = compute_class_weights(train_labels, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    # ── Phase 1 optimiser: trains only the head ────────────────────────────────
    head_params  = [p for p in model.parameters() if p.requires_grad]
    optimiser    = optim.AdamW(head_params, lr=config["lr_head"],
                               weight_decay=config["weight_decay"])
    scheduler    = CosineAnnealingLR(optimiser, T_max=config["warmup_epochs"])

    # ── History tracking ────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Early stopping state ───────────────────────────────────────────────────
    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    patience_count  = 0
    best_epoch      = 0

    # ── Training loop ──────────────────────────────────────────────────────────
    print("\n── Training ─────────────────────────────────────────────")

    for epoch in range(1, config["num_epochs"] + 1):

        # ── Phase transition: unfreeze backbone after warmup ──────────────────
        if epoch == config["warmup_epochs"] + 1:
            print(f"\n  [Epoch {epoch}] Switching to Phase 2: unfreezing backbone")
            model     = unfreeze_all(model)

            # Rebuild optimiser with smaller learning rate for fine-tuning
            optimiser = optim.AdamW(
                model.parameters(),
                lr=config["lr_finetune"],
                weight_decay=config["weight_decay"]
            )
            # Cosine schedule over the remaining epochs
            remaining_epochs = config["num_epochs"] - config["warmup_epochs"]
            scheduler = CosineAnnealingLR(optimiser, T_max=remaining_epochs)

        # ── Train one epoch ───────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion, device, split="Val"
        )

        scheduler.step()

        # ── Log metrics ───────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch:02d}/{config['num_epochs']}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.1f}%  |  "
            f"Val Loss:   {val_loss:.4f}  Acc: {val_acc*100:.1f}%"
        )

        # ── Save best model ───────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0

            checkpoint_path = os.path.join(config["output_dir"], "best_model.pth")
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         val_loss,
                "val_acc":          val_acc,
                "class_names":      CLASS_NAMES,
                "config":           config,
            }, checkpoint_path)
            print(f"    ✓ Saved new best model (val_acc={val_acc*100:.1f}%)")
        else:
            patience_count += 1

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_count >= config["early_stop_patience"]:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {config['early_stop_patience']} epochs)."
            )
            print(f"  Best model was at epoch {best_epoch}.")
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best validation accuracy : {best_val_acc*100:.1f}%  (epoch {best_epoch})")
    print(f"  Model saved to           : {checkpoint_path}")
    print(f"{'='*60}")

    # ── Plot training curves ───────────────────────────────────────────────────
    plot_training_curves(history, config["output_dir"])

    print("\nNext step: run 'python src/evaluate.py' to evaluate on the test set.")

    return history, best_val_acc


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train(CONFIG)
