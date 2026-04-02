"""
model.py — Model Architecture Definition
─────────────────────────────────────────
This file defines the neural network we use for classifying RTV field images.

Architecture choice: EfficientNet-B0 with transfer learning
─────────────────────────────────────────────────────────────
We did NOT design a network from scratch. Instead, we start from EfficientNet-B0
which has already been trained on 1.2 million images (ImageNet). It already
"knows" how to recognise textures, edges, shapes, and object parts.

We then replace only the final classification layer (the "head") with a new one
that outputs 9 probabilities — one per RTV category. We then fine-tune the whole
network on RTV images.

Why EfficientNet-B0 over other options?
  - ResNet50: Larger, slower, uses more memory — unnecessary for this task size
  - MobileNetV3: Slightly faster but slightly less accurate
  - EfficientNet-B0: Best accuracy-to-size ratio in its class. Designed to scale
    efficiently. Important for future TFLite/ONNX edge deployment.
  - VGG16: Too large, outdated, no benefit here

Why transfer learning?
  - We have ~150 images per class (~1,350 total). That is far too few to train
    a deep network from random weights — it would overfit badly.
  - A network pre-trained on ImageNet already understands visual features.
    We just need to teach it RTV-specific categories.
  - In practice, transfer learning achieves better results with 10x less data
    and 10x faster training.

Author: Caleb Kilemba
"""

import torch
import torch.nn as nn
import timm                    # timm = "PyTorch Image Models" — huge library of pretrained models


def build_model(num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    """
    Build and return the EfficientNet-B0 classifier.

    The architecture has two parts:
    1. The BACKBONE: the bulk of the network (pre-trained, frozen initially)
       This extracts rich visual features from any image.
    2. The HEAD: the final layer we replace (randomly initialised, trained by us)
       This maps the extracted features to our 9 RTV categories.

    Args:
        num_classes: Number of output categories (9 for RTV).
        pretrained:  Whether to load ImageNet pre-trained weights.
                     Always True for training; can be False for unit tests.

    Returns:
        model: A PyTorch nn.Module ready for training or inference.
    """

    # ── Load pre-trained EfficientNet-B0 ──────────────────────────────────────
    # timm loads the architecture AND the pre-trained weights in one call.
    # num_classes=9 automatically replaces the final layer for us.
    model = timm.create_model(
        "efficientnet_b0",
        pretrained=pretrained,   # downloads ~20MB weights on first run
        num_classes=num_classes, # replaces the 1000-class ImageNet head with 9-class head
    )

    # ── Dropout for regularisation ────────────────────────────────────────────
    # With only ~1,000 training images, overfitting is the main risk.
    # We wrap the classifier head with dropout to reduce this.
    # Dropout randomly zeroes 30% of neurons during training, forcing the
    # network to not rely on any single feature — acts as built-in regularisation.
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),                  # regularisation
        nn.Linear(in_features, num_classes) # final prediction layer
    )

    return model


def count_parameters(model: nn.Module) -> dict:
    """
    Count the total and trainable parameters in the model.
    Useful for understanding model size and what's being optimised.

    Returns a dict with:
        total:     all parameters
        trainable: parameters that have gradients (will be updated)
        frozen:    parameters with no gradients (backbone, if frozen)
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }


def freeze_backbone(model: nn.Module) -> nn.Module:
    """
    Freeze all layers EXCEPT the final classifier head.

    Why freeze initially?
    In the first few training epochs, the randomly initialised head produces
    large gradient updates. If the backbone is unfrozen, these large updates
    can destroy the pre-trained ImageNet features (called "catastrophic forgetting").

    Strategy:
    - Epochs 1-5: Freeze backbone, train only the head (head learns RTV patterns)
    - Epochs 6+:  Unfreeze backbone, train everything at a small learning rate
                  (fine-tune all features together)

    Args:
        model: The EfficientNet model.

    Returns:
        The same model with backbone parameters frozen.
    """
    for name, param in model.named_parameters():
        # Everything that is NOT part of the classifier head gets frozen
        if "classifier" not in name:
            param.requires_grad = False

    params = count_parameters(model)
    print(f"  Backbone frozen. Trainable parameters: {params['trainable']:,} / {params['total']:,}")
    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """
    Unfreeze all parameters for full fine-tuning.
    Called after the initial warm-up epochs.

    Args:
        model: The EfficientNet model (with frozen backbone).

    Returns:
        The same model with all parameters unfrozen.
    """
    for param in model.parameters():
        param.requires_grad = True

    params = count_parameters(model)
    print(f"  Backbone unfrozen. Trainable parameters: {params['trainable']:,} / {params['total']:,}")
    return model


def load_model_for_inference(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load a saved model checkpoint for inference (prediction), not training.

    This is what the FastAPI endpoint calls to load the trained model.
    The model is set to eval() mode which:
    - Disables dropout (we want deterministic predictions)
    - Uses running statistics in BatchNorm (not per-batch statistics)

    Args:
        checkpoint_path: Path to the saved .pth checkpoint file.
        device:          'cpu' or 'cuda' — where to load the model.

    Returns:
        model: Ready-to-use model in evaluation mode.
    """
    # Build the architecture (same structure as during training)
    model = build_model(num_classes=9, pretrained=False)

    # Load the saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # The checkpoint may contain extra metadata — extract just the weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()   # critical: switches off dropout and sets BatchNorm to eval mode

    return model


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check: build the model and run a dummy image through it.
    Usage: python src/model.py
    """
    print("Building EfficientNet-B0 model...")
    model  = build_model(num_classes=9, pretrained=False)  # pretrained=False skips download
    params = count_parameters(model)

    print(f"  Total parameters   : {params['total']:,}")
    print(f"  Trainable params   : {params['trainable']:,}")

    # Run a single fake image through the model to verify shapes are correct
    dummy_input = torch.randn(1, 3, 224, 224)  # 1 image, 3 channels, 224x224
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape        : {dummy_input.shape}")
    print(f"  Output shape       : {output.shape}")  # should be [1, 9]
    print("  ✓ Model architecture is correct.")
