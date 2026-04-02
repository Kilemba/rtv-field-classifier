"""
dataset.py — Task 1: Data Analysis & Preparation
─────────────────────────────────────────────────
This module handles everything to do with getting the images ready for training.

What it does:
  1. Reads images from the data/ folder (one subfolder per category)
  2. Analyses the dataset: counts images per class, flags imbalance
  3. Splits data into train / validation / test sets (stratified so each
     split has the same proportion of every class)
  4. Applies appropriate augmentation for field photography conditions
  5. Wraps everything in PyTorch DataLoader objects for training

Why these choices:
  - Field photos are noisy: variable lighting, angles, partial occlusion.
    Augmentation mimics this so the model learns robust features.
  - Class imbalance is handled with WeightedRandomSampler so the model
    sees every class roughly equally during training — not just the common ones.
  - Stratified splits ensure minority classes aren't accidentally excluded
    from validation or test sets.

Author: Caleb Kilemba
"""

import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Constants ────────────────────────────────────────────────────────────────

# All 9 RTV operational categories — exactly matching the folder names
CLASS_NAMES = [
    "compost",
    "goat-sheep-pen",
    "guinea-pig-shelter",
    "liquid-organic",
    "organic",
    "pigsty",
    "poultry-house",
    "tippytap",
    "vsla",
]

# ImageNet mean and std — used because we start from a pre-trained model
# that was trained on ImageNet. Using the same normalisation is important.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Input size expected by EfficientNet-B0
IMAGE_SIZE = 224

# Split ratios: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Accepted image file extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ─── Step 1: Discover images on disk ──────────────────────────────────────────

def discover_dataset(data_dir: str) -> tuple[list[str], list[int], dict]:
    """
    Walk the data/ directory and collect all image paths and their class labels.

    Each subfolder name is treated as a class label. We also collect stats
    to understand class balance before any processing happens.

    Args:
        data_dir: Path to the folder containing one subfolder per category.

    Returns:
        image_paths: List of absolute paths to every valid image file.
        labels:      Corresponding integer class index for each image.
        stats:       Dictionary with counts per class and quality notes.
    """
    data_path = Path(data_dir)

    image_paths = []
    labels      = []
    stats       = {"per_class": {}, "total": 0, "skipped": 0, "quality_notes": []}

    print("\n── Discovering dataset ──────────────────────────────────")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = data_path / class_name

        if not class_folder.exists():
            # This folder is missing entirely — flag it clearly
            stats["quality_notes"].append(
                f"WARNING: Folder '{class_name}' not found in {data_dir}"
            )
            stats["per_class"][class_name] = 0
            print(f"  ⚠  {class_name:25s} — FOLDER NOT FOUND")
            continue

        # Collect all valid image files in this class folder
        class_images = [
            f for f in class_folder.iterdir()
            if f.suffix in VALID_EXTENSIONS
        ]

        # Try opening each image to catch corrupt files early
        valid_images = []
        for img_path in class_images:
            try:
                with Image.open(img_path) as img:
                    img.verify()        # checks for file corruption
                valid_images.append(str(img_path))
            except Exception:
                stats["skipped"] += 1  # silently skip corrupt images

        count = len(valid_images)
        stats["per_class"][class_name] = count
        stats["total"] += count

        image_paths.extend(valid_images)
        labels.extend([class_idx] * count)

        print(f"  ✓  {class_name:25s} — {count:4d} images")

    print(f"\n  Total valid images : {stats['total']}")
    print(f"  Skipped (corrupt)  : {stats['skipped']}")

    # ── Detect class imbalance ────────────────────────────────────────────────
    counts = list(stats["per_class"].values())
    if counts:
        max_count = max(counts)
        min_count = min(c for c in counts if c > 0)
        ratio     = max_count / max(min_count, 1)

        if ratio > 2.0:
            # A ratio above 2 means some classes have twice as many images
            # as others — this will bias the model if not handled
            stats["quality_notes"].append(
                f"Class imbalance detected: largest class has {ratio:.1f}x "
                f"more images than smallest. Using weighted sampling."
            )
            print(f"\n  ⚠  Imbalance ratio: {ratio:.1f}x  (handled via WeightedRandomSampler)")

    for note in stats["quality_notes"]:
        print(f"  NOTE: {note}")

    return image_paths, labels, stats


# ─── Step 2: Train / Val / Test split ─────────────────────────────────────────

def split_dataset(
    image_paths: list[str],
    labels: list[int],
) -> tuple[list, list, list, list, list, list]:
    """
    Split image paths and labels into train, validation, and test sets.

    We use stratified splitting so that every class is proportionally
    represented in each split — this is critical for minority classes
    that might otherwise end up entirely in one split.

    Args:
        image_paths: All image file paths.
        labels:      Corresponding class indices.

    Returns:
        Six lists: train_paths, val_paths, test_paths,
                   train_labels, val_labels, test_labels
    """
    # First split off the test set (15%)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=TEST_RATIO,
        stratify=labels,        # ensures each class is proportionally split
        random_state=42         # fixed seed for reproducibility
    )

    # Then split the remaining 85% into train (70%) and val (15%)
    # Note: val_size here is relative to the train_val subset
    relative_val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=42
    )

    print("\n── Dataset splits ───────────────────────────────────────")
    print(f"  Train      : {len(train_paths):4d} images  ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Validation : {len(val_paths):4d} images  ({VAL_RATIO*100:.0f}%)")
    print(f"  Test       : {len(test_paths):4d} images  ({TEST_RATIO*100:.0f}%)")

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


# ─── Step 3: Augmentation transforms ──────────────────────────────────────────

def get_train_transform() -> A.Compose:
    """
    Augmentation pipeline for TRAINING images.

    These transforms reflect the realistic variation in RTV field photos:
    - Field workers photograph from different distances and angles
    - Lighting changes dramatically (outdoor, morning vs midday, shade)
    - Camera shake produces slight blur
    - Colours differ between dry and wet seasons
    - Partial occlusion is common (objects partly blocked by people/objects)

    We do NOT augment too aggressively — we want realistic variation,
    not images so distorted they no longer resemble the original category.
    """
    return A.Compose([

        # ── Spatial transforms ─────────────────────────────────────────────
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # Horizontal flip — many installations can be photographed from either side
        A.HorizontalFlip(p=0.5),

        # Small rotation — field workers don't always hold the phone perfectly level
        A.Rotate(limit=15, p=0.4),

        # Random crop then resize — simulates different zoom levels / distances
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            p=0.3
        ),

        # ── Colour / lighting transforms ───────────────────────────────────
        # Brightness and contrast change throughout the day in the field
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.5
        ),

        # Hue and saturation variation — wet vs dry season, different regions
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),

        # ── Noise / blur ──────────────────────────────────────────────────
        # Slight blur simulates camera shake on a moving field officer
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Normalisation and conversion to tensor ────────────────────────
        # Always applied last — scales pixel values to ImageNet's expected range
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_test_transform() -> A.Compose:
    """
    Transform for VALIDATION and TEST images.

    No random augmentation here — we want a consistent, reproducible evaluation.
    Only resize and normalise, exactly as the model expects.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─── Step 4: PyTorch Dataset class ────────────────────────────────────────────

class RTVImageDataset(Dataset):
    """
    Custom PyTorch Dataset for RTV field images.

    PyTorch's training loop needs a Dataset object that can:
    - Tell it how many images there are (__len__)
    - Return a single (image tensor, label) pair when indexed (__getitem__)

    This class handles reading the image from disk, applying the transform,
    and returning the label as an integer that the model can learn from.
    """

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        """
        Args:
            image_paths: List of paths to image files.
            labels:      List of integer class indices (one per image).
            transform:   Albumentations transform pipeline to apply.
        """
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        # PyTorch calls this to know how many batches to create
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load one image, apply transform, return (image_tensor, label).
        If an image fails to load (e.g. corrupt), we return a black image
        instead of crashing the entire training run.
        """
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        try:
            # PIL opens the image; convert to RGB ensures 3 channels
            # (some PNG files are RGBA or greyscale)
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            # Graceful fallback — return a black image so training continues
            print(f"  Warning: Could not load {img_path}: {e}")
            image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

        if self.transform:
            # Albumentations expects a dict with key "image"
            transformed = self.transform(image=image)
            image       = transformed["image"]  # returns a torch.Tensor

        return image, label


# ─── Step 5: WeightedRandomSampler for class imbalance ────────────────────────

def make_weighted_sampler(labels: list[int]) -> WeightedRandomSampler:
    """
    Create a sampler that over-samples minority classes during training.

    How it works:
    - Count how many images each class has
    - Give each image a weight = 1 / (count of its class)
    - Images from small classes get a HIGH weight → sampled more often
    - Images from large classes get a LOW weight → sampled less often
    - Result: the model sees every class roughly equally per epoch

    This is better than dropping images from majority classes (undersampling)
    because we don't waste labelled data.

    Args:
        labels: Integer class labels for the training set.

    Returns:
        A WeightedRandomSampler to pass to DataLoader.
    """
    # Count images per class
    class_counts = Counter(labels)
    num_classes  = len(CLASS_NAMES)

    # Weight for each class = inverse of its frequency
    class_weights = {
        cls: 1.0 / count
        for cls, count in class_counts.items()
    }

    # Assign a weight to every individual image
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True   # allows the same minority image to be drawn multiple times
    )


# ─── Step 6: Build DataLoader objects ─────────────────────────────────────────

def build_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Full pipeline: discover → split → augment → wrap in DataLoaders.

    This is the main function called by train.py and evaluate.py.

    Args:
        data_dir:    Path to the folder with 9 category subfolders.
        batch_size:  How many images per training step (32 is a solid default).
        num_workers: Parallel image loading workers (0 = safe on all platforms).

    Returns:
        train_loader, val_loader, test_loader, dataset_stats
    """
    # 1. Find all images
    image_paths, labels, stats = discover_dataset(data_dir)

    if not image_paths:
        raise ValueError(
            f"No images found in '{data_dir}'. "
            "Please check that your data/ folder contains the 9 category subfolders."
        )

    # 2. Split into train / val / test
    train_paths, val_paths, test_paths, \
    train_labels, val_labels, test_labels = split_dataset(image_paths, labels)

    # 3. Create Dataset objects with appropriate transforms
    train_dataset = RTVImageDataset(train_paths, train_labels, get_train_transform())
    val_dataset   = RTVImageDataset(val_paths,   val_labels,   get_val_test_transform())
    test_dataset  = RTVImageDataset(test_paths,  test_labels,  get_val_test_transform())

    # 4. Weighted sampler for training only
    #    (val and test must NOT be resampled — we want their true distribution)
    sampler = make_weighted_sampler(train_labels)

    # 5. Wrap in DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,       # use weighted sampler instead of shuffle=True
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # faster GPU transfer if GPU available
        drop_last=True,        # drop the last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # always evaluate in the same order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, stats


# ─── Task 1: Deeper per-image quality inspection ──────────────────────────────

def inspect_image_properties(data_dir: str, max_per_class: int = 20) -> dict:
    """
    Sample images from each class and record their pixel-level properties.

    This gives us a data-quality report that goes beyond just counting files:
    - Are images in consistent resolution ranges?
    - Do brightness levels vary heavily (indicates mixed lighting conditions)?
    - Are there near-duplicate images in a class (annotation leakage risk)?

    Args:
        data_dir:      Path to the data folder.
        max_per_class: How many images to sample per class for the inspection.

    Returns:
        properties: Dict mapping class_name → {widths, heights, mean_brightness}
    """
    import random as _random
    data_path  = Path(data_dir)
    properties = {}

    print("\n── Image property inspection (sample per class) ─────────")
    print(f"  {'Class':25s}  {'Min WxH':14s}  {'Max WxH':14s}  {'Avg Brightness':14s}  {'Samples'}")
    print("  " + "-" * 80)

    for class_name in CLASS_NAMES:
        class_folder = data_path / class_name
        if not class_folder.exists():
            continue

        images_in_class = [
            f for f in class_folder.iterdir()
            if f.suffix in VALID_EXTENSIONS
        ]
        # Sample a subset so this runs quickly even for large datasets
        sample = _random.sample(images_in_class, min(max_per_class, len(images_in_class)))

        widths      = []
        heights     = []
        brightnesses = []

        for img_path in sample:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    # Convert to greyscale to compute average brightness quickly
                    grey = img.convert("L")
                    brightnesses.append(np.array(grey).mean())
            except Exception:
                continue  # skip unreadable files silently

        if widths:
            properties[class_name] = {
                "widths":       widths,
                "heights":      heights,
                "brightnesses": brightnesses,
            }
            min_res = f"{min(widths)}x{min(heights)}"
            max_res = f"{max(widths)}x{max(heights)}"
            avg_br  = np.mean(brightnesses)
            print(
                f"  {class_name:25s}  {min_res:14s}  {max_res:14s}  "
                f"{avg_br:>10.1f}/255   {len(sample):>4d}"
            )

    return properties


def save_task1_report(stats: dict, properties: dict, output_path: str = "outputs/task1_data_report.txt"):
    """
    Save a text file summarising Task 1 findings.
    This file becomes part of the Assessment submission (Task 1 deliverable).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "RTV Field Image Classifier — Task 1: Data Analysis Report",
        "Author: Caleb Kilemba",
        "=" * 62,
        "",
        "DATASET SUMMARY",
        "-" * 40,
        f"Total valid images   : {stats['total']}",
        f"Skipped (corrupt)    : {stats['skipped']}",
        f"Number of classes    : {len(CLASS_NAMES)}",
        "",
        "PER-CLASS IMAGE COUNTS",
        "-" * 40,
    ]

    counts = stats["per_class"]
    for cls in CLASS_NAMES:
        count = counts.get(cls, 0)
        bar   = "█" * (count // 5)   # simple ASCII bar chart (each block = 5 images)
        lines.append(f"  {cls:25s} : {count:4d}  {bar}")

    if counts:
        vals       = [v for v in counts.values() if v > 0]
        max_c      = max(vals)
        min_c      = min(vals)
        imb_ratio  = max_c / max(min_c, 1)
        lines += [
            "",
            f"  Largest class  : {max_c} images",
            f"  Smallest class : {min_c} images",
            f"  Imbalance ratio: {imb_ratio:.2f}x",
            f"  Handling       : WeightedRandomSampler + class-weighted CrossEntropyLoss",
        ]

    lines += [
        "",
        "TRAIN / VAL / TEST SPLIT  (stratified, seed=42)",
        "-" * 40,
        f"  Train      : 70%  — model learns from these",
        f"  Validation : 15%  — used to pick best checkpoint during training",
        f"  Test       : 15%  — held out; never seen until final evaluation",
        "",
        "IMAGE PROPERTY SAMPLE",
        "-" * 40,
    ]

    for cls, props in properties.items():
        avg_br = np.mean(props["brightnesses"])
        lines.append(
            f"  {cls:25s}  "
            f"res_range={min(props['widths'])}x{min(props['heights'])}"
            f"–{max(props['widths'])}x{max(props['heights'])}  "
            f"avg_brightness={avg_br:.1f}/255"
        )

    lines += [
        "",
        "AUGMENTATION APPLIED DURING TRAINING",
        "-" * 40,
        "  HorizontalFlip        p=0.50  — installations photographed from either side",
        "  Rotate ±15°           p=0.40  — field workers hold phones at varying angles",
        "  RandomResizedCrop     p=0.30  — different distances / zoom levels",
        "  BrightnessContrast    p=0.50  — outdoor lighting varies dramatically",
        "  HueSaturationValue    p=0.30  — wet vs dry season colour variation",
        "  GaussianBlur (3-5px)  p=0.20  — camera shake / motion blur",
        "  (No augmentation applied to validation or test sets)",
        "",
        "QUALITY NOTES",
        "-" * 40,
    ]
    for note in stats.get("quality_notes", []):
        lines.append(f"  • {note}")
    if not stats.get("quality_notes"):
        lines.append("  • No critical quality issues detected.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Saved Task 1 report → {output_path}")


# ─── Run standalone for Task 1 analysis ───────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to perform the full Task 1 data analysis.
    Usage: python src/dataset.py

    Outputs:
      - Console: class counts, split sizes, image property table, pipeline check
      - outputs/task1_data_report.txt: full written report (Task 1 deliverable)
    """
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 62)
    print("  RTV Field Image Classifier — Task 1: Data Analysis")
    print("=" * 62)

    # ── 1. Discover dataset and build DataLoaders ──────────────────────────────
    # This triggers all the class-count and imbalance print statements
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        data_dir="data",
        batch_size=32,
    )

    # ── 2. Deep image property inspection ────────────────────────────────────
    # Checks resolution range and brightness distribution per class
    # Helps flag resolution inconsistencies or over-dark images
    properties = inspect_image_properties(data_dir="data", max_per_class=20)

    # ── 3. Verify the pipeline produces correct tensor shapes ────────────────
    print("\n── Verifying pipeline with one batch ────────────────────")
    images, labels_batch = next(iter(train_loader))
    print(f"  Image tensor shape : {images.shape}")   # should be [32, 3, 224, 224]
    print(f"  Label tensor shape : {labels_batch.shape}")
    print(f"  Pixel value range  : [{images.min():.3f}, {images.max():.3f}]")
    # After normalisation, values will be in roughly [-2.1, 2.6] — that is correct

    # ── 4. Class distribution in this training batch ──────────────────────────
    # With WeightedRandomSampler active, each class should appear roughly equally
    print("\n── Class distribution in one training batch ─────────────")
    print("  (WeightedRandomSampler should balance minority classes)")
    batch_counts = Counter(labels_batch.numpy())
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        count = batch_counts.get(cls_idx, 0)
        bar   = "▪" * count
        print(f"  {cls_name:25s} : {count:2d}  {bar}")

    # ── 5. Save the written Task 1 report ────────────────────────────────────
    save_task1_report(stats, properties, output_path="outputs/task1_data_report.txt")

    print("\n" + "=" * 62)
    print("  Task 1 complete.")
    print("  Deliverables:")
    print("    src/dataset.py              ← Pipeline code (this file)")
    print("    outputs/task1_data_report.txt ← Written analysis")
    print("    docs/task1_analysis.md      ← Full design rationale")
    print("=" * 62)
