# Task 1: Data Analysis & Preparation
## Written Analysis — Caleb Kilemba

---

## Dataset Overview

The dataset consists of field check-in images collected by RTV field workers across programme
sites in Uganda (and potentially Rwanda and DRC). Images are organised into 9 operational
categories that map directly to RTV programme compliance domains:

| Category | Domain | What it represents |
|---|---|---|
| `compost` | Agriculture | Compost pit or heap — adoption of organic soil improvement practice |
| `goat-sheep-pen` | Livestock | Pen/shelter for goats or sheep — structured livestock housing |
| `guinea-pig-shelter` | Livestock | Guinea pig housing — small livestock adoption |
| `liquid-organic` | Agriculture | Liquid organic fertiliser (e.g. fermented manure tea) |
| `organic` | Agriculture | Organic matter — mulching, crop residue management |
| `pigsty` | Livestock | Pig housing/sty — pig-keeping adoption |
| `poultry-house` | Livestock | Chicken/poultry house — poultry-keeping adoption |
| `tippytap` | WASH | Tippy tap handwashing station — WASH compliance indicator |
| `vsla` | Financial | Village Savings and Loan Association — financial resilience indicator |

---

## Data Quality Observations

### Class Imbalance
The dataset is intentionally imbalanced (~150 images per class as stated in the brief, with
some classes having fewer). This reflects the real-world condition at RTV: some programme
practices are adopted more widely and therefore photographed more often. Others (e.g. vsla
group photos or guinea-pig shelters) are naturally less common in the field.

**Challenge:** A naïve model trained without accounting for this imbalance will be biased
toward predicting the majority classes, achieving high overall accuracy while failing on
minority classes — which may be the most important to detect.

**Approach taken:** Two complementary strategies:
1. `WeightedRandomSampler` in the training DataLoader: ensures the model sees every class
   roughly equally per epoch (minority classes are drawn more often)
2. `CrossEntropyLoss(weight=class_weights)`: penalises errors on minority classes more
   heavily during training

### Visual Quality Challenges
Field photographs present real-world challenges not present in clean benchmark datasets:

- **Variable lighting**: Photos taken outdoors across times of day — harsh midday sunlight,
  overcast conditions, backlit subjects
- **Motion blur**: Field workers are moving; camera shake produces slight blurring
- **Partial occlusion**: Animals, people, or other objects may partially cover the subject
- **Variable zoom/distance**: Some photos are close-up, others show the installation from
  several metres away
- **Seasonal variation**: Wet season vs. dry season changes the visual appearance of
  agricultural plots (compost, organic matter)
- **Device variation**: Different camera models produce different colour temperatures,
  resolutions, and compression artefacts

These are not defects — they are the reality of field monitoring at scale. The augmentation
strategy is designed to make the model robust to all of these variations.

### Category Similarity
Several pairs of categories are visually similar and represent the hardest classification
challenges:

- `compost` vs `organic` vs `liquid-organic`: All involve organic matter, often similar
  brown/green colours and textures. The distinguishing features are container type (pit,
  barrel, liquid) and context.
- `goat-sheep-pen` vs `pigsty` vs `guinea-pig-shelter`: All are animal shelters. Size,
  construction material, and the visible animal distinguish them.

These visual similarities will likely produce the most off-diagonal entries in the confusion
matrix, which is expected and documented.

---

## Preprocessing Choices and Rationale

### Resize to 224×224
EfficientNet-B0 (our chosen architecture) expects 224×224 pixel inputs. This is the ImageNet
standard size. All images are resized to this dimension before the model processes them.

### Normalisation (ImageNet statistics)
Pixel values are normalised using ImageNet's mean [0.485, 0.456, 0.406] and standard
deviation [0.229, 0.224, 0.225]. This is required because we start from ImageNet pre-trained
weights — the network's internal representations were calibrated to these input statistics.
Using different statistics would degrade performance.

### RGB Conversion
All images are converted to 3-channel RGB before processing. This handles:
- JPEG photos saved in different colour spaces (CMYK, L)
- PNG files with an alpha (transparency) channel (RGBA → RGB)
- Greyscale images (L → RGB) that might appear occasionally

---

## Augmentation Choices and Rationale

Augmentation is applied **only during training** — never during validation or test evaluation.
This is critical because evaluation must measure performance on unmodified images.

| Augmentation | Probability | Rationale |
|---|---|---|
| Horizontal flip | 50% | Installations can be photographed from either side |
| Rotation ±15° | 40% | Field workers don't hold phones perfectly level |
| Random resized crop | 30% | Simulates different distances/zoom levels |
| Brightness & contrast ±25% | 50% | Outdoor lighting varies dramatically throughout the day |
| Hue/saturation shift | 30% | Seasonal colour variation (wet/dry season) |
| Gaussian blur (3–5px) | 20% | Camera shake and motion blur in the field |

**Not applied:** Extreme geometric distortions (shearing, heavy perspective transforms) —
these would produce unrealistic images that don't resemble field photos.

---

## Train / Validation / Test Split

**Ratios:** 70% train / 15% validation / 15% test  
**Method:** Stratified split — each split contains the same proportion of every class

| Split | Purpose |
|---|---|
| Train | Images the model learns from |
| Validation | Used during training to monitor overfitting and select the best model checkpoint |
| Test | Held out entirely — never seen until final evaluation. The most honest measure of real-world performance. |

**Why stratified?** Without stratification, a minority class with only 80 images could
accidentally land almost entirely in the train set, leaving the validation and test sets
with too few examples to evaluate performance on that class reliably.

---

## How Class Imbalance Was Handled

Three complementary mechanisms:

1. **WeightedRandomSampler** (in DataLoader): Each image in the training set is assigned a
   weight = 1 / (count of its class). Images from small classes are therefore drawn more
   often in each epoch, giving the model balanced exposure to all categories.

2. **Class-weighted CrossEntropyLoss** (in train.py): The loss for misclassifying a minority
   class image is multiplied by a weight proportional to how underrepresented that class is.
   This makes the gradient signal stronger for minority classes.

3. **Stratified splitting**: Ensures minority classes are proportionally represented in
   validation and test sets, not accidentally concentrated in the training set.

Together, these three mechanisms address imbalance at the data level, loss level, and
evaluation level respectively.
