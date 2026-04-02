# Task 2: Model Development — Evaluation Report
## Caleb Kilemba

---

## Model Architecture Choice

**Chosen architecture:** EfficientNet-B0 with transfer learning (pre-trained on ImageNet)

### Why EfficientNet-B0?

EfficientNet-B0 was selected after considering the constraints of this specific problem:

**Dataset size constraint (~1,350 images total across 9 classes):**
Training a deep neural network from scratch requires tens of thousands of images per class
to learn meaningful visual representations. With ~150 images per class, training from scratch
would result in severe overfitting — the model would memorise the training images rather
than learning generalisable features.

Transfer learning is the principled solution: EfficientNet-B0, pre-trained on ImageNet
(1.28 million images, 1,000 classes), has already learned rich visual representations —
edges, textures, shapes, object parts. We only need to adapt these learned features to
RTV's 9 categories.

**EfficientNet-B0 vs alternatives:**

| Architecture | Params (M) | Accuracy (ImageNet Top-1) | Why accepted/rejected |
|---|---|---|---|
| EfficientNet-B0 | 5.3M | 77.1% | **Selected** — best accuracy/size ratio |
| ResNet50 | 25.6M | 76.1% | 5x larger for less accuracy — unnecessary overhead |
| MobileNetV3-S | 2.5M | 67.7% | Smaller but significantly less accurate |
| VGG16 | 138M | 71.6% | 26x larger, outdated, no benefit |
| ViT-S/16 | 22M | 81.4% | High accuracy but needs more data, slower inference |

EfficientNet's "compound scaling" principle — scaling width, depth, and resolution together
rather than independently — makes it structurally efficient for this class of problem.

**Edge deployment consideration:** EfficientNet-B0 at 5.3M parameters exports to a
~20MB ONNX file and ~6MB TFLite INT8 quantised file — well within the size budget for
WorkMate Android integration, which is a stated goal of the role.

### Architecture Modifications

The final classification layer (1,000 classes for ImageNet) was replaced with:
```
Dropout(p=0.3) → Linear(1280 → 9)
```

Dropout (randomly zeroing 30% of neurons during training) is essential with this dataset
size to prevent the head from overfitting even if the backbone is well-regularised.

---

## Training Strategy

### Two-Phase Transfer Learning

**Phase 1 — Warmup (Epochs 1–5):**
The backbone is frozen. Only the new classification head is trained.

*Rationale:* The head is randomly initialised and produces large, noisy gradient updates in
early epochs. If the backbone is unfrozen, these updates corrupt the pre-trained ImageNet
features — a phenomenon called "catastrophic forgetting." Freezing the backbone for the
first 5 epochs lets the head stabilise on RTV-specific features before full fine-tuning begins.

**Phase 2 — Full Fine-Tuning (Epochs 6–20):**
All layers are unfrozen and trained together at a lower learning rate.

*Rationale:* After the head has converged to reasonable representations, we allow all layers
to adapt together. The lower learning rate (1e-4 vs 1e-3) ensures we refine the pre-trained
features rather than overwriting them.

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Batch size | 32 | Standard for this image size; fits comfortably in CPU RAM |
| Phase 1 LR | 1e-3 | Standard for training a randomly initialised head |
| Phase 2 LR | 1e-4 | 10x smaller — fine-tuning, not training from scratch |
| Weight decay | 1e-4 | L2 regularisation — additional overfitting prevention |
| Scheduler | CosineAnnealingLR | Smoothly decays LR to near zero; avoids abrupt drops |
| Early stopping patience | 5 | Halts training if val loss stagnates for 5 epochs |
| Dropout | 0.3 | Conservative — prevents head overfitting without hurting accuracy |

### Loss Function

`CrossEntropyLoss` with per-class weights inversely proportional to class frequency.

Standard CrossEntropyLoss gives equal weight to every image. With class imbalance, this
means the model optimises mainly for the majority class. Weighting the loss ensures the
gradient signal from minority class errors is amplified proportionally.

### Optimiser

`AdamW` — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

Standard Adam accumulates weight decay in the adaptive gradient, which can cause
sub-optimal regularisation. AdamW applies weight decay separately, which is the correct
formulation and consistently outperforms Adam on vision tasks.

---

## Evaluation Metrics

The model is evaluated on the **held-out test set** — images never seen during training
or validation. Results are reported as:

- **Overall accuracy**: Fraction of correctly classified test images
- **Per-class precision**: Of all images predicted as class X, what fraction are truly X?
- **Per-class recall**: Of all images that are truly class X, what fraction did the model find?
- **Per-class F1**: Harmonic mean of precision and recall — the primary metric for imbalanced datasets
- **Macro F1**: Unweighted average F1 across all 9 classes (each class equally weighted)
- **Confusion matrix**: Full cross-tabulation of true vs. predicted classes

**Why F1 is the primary metric, not accuracy:**
A model that always predicts "compost" (the most common class) would achieve high accuracy
but 0% recall on every other category — useless for RTV's operational needs. F1 penalises
both missing true positives (low recall) and over-predicting (low precision).

---

## Expected Results and Qualitative Analysis

With ~1,350 training images and EfficientNet-B0 transfer learning, the expected performance
is in the range of **75–88% overall accuracy** and **0.70–0.85 macro F1**.

### Anticipated Hard Cases

**High confusion expected:**
- `compost` ↔ `organic`: Both show organic matter in outdoor settings. Distinguishing features
  are structural (pit vs. heap, container type). Model may need more annotated examples of
  the structural cues.
- `goat-sheep-pen` ↔ `pigsty`: Both are animal enclosures with similar construction
  materials. The visible animal species is the key discriminator — size, colour, body shape.
- `guinea-pig-shelter` ↔ `poultry-house`: Both are small shelters, sometimes made from
  similar materials (wire mesh, wood).

**High accuracy expected:**
- `tippytap`: Distinctive structure (tippy tap design is unique and visually consistent)
- `vsla`: Group meeting photos are contextually distinct from livestock/agriculture images
- `pigsty`: Pigs are visually distinctive animals rarely confused with others

### Ideas for Further Improvement

1. **More labelled data**: The single most impactful improvement. 300–500 images per class
   would meaningfully close the accuracy gap.

2. **Hard negative mining**: Collect and annotate specifically the images where compost/organic
   or the animal enclosure pairs are most visually similar.

3. **Multi-scale inference (TTA)**: Test-time augmentation — average predictions over
   several augmented versions of each test image — typically adds 1–2% accuracy with no
   additional training.

4. **Semi-supervised learning**: Use the unlabelled field images (which RTV has in abundance)
   to improve feature representations via self-supervised pre-training before fine-tuning.

5. **EfficientNet-B2 or B3**: Larger EfficientNet variants would improve accuracy by 2–4
   percentage points at the cost of slightly larger model size and inference time.

6. **Active learning loop**: Deploy the current model, collect low-confidence predictions,
   prioritise those for annotation, and retrain — the approach described in `docs/task4_architecture.md`.
