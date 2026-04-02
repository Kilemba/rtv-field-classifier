# Task 4: Deployment & MLOps Architecture Notes
## Caleb Kilemba

---

## CI/CD Approach (If Given More Time)

A production ML system at RTV needs a CI/CD pipeline that covers both the code and the model.
My approach would use **GitHub Actions** as the CI/CD engine:

**On every pull request:**
- Run linting (flake8/black) and unit tests (pytest) against the API and src/ code
- Run a "smoke test" training run on 10% of the data to verify the pipeline runs end-to-end
  without errors — catching broken imports, shape mismatches, and schema changes early

**On merge to main:**
- Run the full evaluation suite against the latest model checkpoint on the validation set
- Automatically enforce performance thresholds: if the new model's macro F1 drops below
  the registered baseline, the pipeline fails and no deployment occurs
- Rebuild and push the Docker image to a container registry (GCR or ECR)
- Deploy to staging environment (Cloud Run or Kubernetes staging namespace)
- Run automated integration tests against the live staging endpoint

**On manual approval:**
- Promote from staging to production with zero-downtime rolling deployment

---

## Model Versioning and Management in Production

Model versioning is handled through **Weights & Biases Artifacts**, which I use at my
current role at Data Science East Africa:

- Every training run produces a versioned model artifact (e.g. `rtv-classifier:v1.2`)
- The artifact stores the model weights alongside the exact dataset version used, all
  hyperparameters, evaluation metrics, and the Git commit SHA of the training code
- A model "alias" (`production`, `staging`, `archived`) tracks which version is active
  in each environment without changing the underlying version numbers
- Promoting a model to production means updating the alias — not replacing files

For the Databricks/Snowflake data warehouse integration:
- Every inference result is logged with the `model_version` field (e.g. `efficientnet-b0-v1.2`)
- This allows retrospective analysis: "how did predictions change between model versions?"
- When a new model version is deployed, we can compare its outputs on the same set of
  images against the previous version to validate consistency before full rollout

---

## Monitoring Model Performance Over Time

Three layers of monitoring for RTV's production deployment:

**1. Prediction confidence monitoring (daily):**
An Airflow DAG queries the Bronze-layer predictions table daily and computes the
distribution of confidence scores per class. If average confidence for a category drops
below 0.55 (our low-confidence threshold), an alert fires to Slack/email.

Falling confidence is often the first signal that the model is encountering images from a
new distribution it wasn't trained on — new device type, new region, seasonal change.

**2. Label distribution monitoring (weekly):**
Compare the distribution of predicted labels this week vs. the 4-week rolling baseline.
A sudden shift (e.g. 40% of images predicted as "compost" when it was 15% last month)
either reflects a real programme change or a model failure — both warrant investigation.

**3. Ground-truth validation (monthly):**
A random sample of model predictions is manually reviewed by field staff and compared to
the model's label. This gives us a "ground truth recall" estimate over time.
When this drops below 80%, a retraining cycle is triggered.

**Retraining trigger conditions:**
- Weekly confidence distribution shift >15% from baseline
- Ground-truth recall drops below 80% in monthly audit
- New programme category is added (e.g. RTV adds a new practice domain)
- Deployment context changes (new country, new device type, new season)

---

## Scaling with Increased Request Volume

The current architecture handles light to moderate load. For higher volume:

**Horizontal scaling with Cloud Run / Kubernetes:**
- Cloud Run automatically scales the number of container instances based on request rate
- Each instance handles one request at a time (CPU inference is synchronous)
- For burst traffic (e.g. all field workers uploading at end-of-day), set minimum instances
  to 2 and maximum to 10 — Cloud Run scales between them within seconds

**GPU deployment (for high sustained load):**
- If request volume exceeds ~50 images/second sustained, migrate to a GPU-backed
  deployment on GCP Vertex AI or a GPU Kubernetes node
- ONNX Runtime with CUDA achieves ~3ms/image on a single V100 vs. ~50ms on CPU
- Batch inference: group incoming requests into mini-batches of 8–16 images for GPU efficiency

**Edge deployment (WorkMate integration):**
For fully offline field use, the model is exported to TFLite (INT8 quantised):
- Model size: ~6MB (fits easily in an Android APK)
- Inference time: <100ms on a mid-range Android device (no internet required)
- The WorkMate app bundles the TFLite file and runs inference locally
- Results are synced to the data warehouse when connectivity is restored

This is the target production architecture for RTV's field context — the API serves as
the online fallback and the source of truth for model versioning, while TFLite handles
the offline-first field reality.

---

## Integration with RTV Data Warehouse

Model predictions are logged to the **Databricks medallion architecture** (Bronze → Silver → Gold):

**Bronze (raw predictions):**
```sql
CREATE TABLE bronze.cv_predictions (
    prediction_id   STRING,      -- UUID
    household_id    STRING,
    image_path      STRING,      -- GCS path
    predicted_class STRING,
    confidence      FLOAT,
    model_version   STRING,
    device_id       STRING,
    created_at      TIMESTAMP
);
```

**Silver (cleaned, joined):**
DBT model filters predictions with confidence < 0.60, joins with the household master
table to add village_id and cohort_id, and deduplicates to one prediction per household
per category per day.

**Gold (business-ready):**
DBT model aggregates to weekly adoption rates per village and programme domain —
directly queried by Looker dashboards used by programme managers.
