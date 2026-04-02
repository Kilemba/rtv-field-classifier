"""
api/main.py — Task 3: Model Serving (FastAPI)
──────────────────────────────────────────────
This file implements a production-grade REST API that exposes the trained
image classifier as an HTTP endpoint.

How the API works:
  - A field worker's device sends a POST request to /predict with an image file
  - The API validates the file, preprocesses the image, and runs inference
  - It returns a JSON response with the predicted category and confidence score
  - Invalid inputs get descriptive error messages, not confusing stack traces

Why FastAPI over Flask?
  - Async-capable: handles multiple simultaneous requests efficiently
  - Auto-generates Swagger documentation at /docs — no manual docs needed
  - Pydantic models validate request/response types automatically
  - Industry standard for ML serving in Python (2023–2025)

Design decisions:
  - Model is loaded ONCE on startup (not per request) — fast inference
  - Input validation rejects corrupt files, wrong formats, empty files early
  - Confidence threshold flag warns downstream consumers of low-confidence predictions
  - All error codes follow HTTP standards (400, 415, 422, 500)

To run locally:
  cd api
  uvicorn main:app --reload --port 8000

API documentation (Swagger UI):
  http://localhost:8000/docs

Author: Caleb Kilemba
"""

import io
import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Make sure we can import from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from model import load_model_for_inference
from dataset import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Logging ──────────────────────────────────────────────────────────────────
# Good logging is essential in production — it helps debug issues without
# needing to reproduce them manually
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

# Path to the saved model checkpoint — the API loads this on startup
CHECKPOINT_PATH = os.getenv(
    "MODEL_CHECKPOINT_PATH",           # read from environment variable if set
    "../outputs/best_model.pth"        # fallback to default path
)

# Confidence below this threshold triggers a "low_confidence" warning in the response
CONFIDENCE_THRESHOLD = 0.60

# Only these image file types are accepted
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp"}

# Maximum image file size: 10MB (field phone photos are typically 1–5MB)
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ─── Global model state ───────────────────────────────────────────────────────
# We store the model and device here so they are loaded once at startup
# and reused for every request — much faster than loading per request

class ModelState:
    model  = None
    device = None

model_state = ModelState()


# ─── Inference preprocessing transform ───────────────────────────────────────

def get_inference_transform() -> A.Compose:
    """
    Preprocessing transform for inference — same as validation transform.
    No augmentation: just resize and normalise.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

INFERENCE_TRANSFORM = get_inference_transform()


# ─── Lifespan handler (startup + shutdown) ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code that runs at startup and shutdown.

    On startup: load the model once so it's ready for requests.
    On shutdown: clean up resources.

    Using the lifespan pattern is the modern FastAPI approach
    (replaces deprecated @app.on_event("startup")).
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logger.info("Starting RTV Field Image Classifier API...")

    model_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on device: {model_state.device}")

    checkpoint = Path(CHECKPOINT_PATH)
    if not checkpoint.exists():
        logger.error(
            f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
            "Train the model first: python src/train.py"
        )
        # We still start the server — health endpoint will work,
        # but /predict will return a 503 until the model is available
        model_state.model = None
    else:
        logger.info(f"Loading model from {CHECKPOINT_PATH}...")
        model_state.model = load_model_for_inference(CHECKPOINT_PATH, model_state.device)
        logger.info("Model loaded successfully. API is ready.")

    yield  # ← application runs here

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("Shutting down API...")
    model_state.model = None


# ─── FastAPI application ───────────────────────────────────────────────────────

app = FastAPI(
    title       = "RTV Field Image Classifier",
    description = (
        "Classifies RTV field check-in images into 9 operational categories. "
        "Built for the Raising The Village VENN department. "
        "Designed for integration with the WorkMate AI ecosystem."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)


# ─── Response models ───────────────────────────────────────────────────────────
# Pydantic models define the exact structure of our API responses.
# FastAPI uses these to auto-validate outputs and generate documentation.

class PredictionResponse(BaseModel):
    category:         str    # e.g. "poultry-house"
    confidence:       float  # e.g. 0.84 — probability between 0.0 and 1.0
    status:           str    # "success" or "low_confidence"
    all_probabilities: dict  # {class_name: probability} for all 9 classes

class HealthResponse(BaseModel):
    status:       str   # "healthy" or "degraded"
    model_loaded: bool
    device:       str
    num_classes:  int
    classes:      list[str]

class ErrorResponse(BaseModel):
    status:  str   # "error"
    message: str   # human-readable explanation


# ─── Helper: preprocess an uploaded image ─────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw image bytes into a model-ready tensor.

    Steps:
    1. Decode bytes → PIL Image
    2. Convert to RGB (handles greyscale, RGBA, etc.)
    3. Apply resize + normalise transform
    4. Add batch dimension (model expects [batch, channels, height, width])

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        Tensor of shape [1, 3, 224, 224] ready for the model.

    Raises:
        ValueError: If the image cannot be decoded.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")                      # ensure 3-channel RGB
        image_np = np.array(image)                        # PIL → NumPy (Albumentations expects this)
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")

    # Apply preprocessing transform
    transformed = INFERENCE_TRANSFORM(image=image_np)
    tensor      = transformed["image"]                    # shape: [3, 224, 224]

    # Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)
    return tensor


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=dict, tags=["General"])
async def root():
    """
    Root endpoint — confirms the API is running.
    """
    return {
        "message": "RTV Field Image Classifier API",
        "version": "1.0.0",
        "docs":    "/docs",
        "predict": "/predict",
        "health":  "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint — reports whether the model is loaded and ready.

    Useful for:
    - Kubernetes readiness/liveness probes
    - Monitoring dashboards
    - CI/CD smoke tests after deployment
    """
    return HealthResponse(
        status       = "healthy" if model_state.model is not None else "degraded",
        model_loaded = model_state.model is not None,
        device       = str(model_state.device) if model_state.device else "none",
        num_classes  = len(CLASS_NAMES),
        classes      = CLASS_NAMES,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid or empty file"},
        415: {"description": "Unsupported file type"},
        503: {"description": "Model not loaded"},
    }
)
async def predict(file: UploadFile = File(..., description="Field image to classify")):
    """
    Classify a field check-in image into one of 9 RTV operational categories.

    **Accepted formats:** JPEG, PNG, WebP

    **Returns:**
    - `category`: The predicted operational category (e.g. "poultry-house")
    - `confidence`: Probability score between 0.0 and 1.0
    - `status`: "success" if confidence ≥ 0.60, else "low_confidence"
    - `all_probabilities`: Full probability distribution across all 9 classes

    **Example request (curl):**
    ```
    curl -X POST "http://localhost:8000/predict" \\
         -F "file=@field_image.jpg"
    ```
    """

    # ── Guard: model must be loaded ───────────────────────────────────────────
    if model_state.model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Model is not loaded. Contact the system administrator."
        )

    # ── Validate file type ────────────────────────────────────────────────────
    # Check both the content-type header and the file extension
    content_type = (file.content_type or "").lower()
    file_ext     = Path(file.filename or "").suffix.lower()

    if content_type not in ALLOWED_CONTENT_TYPES and file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code = 415,
            detail      = (
                f"Unsupported file type: '{file.content_type}'. "
                f"Accepted types: JPEG, PNG, WebP."
            )
        )

    # ── Read image bytes ──────────────────────────────────────────────────────
    image_bytes = await file.read()

    # Validate file is not empty
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code = 400,
            detail      = "Empty file received. Please upload a valid image."
        )

    # Validate file size
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code = 400,
            detail      = (
                f"File too large ({len(image_bytes)/1024/1024:.1f} MB). "
                f"Maximum accepted size: {MAX_FILE_SIZE_MB} MB."
            )
        )

    # ── Preprocess ────────────────────────────────────────────────────────────
    try:
        input_tensor = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code = 400,
            detail      = str(e)
        )

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        input_tensor = input_tensor.to(model_state.device)

        with torch.no_grad():
            logits = model_state.model(input_tensor)              # raw scores
            probs  = torch.softmax(logits, dim=1).squeeze()       # probabilities [9]

        # Get the top prediction
        confidence_tensor, class_idx_tensor = probs.max(dim=0)
        confidence = float(confidence_tensor.cpu())
        class_idx  = int(class_idx_tensor.cpu())
        category   = CLASS_NAMES[class_idx]

        # Build the full probability distribution (all 9 classes)
        all_probs = {
            CLASS_NAMES[i]: round(float(probs[i].cpu()), 4)
            for i in range(len(CLASS_NAMES))
        }

        logger.info(
            f"Prediction: {category} (confidence={confidence:.3f}) "
            f"| File: {file.filename}"
        )

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = "Inference failed. Please try again."
        )

    # ── Determine status ──────────────────────────────────────────────────────
    # Low confidence predictions should be flagged for human review
    # in the RTV field workflow — not silently accepted
    status = "success" if confidence >= CONFIDENCE_THRESHOLD else "low_confidence"

    return PredictionResponse(
        category          = category,
        confidence        = round(confidence, 4),
        status            = status,
        all_probabilities = all_probs,
    )
