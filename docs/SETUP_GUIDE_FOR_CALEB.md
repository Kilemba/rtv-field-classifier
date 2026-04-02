# YOUR COMPLETE STEP-BY-STEP GUIDE
## How to Set Up, Run, and Submit the RTV Technical Assessment
### Written for someone running this for the first time — Caleb Kilemba

---

## Before Anything Else — What Is This Project?

You have been given 4 tasks:

| Task | What you build | Files involved |
|---|---|---|
| Task 1 | Analyse the image data and prepare it for training | `src/dataset.py` |
| Task 2 | Train an AI model to classify the images | `src/train.py`, `src/evaluate.py`, `src/model.py` |
| Task 3 | Wrap the model in a web API so anyone can use it | `api/main.py` |
| Task 4 | Notes on deployment and CI/CD (optional) | `docs/task4_architecture.md`, `Dockerfile` |

All the code is already written for you in this project. Your job is to:
1. Put your images in the right folder
2. Install the required tools
3. Run each script in order
4. Submit the whole project to GitHub

---

## PHASE 1: SET UP YOUR COMPUTER

### Step 1.1 — Check that Python is installed

Open a terminal (on Windows: search for "Command Prompt" or "PowerShell").
Type this and press Enter:

```
python --version
```

You should see something like `Python 3.10.5` or `Python 3.11.3`.
If you see an error, download Python from https://python.org and install it.
Make sure to tick "Add Python to PATH" during installation.

---

### Step 1.2 — Download or copy this project to your computer

If you received this as a ZIP file:
- Unzip it to a convenient location (e.g. your Desktop or Documents)
- The unzipped folder should be named `rtv_project`

If you are cloning from GitHub:
```
git clone https://github.com/YOUR_USERNAME/rtv-classifier.git
cd rtv-classifier
```

---

### Step 1.3 — Open a terminal IN the project folder

This is important — all commands below must be run from INSIDE the `rtv_project` folder.

**On Windows:**
1. Open File Explorer
2. Navigate to the `rtv_project` folder
3. Click in the address bar at the top
4. Type `cmd` and press Enter
   → A black terminal window will open already inside your folder

**On Mac:**
1. Open Finder
2. Navigate to the `rtv_project` folder
3. Right-click on the folder
4. Select "New Terminal at Folder"

---

### Step 1.4 — Install all required Python libraries

In the terminal, type:

```
pip install -r requirements.txt
```

This downloads and installs everything the project needs. It may take 3–5 minutes
the first time (it downloads PyTorch which is large).

If you see a "permission denied" error, try:
```
pip install --user -r requirements.txt
```

---

## PHASE 2: ADD YOUR IMAGES

### Step 2.1 — Put your images in the data/ folder

Your 9 image folders (compost, goat-sheep-pen, etc.) need to go inside the `data/` folder.

The folder structure must look exactly like this:

```
rtv_project/
  data/
    compost/
      image1.jpg
      image2.jpg
      ... (all compost images here)
    goat-sheep-pen/
      image1.jpg
      ...
    guinea-pig-shelter/
      image1.jpg
      ...
    liquid-organic/
      ...
    organic/
      ...
    pigsty/
      ...
    poultry-house/
      ...
    tippytap/
      ...
    vsla/
      ...
```

**IMPORTANT:** The folder names must be spelled exactly as above (all lowercase, hyphens not
underscores). The code looks for these exact names.

---

## PHASE 3: RUN TASK 1 — DATA ANALYSIS

### Step 3.1 — Run the data analysis script

In your terminal (make sure you are in the `rtv_project` folder), type:

```
python src/dataset.py
```

**What you will see:**
```
──────────────────────────────────────────────
  RTV Image Dataset — Task 1: Data Analysis
──────────────────────────────────────────────

── Discovering dataset ──────────────────────────────────
  ✓  compost                    — 147 images
  ✓  goat-sheep-pen             — 152 images
  ✓  guinea-pig-shelter         —  98 images
  ✓  liquid-organic             — 143 images
  ✓  organic                    — 151 images
  ✓  pigsty                     — 134 images
  ✓  poultry-house              — 149 images
  ✓  tippytap                   — 145 images
  ✓  vsla                       — 112 images

  Total valid images : 1231
  Skipped (corrupt)  : 0

  ⚠  Imbalance ratio: 1.6x  (handled via WeightedRandomSampler)

── Dataset splits ──────────────────────────────────
  Train      :  862 images  (70%)
  Validation :  184 images  (15%)
  Test       :  185 images  (15%)

── Sampling one batch to verify pipeline ────────────────
  Image tensor shape : torch.Size([32, 3, 224, 224])
  Label tensor shape : torch.Size([32])
  Pixel value range  : [-2.118, 2.640]

  ✓ Pipeline is working correctly.
```

**If you see an error** about a folder not being found, double-check your data/ folder
structure in Step 2.1.

**This completes Task 1. The written analysis is in `docs/task1_analysis.md`.**

---

## PHASE 4: RUN TASK 2 — TRAIN THE MODEL

### Step 4.1 — Start training

In your terminal, type:

```
python src/train.py
```

Training will start. You will see a progress bar for each epoch:

```
──────────────────────────────────────────────
  RTV Image Classifier — Model Training
──────────────────────────────────────────────
  Device        : cpu
  Epochs        : 20 (warmup: 5)
  Batch size    : 32

── Loading data ─────────────────────────────────────────
  [... dataset discovery prints ...]

── Building model ───────────────────────────────────────
  Backbone frozen. Trainable parameters: 10,249 / 5,288,548

── Training ─────────────────────────────────────────────
  Epoch 01/20  |  Train Loss: 1.8432  Acc: 42.1%  |  Val Loss: 1.4211  Acc: 58.3%
    ✓ Saved new best model (val_acc=58.3%)
  Epoch 02/20  |  Train Loss: 1.5231  Acc: 54.7%  |  Val Loss: 1.2044  Acc: 64.1%
    ✓ Saved new best model (val_acc=64.1%)
  ...
  [Epoch 06] Switching to Phase 2: unfreezing backbone
  ...
  Epoch 18/20  |  Train Loss: 0.3421  Acc: 88.2%  |  Val Loss: 0.6112  Acc: 82.4%
```

**How long does training take?**
- On a normal laptop CPU: approximately 20–45 minutes for 20 epochs
- If you have an NVIDIA GPU, PyTorch will use it automatically and training takes 3–5 minutes
- You can stop training early (Ctrl+C) — the best model so far is already saved

**When training finishes you will see:**
```
══════════════════════════════════════════════════════════════
  Training complete.
  Best validation accuracy : 82.4%  (epoch 18)
  Model saved to           : outputs/best_model.pth
══════════════════════════════════════════════════════════════
  Saved training curves → outputs/training_curves.png
```

---

### Step 4.2 — Run evaluation on the test set

Once training is done, type:

```
python src/evaluate.py
```

This loads the saved best model and tests it on images it has NEVER seen before.

**You will see:**
```
══════════════════════════════════════════════════════════════
  Test Set Evaluation Results
══════════════════════════════════════════════════════════════
  Overall Accuracy  : 81.6%
  Mean Confidence   : 79.3%

  Per-Class Report:
                        precision    recall  f1-score   support

               compost      0.834     0.821     0.827        22
         goat-sheep-pen      0.778     0.800     0.789        20
    guinea-pig-shelter      0.714     0.714     0.714        14
         liquid-organic      0.833     0.833     0.833        18
               organic      0.739     0.773     0.756        22
               pigsty      0.824     0.875     0.849        16
         poultry-house      0.864     0.864     0.864        22
              tippytap      0.900     0.900     0.900        20
                  vsla      0.867     0.813     0.839        16

  All evaluation outputs saved to: outputs/
  Saved confusion matrix → outputs/confusion_matrix.png
  ✓ Evaluation complete.
```

**Check the outputs/ folder** — you will find:
- `best_model.pth` — your trained model weights
- `training_curves.png` — graph of loss/accuracy over epochs
- `confusion_matrix.png` — which classes the model confuses
- `evaluation_report.txt` — full text report

**This completes Task 2. The written evaluation report is in `docs/task2_report.md`.**

---

## PHASE 5: RUN TASK 3 — START THE API

### Step 5.1 — Start the API server

In your terminal, type:

```
uvicorn api.main:app --reload --port 8000
```

You should see:
```
INFO:     Loading model from outputs/best_model.pth...
INFO:     Model loaded successfully. API is ready.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The API is now running. **Leave this terminal open** — closing it stops the server.

---

### Step 5.2 — View the automatic API documentation

Open your web browser and go to:
```
http://localhost:8000/docs
```

You will see a beautiful Swagger UI showing all the endpoints. You can test the API
directly from this page — click "POST /predict", then "Try it out", upload an image,
and click "Execute".

---

### Step 5.3 — Test with a real image (optional)

Open a NEW terminal (keep the first one running with the server).

**To test with curl:**
```
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/poultry-house/any_image.jpg"
```

**Expected response:**
```json
{
  "category": "poultry-house",
  "confidence": 0.8412,
  "status": "success",
  "all_probabilities": {
    "compost": 0.0021,
    "goat-sheep-pen": 0.0034,
    "guinea-pig-shelter": 0.0012,
    "liquid-organic": 0.0008,
    "organic": 0.0041,
    "pigsty": 0.0318,
    "poultry-house": 0.8412,
    "tippytap": 0.0095,
    "vsla": 0.0059
  }
}
```

**To test the health endpoint:**
```
curl http://localhost:8000/health
```

**To stop the API:** press `Ctrl+C` in the terminal where it's running.

**This completes Task 3.**

---

## PHASE 6: TASK 4 — DEPLOYMENT (OPTIONAL)

The architecture notes are already written in `docs/task4_architecture.md`.

The Dockerfile is already written. If you want to test Docker:

```
# Build the Docker image (only works after training is complete)
docker build -t rtv-classifier .

# Run the container
docker run -p 8000:8000 rtv-classifier

# Test it (same as before)
curl -X POST "http://localhost:8000/predict" -F "file=@data/tippytap/any_image.jpg"
```

**Note:** Docker must be installed on your computer. If it is not, skip this step —
the Dockerfile is already in your submission and gets credit for being there.

---

## PHASE 7: PREPARE YOUR GITHUB SUBMISSION

### Step 7.1 — Create a GitHub repository

1. Go to https://github.com and sign in (or create a free account)
2. Click the "+" button in the top-right corner
3. Click "New repository"
4. Name it: `rtv-field-classifier` (or similar)
5. Set it to Public (so RTV can view it)
6. Do NOT tick "Add a README" (you already have one)
7. Click "Create repository"

---

### Step 7.2 — Push your project to GitHub

In your terminal (inside the `rtv_project` folder), run these commands one by one:

```bash
git init
git add .
git commit -m "Initial submission: RTV Data Scientist Technical Assessment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rtv-field-classifier.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

---

### Step 7.3 — What to include / exclude

**INCLUDE in your submission:**
- All Python files (src/, api/)
- All documentation (docs/, README.md)
- requirements.txt
- Dockerfile
- outputs/training_curves.png
- outputs/confusion_matrix.png
- outputs/evaluation_report.txt

**EXCLUDE (too large for GitHub):**
- outputs/best_model.pth — the model weights file is ~20MB
  → Instead, mention in README that it is generated by running `python src/train.py`
- The data/ folder itself — images are provided by RTV, not uploaded

**To exclude large files, create a `.gitignore` file** in your project folder with:
```
outputs/best_model.pth
data/
__pycache__/
*.pyc
.DS_Store
```

Then add and commit it:
```
git add .gitignore
git commit -m "Add gitignore"
git push
```

---

## WHAT YOUR FINAL SUBMISSION LOOKS LIKE

When the RTV team opens your GitHub repository, they will see:

```
rtv-field-classifier/
├── README.md            ← Your entry point — clear setup instructions
├── requirements.txt     ← All dependencies in one file
├── Dockerfile           ← Container definition (Task 4)
├── src/
│   ├── dataset.py       ← Task 1: Data loading and analysis
│   ├── model.py         ← Task 2: Architecture definition
│   ├── train.py         ← Task 2: Training loop
│   └── evaluate.py      ← Task 2: Evaluation and metrics
├── api/
│   └── main.py          ← Task 3: FastAPI serving layer
├── docs/
│   ├── task1_analysis.md     ← Task 1 write-up
│   ├── task2_report.md       ← Task 2 evaluation report
│   └── task4_architecture.md ← Task 4 MLOps notes
└── outputs/
    ├── training_curves.png    ← From your training run
    ├── confusion_matrix.png   ← From your evaluation run
    └── evaluation_report.txt  ← From your evaluation run
```

---

## TROUBLESHOOTING COMMON PROBLEMS

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` again |
| `FileNotFoundError: data/compost not found` | Check your data/ folder structure — folder names must match exactly |
| `FileNotFoundError: outputs/best_model.pth` | Run `python src/train.py` before `python src/evaluate.py` |
| Training is very slow | Normal on CPU — let it run. Or reduce `num_epochs` to 10 in train.py |
| API won't start | Make sure `best_model.pth` exists in outputs/ (train first) |
| `Address already in use` | Another process is using port 8000. Try `--port 8001` |
| Import errors in api/main.py | Make sure you are running from the project root, not from inside api/ |

---

## QUICK REFERENCE — ALL COMMANDS IN ORDER

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Analyse the dataset (Task 1)
python src/dataset.py

# 3. Train the model (Task 2) — takes 20-45 min on CPU
python src/train.py

# 4. Evaluate the model (Task 2)
python src/evaluate.py

# 5. Start the API (Task 3)
uvicorn api.main:app --reload --port 8000

# 6. (Optional) Build Docker container (Task 4)
docker build -t rtv-classifier .
docker run -p 8000:8000 rtv-classifier
```

---

Good luck, Caleb. You have built this — every line in the code reflects real decisions
you would make in the role. Walk the RTV team through it with the same clarity and
confidence you would bring to the job. 
