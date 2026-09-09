"""
config.py
---------
Central configuration for the AI-Generated vs Real Image Detection project.

Purpose:
    Single source of truth for dataset paths, class labels, split ratios,
    preprocessing constants, and training hyperparameters (SRS Sections
    6, 7, 9, 11). Every other module imports from here instead of
    hardcoding values, so changing a setting only requires editing this
    one file.

Inputs:
    None (this module defines constants; it does not read external files).

Outputs:
    Module-level constants/paths used by dataset.py, transforms.py,
    train.py, evaluate.py, etc.

Dependencies:
    pathlib, torch (only to detect CUDA availability)
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT / DATA PATHS
# ---------------------------------------------------------------------------
# PROJECT_ROOT = the "project/" folder itself (parent of this configs/ dir)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where you extract/download the Google Drive dataset locally (see the
# data_raw/ layout shared in chat). Change this if your local path differs.
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"

ARCHIVE_TRAIN_REAL_DIR = DATA_RAW_DIR / "archives" / "train" / "real"
ARCHIVE_TRAIN_FAKE_DIR = DATA_RAW_DIR / "archives" / "train" / "fake"
ARCHIVE_TEST_REAL_DIR  = DATA_RAW_DIR / "archives" / "test" / "real"
ARCHIVE_TEST_FAKE_DIR  = DATA_RAW_DIR / "archives" / "test" / "fake"

REAL_SS_DIR       = DATA_RAW_DIR / "real_ss"
REAL_SS_TRAIN_PQ  = REAL_SS_DIR / "train.parquet"
REAL_SS_TEST_PQ   = REAL_SS_DIR / "test.parquet"
REAL_SS_VAL_PQ    = REAL_SS_DIR / "validation.parquet"

AI_SS_DIR = DATA_RAW_DIR / "ai_ss"
AI_SS_GENERATORS = ["midjourney", "gemini", "flux", "stable_diffusion"]

# Where the unified manifest (built by dataset/manifest_builder.py) is saved.
MANIFEST_DIR = PROJECT_ROOT / "dataset" / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "manifest.csv"

# Checkpoints, logs, MLflow tracking
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINT_DIR = EXPERIMENTS_DIR / "checkpoints"
MLFLOW_TRACKING_DIR = EXPERIMENTS_DIR / "mlruns"

for _d in [MANIFEST_DIR, CHECKPOINT_DIR, MLFLOW_TRACKING_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. LABELS
# ---------------------------------------------------------------------------
LABEL_REAL = 0
LABEL_AI = 1
CLASS_NAMES = {LABEL_REAL: "Real", LABEL_AI: "AI-Generated"}

# ---------------------------------------------------------------------------
# 3. DATASET SPLIT STRATEGY (SRS Sec 6.3, adapted to actual data — see chat)
# ---------------------------------------------------------------------------
# Archive has train/test only -> we carve validation out of its train split.
ARCHIVE_VAL_FRACTION = 0.15  # taken from archive "train" to make a val set

# AISS (AI screenshots) has no split at all -> we split it ourselves.
# NOTE (deviation from SRS Sec 6.2): all 4 generators are included in
# training for this phase; no generator is held out for cross-generator
# testing. Record this as a known limitation in the final report.
AISS_TRAIN_FRACTION = 0.70
AISS_VAL_FRACTION   = 0.15
AISS_TEST_FRACTION  = 0.15

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 4. PREPROCESSING (SRS Sec 7)
# ---------------------------------------------------------------------------
IMG_SIZE = 224                 # final H x W fed to the model
MIN_RESOLUTION = 224
MAX_RESOLUTION = 4096
MAX_FILE_SIZE_MB = 10
ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}

# ImageNet normalization stats (matches EfficientNet-B0 pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 5. MODEL (SRS Sec 9)
# ---------------------------------------------------------------------------
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 1  # single logit, BCEWithLogitsLoss (sigmoid at inference)
DROPOUT = 0.3

# ---------------------------------------------------------------------------
# 6. TRAINING (SRS Sec 11)
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_WORKERS = 4

# Phase 1: train classification head only (backbone frozen)
HEAD_LR = 3e-4
# Phase 2: fine-tune deeper blocks at a lower LR
FINE_TUNE_LR = 1e-5

WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 6

# ---------------------------------------------------------------------------
# 7. EVALUATION (SRS Sec 12)
# ---------------------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.5
