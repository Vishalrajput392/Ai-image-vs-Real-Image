# AI-Generated vs Real Image Detection

Binary image classifier (Real vs AI-Generated) built with transfer-learning
EfficientNet-B0, robust to screenshots, compression, and resizing.
Based on SRS v1.1.

## Folder structure

```
project/
├── requirements.txt
├── configs/config.py            # all paths, hyperparameters
├── dataset/
│   ├── manifest_builder.py      # unifies Archive + RealSS parquet + AISS into manifest.csv
│   ├── dataset.py                # PyTorch Dataset (OpenCV-based decoding)
│   ├── transforms.py             # preprocessing + augmentation (OpenCV + torch)
│   └── manifests/manifest.csv    # generated
├── models/efficientnet.py        # EfficientNet-B0, 2-phase freeze/unfreeze
├── training/
│   ├── engine.py                  # train/val epoch loops
│   └── train.py                   # Phase 1 -> Phase 2 driver, early stopping, MLflow
├── evaluation/evaluate.py         # test metrics + robustness/generator breakdown
├── backend/main.py                # FastAPI /predict endpoint
├── frontend/app.py                # Streamlit UI
├── experiments/                   # checkpoints + MLflow runs (generated)
└── data_raw/                      # your extracted dataset (not in repo)
    ├── archives/{train,test}/{real,fake}/
    ├── real_ss/{train,test,validation}.parquet
    └── ai_ss/{midjourney,gemini,flux,stable_diffusion}/
```

## Setup

```bash
pip install -r requirements.txt
# CUDA-matched PyTorch build for RTX 3060:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Extract your Google Drive dataset into `data_raw/` matching the layout above.

## Run order

```bash
python dataset/manifest_builder.py   # 1. builds manifest.csv (run once, or after any data change)
python dataset/dataset.py            # 2. sanity check
python dataset/transforms.py         # 2b. sanity check
python models/efficientnet.py        # 3. sanity check (downloads pretrained weights)
python training/train.py             # 4. Phase 1 + Phase 2 training, GPU required
python evaluation/evaluate.py        # 5. test-set metrics + robustness breakdown
```

## Run the app

```bash
# terminal 1
uvicorn backend.main:app --reload --port 8000
# terminal 2
streamlit run frontend/app.py
```

Open the Streamlit URL it prints, upload an image, click Analyze.

## Known deviations from the baseline SRS (record these in the final report)

- **Image processing**: SRS Section 10.2 lists OpenCV as the only imaging
  library; this project uses **OpenCV + PyTorch**, Pillow intentionally
  avoided throughout (dataset loading, preprocessing, and backend inference).
- **Cross-generator holdout (SRS 6.2)**: all 4 AI generators (Midjourney,
  Gemini, Flux, Stable Diffusion) are included in training for this phase.
  No generator was held out for cross-generator generalization testing —
  flagged as a known limitation, listed as a future improvement.
- **Archive/AISS splits**: Archive shipped with only train/test (no
  validation) — validation was carved out of train
  (`ARCHIVE_VAL_FRACTION` in config.py). AISS shipped with no split at
  all — split 70/15/15 per generator in `manifest_builder.py`.
- **Training hardware**: local Lenovo LOQ (RTX 3060) was used instead of
  Google Colab, chosen for faster data I/O on the large image-file
  dataset and no session-time limits.

## Reproducibility

- Random seed fixed at `config.RANDOM_SEED` (42), applied to Python,
  NumPy, and PyTorch (see `training/train.py: set_seed`).
- Every run logs params/metrics to MLflow under `experiments/mlruns/`.
- Best checkpoint (lowest validation loss) saved to
  `experiments/checkpoints/best_model.pt`.
