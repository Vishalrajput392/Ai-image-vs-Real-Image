# AI-Generated vs Real Image Detection

Binary image classifier (Real vs AI-Generated) built with transfer-learning
EfficientNet-B0 (RGB branch) fused with an FFT-magnitude-spectrum branch,
robust to screenshots, compression, and resizing. 

## Current status (as of latest local run)

**Overall test metrics** (21,060 samples, threshold = 0.35):

| Metric | Value |
|---|---|
| Accuracy | 93.85% |
| Precision | 91.14% |
| Recall | 97.49% |
| F1 | 94.21% |
| ROC-AUC | 0.9886 |
| PR-AUC | 0.9893 |
| False Positive Rate | 10.01% |
| False Negative Rate | 2.51% |

**By source:** archive 93.75% (n=20,000) - ai_ss 95.35% (n=818) - real_ss 96.69% (n=242)

**By generator:** flux 95.16% (n=620) - stable_diffusion 96.99% (n=166) - midjourney 93.75% (n=16) - gemini 87.5% (n=16)

flux improved from 43.75% to 95.16% after adding a `WeightedRandomSampler`
(to counter the archive source's ~120k images drowning out the ~424-image
ai_ss class) and collecting more flux-generated images. gemini and
midjourney still sit at only 16 test samples each -- **too small to trust**;
a couple of misclassifications swing that number by 6-12 points. Treat
those two as unvalidated until more data is collected.



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
├── models/efficientnet.py        # EfficientNet-B0 + FFT branch, 2-phase freeze/unfreeze
├── training/
│   ├── engine.py                  # train/val epoch loops
│   └── train.py                   # Phase 1 -> Phase 2 driver, early stopping, MLflow
├── evaluation/
│   ├── evaluate.py                # test metrics + robustness/generator breakdown
│   ├── check_leakage.py           # perceptual-hash cross-split duplicate detector
│   ├── fix_leakage.py             # removes leaked rows from manifest (priority: test > val > train)
│   ├── tune_threshold.py          # sweeps classification threshold on validation set
│   ├── diagnostics.py
│   ├── robustness_test.py
│   └── predict_folder.py
├── backend/main.py                # FastAPI /predict endpoint
├── frontend/app.py                # Streamlit UI
├── experiments/                   # checkpoints, MLflow runs, leakage/threshold/eval reports (generated)
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

## Run order (updated -- includes leakage fix + threshold revalidation)

```bash
python dataset/manifest_builder.py   # 1. builds manifest.csv (run once, or after any data change)
python dataset/dataset.py            # 2. sanity check
python dataset/transforms.py         # 2b. sanity check
python models/efficientnet.py        # 3. sanity check (downloads pretrained weights)

python evaluation/check_leakage.py   # 4. detect cross-split duplicates (already run -- see leakage_report.json)
python evaluation/fix_leakage.py     # 5. remove leaked rows from manifest.csv -- NOT YET APPLIED

python training/train.py             # 6. Phase 1 + Phase 2 training, GPU required -- retrain after step 5
python evaluation/evaluate.py        # 7. test-set metrics + robustness breakdown
python evaluation/tune_threshold.py  # 8. revalidate threshold after retrain, then update config.py
python evaluation/diagnostics.py     # 9.
python evaluation/robustness_test.py # 10.
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
  No generator was held out for cross-generator generalization testing --
  flagged as a known limitation, listed as a future improvement (see
  Known Issues above).
- **Archive/AISS splits**: Archive shipped with only train/test (no
  validation) -- validation was carved out of train
  (`ARCHIVE_VAL_FRACTION` in config.py). AISS shipped with no split at
  all -- split 70/15/15 per generator in `manifest_builder.py`.
- **Training hardware**: local Lenovo LOQ (RTX 3060) was used instead of
  Google Colab, chosen for faster data I/O on the large image-file
  dataset and no session-time limits.

## Reproducibility

- Random seed fixed at `config.RANDOM_SEED` (42), applied to Python,
  NumPy, and PyTorch (see `training/train.py: set_seed`).
- Every run logs params/metrics to MLflow under `experiments/mlruns/`.
- Best checkpoint (lowest validation loss) saved to
  `experiments/checkpoints/best_model.pt`.
- **Caveat**: the checkpoint and metrics currently in `experiments/` were
  produced *before* the leakage fix in `fix_leakage.py` was applied -- see
  Known Issues above before treating them as final.
