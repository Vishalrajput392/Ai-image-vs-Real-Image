"""
evaluation/tune_threshold.py
------------------------------
Finds a properly justified classification threshold using the
VALIDATION set (never the test set — SRS Section 12.1/12.2: test stays
untouched until the final threshold is chosen).

For a range of candidate thresholds, reports accuracy, precision,
recall, F1, and false-positive/false-negative rate — so you can pick
the threshold that fits your priorities (e.g. minimize false positives
on real photos vs. minimize missed AI images) with actual evidence,
instead of guessing a number.

Run:
    python evaluation/tune_threshold.py

Outputs:
    Prints a table to stdout and saves full results to
    experiments/threshold_tuning.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset
from dataset.transforms import build_transform
from models.efficientnet import build_model

CANDIDATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    for (rgb, freq), labels in dataloader:
        rgb, freq = rgb.to(device), freq.to(device)
        probs = torch.sigmoid(model(rgb, freq).squeeze(-1)).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / len(labels)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")  # real -> wrongly called AI
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")  # AI -> wrongly called real

    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fpr), 4),  # real photos called AI
        "false_negative_rate": round(float(fnr), 4),  # AI images called real
    }


def main():
    print(f"Using device: {config.DEVICE}")
    print("Evaluating on the VALIDATION set (test set stays untouched).\n")

    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    model = build_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

    val_ds = AIRealDataset(split="val", transform=build_transform("val"))
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    probs, labels = run_inference(model, val_loader, config.DEVICE)

    results = [metrics_at_threshold(probs, labels, t) for t in CANDIDATE_THRESHOLDS]

    print(f"{'Threshold':<10}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}"
          f"{'FPR (real->AI)':<16}{'FNR (AI->real)'}")
    print("-" * 80)
    for r in results:
        print(f"{r['threshold']:<10}{r['accuracy']:<10}{r['precision']:<11}"
              f"{r['recall']:<9}{r['f1']:<8}{r['false_positive_rate']:<16}"
              f"{r['false_negative_rate']}")

    best_f1 = max(results, key=lambda r: r["f1"])
    print(f"\nBest F1 (balanced): threshold={best_f1['threshold']} -> F1={best_f1['f1']}")

    print(
        "\nGuidance:\n"
        "  - Lower threshold  -> fewer missed AI images, but more real photos flagged as AI\n"
        "  - Higher threshold -> fewer real photos flagged as AI, but more AI images missed\n"
        "  - Pick based on which mistake matters more for your use case, using the table above.\n"
        "  - Whatever you choose, update CLASSIFICATION_THRESHOLD in configs/config.py and\n"
        "    record the choice + reasoning in your final report (SRS Section 12.2)."
    )

    out_path = config.EXPERIMENTS_DIR / "threshold_tuning.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
