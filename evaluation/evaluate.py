"""
evaluation/evaluate.py
-----------------------
Runs the best checkpoint on the held-out test split and reports:
    - Standard metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
      confusion matrix, false-positive/false-negative rate (SRS 12.3)
    - Robustness breakdown by source: 'archive' (original camera/AI
      images) vs 'real_ss'/'ai_ss' (screenshot-derived), reported
      separately rather than blended (SRS 12.4)
    - Per-generator breakdown for AI images (Midjourney/Gemini/Flux/
      Stable Diffusion), since no generator was held out this phase
      (see known-limitation note in dataset/manifest_builder.py)

Run:
    python evaluation/evaluate.py

Inputs:
    experiments/checkpoints/best_model.pt (trained weights)
    dataset/manifests/manifest.csv (test split)

Outputs:
    Prints a full report to stdout and saves it as JSON to
    experiments/evaluation_results.json

Dependencies:
    torch, pandas, numpy, scikit-learn
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset
from dataset.transforms import build_transform
from models.efficientnet import build_model


@torch.no_grad()
def run_inference(model, dataloader, device):
    """Runs the model over a dataloader (shuffle=False!) and returns
    probabilities + labels in the same order as the underlying dataset."""
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        logits = model(images).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                     threshold: float = config.CLASSIFICATION_THRESHOLD) -> dict:
    preds = (probs >= threshold).astype(int)
    labels = labels.astype(int)

    if len(np.unique(labels)) < 2:
        # A subset with only one class present (e.g. a tiny generator
        # slice) — most metrics below are undefined, report what we can.
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "num_samples": len(labels),
                "note": "only one class present in this subset"}

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "confusion_matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "num_samples": int(len(labels)),
    }


def main():
    print(f"Using device: {config.DEVICE}")

    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    model = build_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    print(f"Loaded checkpoint: {ckpt_path}")

    test_ds = AIRealDataset(split="test", transform=build_transform("test"))
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    probs, labels = run_inference(model, test_loader, config.DEVICE)

    # Attach predictions back onto the (ordered) test manifest for breakdowns.
    results_df = test_ds.manifest.copy()
    results_df["prob_ai"] = probs
    results_df["true_label"] = labels

    report = {}

    print("\n=== Overall test metrics ===")
    overall = compute_metrics(probs, labels)
    print(json.dumps(overall, indent=2))
    report["overall"] = overall

    # ---- Robustness breakdown: original (archive) vs screenshot-derived ----
    print("\n=== Robustness breakdown by source ===")
    report["by_source"] = {}
    for source_name, group in results_df.groupby("source"):
        m = compute_metrics(group["prob_ai"].values, group["true_label"].values)
        print(f"\n-- source = {source_name} --")
        print(json.dumps(m, indent=2))
        report["by_source"][source_name] = m

    # ---- Per-generator breakdown (AI images only) ----
    print("\n=== Per-generator breakdown (AI images) ===")
    report["by_generator"] = {}
    ai_rows = results_df[results_df["source"] == "ai_ss"]
    for gen_name, group in ai_rows.groupby("generator"):
        m = compute_metrics(group["prob_ai"].values, group["true_label"].values)
        print(f"\n-- generator = {gen_name} --")
        print(json.dumps(m, indent=2))
        report["by_generator"][gen_name] = m

    out_path = config.EXPERIMENTS_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()
