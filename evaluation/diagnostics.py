"""
evaluation/diagnostics.py
--------------------------
Answers the dataset-composition and per-generator questions that plain
evaluate.py doesn't cover:
    - Total images, real vs AI counts (Q1)
    - Exact train/val/test split (Q2)
    - Generator-wise counts per split (Q3, Q7)
    - Current threshold (Q10)
    - Generator-wise accuracy/recall/false-negative-rate (Q14)
    - Flux misclassified samples with their probabilities (Q8, Q9)

Run:
    python evaluation/diagnostics.py

Outputs:
    Prints everything to stdout and saves it to
    experiments/diagnostics_report.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset
from dataset.transforms import build_transform
from models.efficientnet import build_model


def dataset_composition() -> dict:
    manifest = pd.read_csv(config.MANIFEST_PATH, low_memory=False)
    report = {
        "total_images": int(len(manifest)),
        "by_label": {str(k): int(v) for k, v in manifest["label"].value_counts().items()},
        "by_split": {str(k): int(v) for k, v in manifest["split"].value_counts().items()},
        "by_split_label": manifest.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(),
        "by_source": {str(k): int(v) for k, v in manifest["source"].value_counts().items()},
        "by_source_split": manifest.groupby(["source", "split"]).size().unstack(fill_value=0).to_dict(),
    }
    gen_rows = manifest[manifest["source"] == "ai_ss"]
    report["generator_by_split"] = (
        gen_rows.groupby(["generator", "split"]).size().unstack(fill_value=0).to_dict()
    )
    return report


@torch.no_grad()
def run_inference_with_identity(model, ds, device):
    """Runs inference batch-by-batch but keeps track of which manifest
    row each prediction belongs to (needed to look up misclassified
    Flux samples specifically, not just aggregate accuracy)."""
    model.eval()
    records = []
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False)

    idx = 0
    for (rgb, freq), labels in loader:
        rgb, freq = rgb.to(device), freq.to(device)
        probs = torch.sigmoid(model(rgb, freq).squeeze(-1)).cpu().numpy()
        for p, l in zip(probs, labels.numpy()):
            row = ds.manifest.iloc[idx]
            records.append({
                "sample_id": int(row["sample_id"]),
                "filepath": str(row.get("filepath", "")),
                "generator": str(row.get("generator", "")),
                "source": str(row["source"]),
                "true_label": int(l),
                "prob_ai": float(p),
            })
            idx += 1
    return records


def main():
    print(f"Using device: {config.DEVICE}")

    comp = dataset_composition()
    print("\n=== Q1/Q2/Q3/Q7: Dataset composition ===")
    print(json.dumps(comp, indent=2, default=str))

    print(f"\n=== Q10: Current classification threshold ===")
    print(config.CLASSIFICATION_THRESHOLD)

    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    model = build_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))

    test_ds = AIRealDataset(split="test", transform=build_transform("test"))
    records = run_inference_with_identity(model, test_ds, config.DEVICE)
    results_df = pd.DataFrame(records)

    print("\n=== Q14: Generator-wise accuracy / recall / FNR (test split) ===")
    # Note: each generator's ai_ss subset is 100% label=1 (AI), so
    # accuracy == recall here by definition, and FNR = 1 - recall.
    generator_report = {}
    ai_rows = results_df[results_df["generator"].notna() & (results_df["generator"] != "")]
    for gen, group in ai_rows.groupby("generator"):
        preds = (group["prob_ai"] >= config.CLASSIFICATION_THRESHOLD).astype(int)
        n = len(group)
        correct = int((preds == group["true_label"]).sum())
        recall = correct / n
        generator_report[gen] = {
            "num_samples": int(n),
            "accuracy": round(correct / n, 4),
            "recall": round(recall, 4),
            "false_negative_rate": round(1 - recall, 4),
        }
        print(f"{gen}: {generator_report[gen]}")

    print("\n=== Q8/Q9: Flux misclassified samples (with probabilities) ===")
    flux_rows = ai_rows[ai_rows["generator"] == "flux"]
    flux_wrong = flux_rows[flux_rows["prob_ai"] < config.CLASSIFICATION_THRESHOLD]
    flux_wrong_list = flux_wrong[["sample_id", "filepath", "prob_ai"]].to_dict("records")
    print(json.dumps(flux_wrong_list, indent=2))
    print(
        "\n(prob_ai close to the threshold = model was 'unsure'; "
        "prob_ai near 0 = model was confidently wrong)"
    )

    out = {
        "dataset_composition": comp,
        "threshold": config.CLASSIFICATION_THRESHOLD,
        "generator_report": generator_report,
        "flux_misclassified": flux_wrong_list,
    }
    out_path = config.EXPERIMENTS_DIR / "diagnostics_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
