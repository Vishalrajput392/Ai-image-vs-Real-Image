"""
evaluation/fix_leakage.py
----------------------------
Reads experiments/leakage_report.json (from check_leakage.py) and
removes duplicate rows from the manifest so that no image appears in
more than one split.

Priority rule: test > val > train. If the same image is found in both
train and test, the TRAIN copy is removed (test/val stay untouched as
the true holdout). If found in both val and test, the VAL copy is
removed.

Run:
    python evaluation/check_leakage.py     (if not already run)
    python evaluation/fix_leakage.py

Outputs:
    Overwrites dataset/manifests/manifest.csv with the deduplicated
    version (a backup of the original is saved alongside it as
    manifest_before_dedup.csv).
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}  # lower number = higher priority, kept


def main():
    report_path = config.EXPERIMENTS_DIR / "leakage_report.json"
    if not report_path.exists():
        print(f"Leakage report not found at {report_path}. Run check_leakage.py first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    leaks = report["cross_split_leaks"]
    sample_ids_to_remove = set()

    for group in leaks:
        items = group["items"]
        # Sort by split priority; keep the first (highest priority),
        # remove sample_ids belonging to any split that appears AFTER
        # the highest-priority split present in this group.
        best_priority = min(SPLIT_PRIORITY[item["split"]] for item in items)
        for item in items:
            if SPLIT_PRIORITY[item["split"]] > best_priority:
                sample_ids_to_remove.add(item["sample_id"])

    print(f"Cross-split leak groups: {len(leaks)}")
    print(f"Rows to remove (duplicates in lower-priority split): {len(sample_ids_to_remove)}")

    manifest_path = config.MANIFEST_PATH
    manifest = pd.read_csv(manifest_path, low_memory=False)

    backup_path = config.MANIFEST_DIR / "manifest_before_dedup.csv"
    manifest.to_csv(backup_path, index=False)
    print(f"Backup of original manifest saved -> {backup_path}")

    before_counts = manifest["split"].value_counts().to_dict()

    cleaned = manifest[~manifest["sample_id"].isin(sample_ids_to_remove)].reset_index(drop=True)

    after_counts = cleaned["split"].value_counts().to_dict()

    print("\nRows per split — before -> after:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {before_counts.get(split, 0)} -> {after_counts.get(split, 0)}")

    cleaned.to_csv(manifest_path, index=False)
    print(f"\nCleaned manifest saved -> {manifest_path}")
    print(
        "\nNext steps: retrain (training/train.py) since the train set composition changed, "
        "then re-run evaluation/evaluate.py — the test-set accuracy number after this is a "
        "more trustworthy measure of real generalization."
    )


if __name__ == "__main__":
    main()
