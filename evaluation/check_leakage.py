"""
evaluation/check_leakage.py
------------------------------
Detects (near-)duplicate images across the train/val/test splits using
a perceptual hash (average hash) computed with OpenCV only — no
Pillow, no external hashing library.

Why this matters (SRS Section 6.3 / 17.2):
    If the same image (or a near-identical copy) appears in both the
    training set and the test set, the model can effectively
    "memorize" it, inflating test accuracy without real generalization.
    This has NOT been checked before in this project.

Method:
    1. Compute a 16x16 average-hash (256-bit fingerprint) for every
       image in the manifest, across all three splits.
    2. Group images by identical hash.
    3. Any group whose images span MORE THAN ONE split is flagged as
       a likely leak.
    4. Groups within a single split (duplicates that don't cross
       splits) are reported separately — not leakage, but still a
       data-quality note worth knowing.

Limitation (stated honestly): average-hash catches exact copies and
near-identical re-compressions/resizes. It will NOT catch a cropped,
heavily-filtered, or otherwise substantially altered duplicate of the
same source photo — a full solution would need a more expensive
similarity search, out of scope for this quick check.

Run:
    python evaluation/check_leakage.py

Outputs:
    Prints a summary and saves full details to
    experiments/leakage_report.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset

HASH_SIZE = 16  # 16x16 = 256-bit fingerprint


def average_hash(img_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    avg = small.mean()
    bits = (small > avg).astype(np.uint8).flatten()
    # Pack bits into a hex string (compact, hashable, easy to store in JSON)
    bit_str = "".join(str(b) for b in bits)
    return hex(int(bit_str, 2))[2:]


def hash_all_images():
    """Returns a dict: hash -> list of {sample_id, split, source, generator, label, ref}"""
    hash_groups = defaultdict(list)

    for split in ["train", "val", "test"]:
        print(f"\nHashing '{split}' split...")
        ds = AIRealDataset(split=split, transform=None)

        for i in tqdm(range(len(ds.manifest)), desc=split):
            row = ds.manifest.iloc[i]
            try:
                img_bgr = ds._load_image_bgr(row)
            except Exception as e:
                continue  # unreadable image, skip for this check

            h = average_hash(img_bgr)
            ref = row["filepath"] if row["source"] != "real_ss" else f"{row['parquet_path']}#{row['row_index']}"

            hash_groups[h].append({
                "sample_id": int(row["sample_id"]),
                "split": split,
                "source": row["source"],
                "generator": row.get("generator", ""),
                "label": int(row["label"]),
                "ref": str(ref),
            })

    return hash_groups


def main():
    hash_groups = hash_all_images()

    cross_split_leaks = []
    same_split_duplicates = []

    for h, items in hash_groups.items():
        if len(items) < 2:
            continue
        splits_involved = {item["split"] for item in items}
        if len(splits_involved) > 1:
            cross_split_leaks.append({"hash": h, "items": items})
        else:
            same_split_duplicates.append({"hash": h, "items": items})

    print("\n" + "=" * 60)
    print(f"Total unique images hashed: {sum(len(v) for v in hash_groups.values())}")
    print(f"Cross-split leakage groups found: {len(cross_split_leaks)}")
    print(f"Same-split duplicate groups found: {len(same_split_duplicates)}")
    print("=" * 60)

    if cross_split_leaks:
        print("\n⚠️  LEAKAGE DETECTED — example groups (up to 10 shown):")
        for group in cross_split_leaks[:10]:
            splits = [item["split"] for item in group["items"]]
            print(f"  hash={group['hash'][:12]}...  splits={splits}  "
                  f"sample_ids={[it['sample_id'] for it in group['items']]}")
    else:
        print("\n✅ No cross-split leakage detected (within this hash method's sensitivity).")

    report = {
        "total_images_hashed": sum(len(v) for v in hash_groups.values()),
        "cross_split_leak_group_count": len(cross_split_leaks),
        "same_split_duplicate_group_count": len(same_split_duplicates),
        "cross_split_leaks": cross_split_leaks,
        "same_split_duplicates_sample": same_split_duplicates[:50],  # cap for file size
    }

    out_path = config.EXPERIMENTS_DIR / "leakage_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()
