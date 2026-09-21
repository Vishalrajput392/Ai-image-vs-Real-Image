"""
evaluation/robustness_test.py
------------------------------
Answers Q12: how much does accuracy change under real-world image
modifications — original, JPEG compression, resize, crop, blur —
applied BEFORE the normal preprocessing pipeline (so this measures
actual robustness, not just what TrainTransform already saw).

Uses a random sample of the test set (SAMPLE_SIZE) rather than the
full set, for speed. Increase SAMPLE_SIZE for a tighter estimate.

Run:
    python evaluation/robustness_test.py

Outputs:
    Prints accuracy per transformation and saves to
    experiments/robustness_report.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset
from dataset.transforms import EvalTransform

SAMPLE_SIZE = 500


def apply_jpeg(img, quality=50):
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def apply_resize(img, scale=0.5):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)),
                        interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_crop(img, keep=0.8):
    h, w = img.shape[:2]
    ch, cw = int(h * keep), int(w * keep)
    top, left = (h - ch) // 2, (w - cw) // 2
    return img[top:top + ch, left:left + cw]


def apply_blur(img, ksize=5):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


TRANSFORMS = {
    "original": lambda img: img,
    "jpeg_compressed_q50": apply_jpeg,
    "resized_0.5x_roundtrip": apply_resize,
    "cropped_80pct": apply_crop,
    "blurred_5x5": apply_blur,
}


@torch.no_grad()
def predict_one(model, transform, img_rgb, device):
    rgb_t, freq_t = transform(img_rgb)
    rgb_t = rgb_t.unsqueeze(0).to(device)
    freq_t = freq_t.unsqueeze(0).to(device)
    return torch.sigmoid(model(rgb_t, freq_t).squeeze()).item()


def main():
    from models.efficientnet import build_model

    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    model = build_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    eval_tfm = EvalTransform()
    test_ds = AIRealDataset(split="test", transform=eval_tfm)

    n = min(SAMPLE_SIZE, len(test_ds.manifest))
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_idx = rng.choice(len(test_ds.manifest), n, replace=False)

    results = {name: {"correct": 0, "total": 0} for name in TRANSFORMS}

    for i in sample_idx:
        row = test_ds.manifest.iloc[i]
        img_bgr = test_ds._load_image_bgr(row)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        label = int(row["label"])

        for name, fn in TRANSFORMS.items():
            transformed = fn(img_rgb)
            prob = predict_one(model, eval_tfm, transformed, config.DEVICE)
            pred = int(prob >= config.CLASSIFICATION_THRESHOLD)
            results[name]["correct"] += int(pred == label)
            results[name]["total"] += 1

    print("\n=== Q12: Robustness under real-world transformations ===")
    report = {}
    for name, r in results.items():
        acc = r["correct"] / r["total"]
        report[name] = {"accuracy": round(acc, 4), "num_samples": r["total"]}
        print(f"{name}: accuracy={acc:.4f} (n={r['total']})")

    out_path = config.EXPERIMENTS_DIR / "robustness_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
