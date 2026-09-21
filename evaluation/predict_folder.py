"""
evaluation/predict_folder.py
------------------------------
Ad-hoc diagnostic tool: point it at a folder of images (any images you
downloaded/tested manually — not part of the dataset) and it prints the
model's prediction + exact probability for each one. Useful for seeing
whether "wrong" predictions are borderline (near the threshold) or
confidently wrong (a real generalization gap).

Run:
    python evaluation/predict_folder.py "C:\\path\\to\\your\\test\\images"
"""

import sys
from pathlib import Path

import cv2
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.transforms import EvalTransform, ensure_rgb_3channel
from models.efficientnet import build_model

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"
    model = build_model(pretrained=False).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    transform = EvalTransform()
    image_paths = [p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_EXT]

    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"Found {len(image_paths)} images. Threshold = {config.CLASSIFICATION_THRESHOLD}\n")
    print(f"{'Filename':<45} {'Prediction':<15} {'AI Probability'}")
    print("-" * 80)

    for path in sorted(image_paths):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"{path.name:<45} COULD NOT READ")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = ensure_rgb_3channel(img_rgb)

        rgb_t, freq_t = transform(img_rgb)
        rgb_t = rgb_t.unsqueeze(0).to(config.DEVICE)
        freq_t = freq_t.unsqueeze(0).to(config.DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(model(rgb_t, freq_t).squeeze()).item()

        label = "AI-Generated" if prob >= config.CLASSIFICATION_THRESHOLD else "Real"
        print(f"{path.name:<45} {label:<15} {prob*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python evaluation/predict_folder.py "path\\to\\folder"')
    else:
        main(sys.argv[1])
