"""
training/train.py
------------------
Main training driver. Orchestrates:
    Phase 1 : train classification head only (backbone frozen)
    Phase 2 : fine-tune deepest backbone blocks at a lower LR
with early stopping on validation loss, best-checkpoint saving, a fixed
random seed for reproducibility, and MLflow experiment tracking
(SRS Sections 9, 11, 16).

Run:
    python training/train.py

Inputs:
    Reads manifest.csv (via dataset.dataset.AIRealDataset) and all
    hyperparameters from configs/config.py.

Outputs:
    - experiments/checkpoints/best_model.pt  (best validation-loss weights)
    - experiments/mlruns/                     (MLflow run logs)

Dependencies:
    torch, mlflow, numpy
"""

import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config
from dataset.dataset import AIRealDataset
from dataset.transforms import build_transform
from models.efficientnet import (
    build_model, freeze_backbone, unfreeze_for_finetune, get_param_groups,
)
from training.engine import train_one_epoch, validate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders():
    train_ds = AIRealDataset(split="train", transform=build_transform("train"))
    val_ds = AIRealDataset(split="val", transform=build_transform("val"))

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader


def run_phase(model, train_loader, val_loader, optimizer, criterion, scheduler,
              max_epochs, patience, phase_name, checkpoint_path, start_epoch=0):
    """Runs one training phase with early stopping on validation loss."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        global_epoch = start_epoch + epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE, epoch_num=global_epoch
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.DEVICE, epoch_num=global_epoch
        )

        print(f"[{phase_name}] Epoch {epoch}/{max_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        mlflow.log_metrics({
            f"{phase_name}_train_loss": train_loss,
            f"{phase_name}_train_acc": train_acc,
            f"{phase_name}_val_loss": val_loss,
            f"{phase_name}_val_acc": val_acc,
        }, step=global_epoch)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best {phase_name} val_loss={val_loss:.4f}, checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  -> Early stopping {phase_name} (no improvement for {patience} epochs).")
                break

    return best_val_loss, epoch


def main():
    set_seed(config.RANDOM_SEED)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{config.MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment("ai_vs_real_detection")

    print(f"Using device: {config.DEVICE}")
    train_loader, val_loader = build_dataloaders()
    model = build_model().to(config.DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_ckpt_path = config.CHECKPOINT_DIR / "best_model.pt"

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": config.BACKBONE,
            "batch_size": config.BATCH_SIZE,
            "head_lr": config.HEAD_LR,
            "finetune_lr": config.FINE_TUNE_LR,
            "weight_decay": config.WEIGHT_DECAY,
            "dropout": config.DROPOUT,
            "img_size": config.IMG_SIZE,
            "seed": config.RANDOM_SEED,
        })

        # ---------------- Phase 1: head-only training ----------------
        print("\n=== Phase 1: training classification head (backbone frozen) ===")
        freeze_backbone(model)
        optimizer = torch.optim.AdamW(
            get_param_groups(model, config.HEAD_LR, config.FINE_TUNE_LR),
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        _, phase1_epochs = run_phase(
            model, train_loader, val_loader, optimizer, criterion, scheduler,
            max_epochs=config.MAX_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
            phase_name="phase1", checkpoint_path=best_ckpt_path,
        )

        # ---------------- Phase 2: fine-tuning ----------------
        print("\n=== Phase 2: fine-tuning deeper backbone blocks ===")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=config.DEVICE))
        unfreeze_for_finetune(model, num_blocks_to_unfreeze=2)

        optimizer = torch.optim.AdamW(
            get_param_groups(model, config.HEAD_LR, config.FINE_TUNE_LR),
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        run_phase(
            model, train_loader, val_loader, optimizer, criterion, scheduler,
            max_epochs=config.MAX_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
            phase_name="phase2", checkpoint_path=best_ckpt_path,
            start_epoch=phase1_epochs,
        )

        mlflow.log_artifact(str(best_ckpt_path))

    print(f"\nTraining complete. Best model saved at: {best_ckpt_path}")


if __name__ == "__main__":
    main()
