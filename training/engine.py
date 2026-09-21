"""
training/engine.py
-------------------
Core train/validation loop functions, one epoch at a time. Kept separate
from training/train.py so the epoch logic can be unit-tested or reused
(e.g. by evaluation/evaluate.py) without pulling in the full training
orchestration (phases, scheduler, early stopping, checkpointing).

Inputs:
    model      : nn.Module, single-logit binary classifier
    dataloader : torch.utils.data.DataLoader yielding (images, labels)
    optimizer  : torch.optim.Optimizer (train_one_epoch only)
    criterion  : loss function, expected nn.BCEWithLogitsLoss()
    device     : torch.device

Outputs:
    avg_loss (float), accuracy (float in [0,1])

Dependencies:
    torch, tqdm
"""

import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs import config


def _batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.detach())
    preds = (probs >= config.CLASSIFICATION_THRESHOLD).float()
    correct = (preds.squeeze(-1) == labels).float().sum().item()
    return correct


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch_num=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    desc = f"Train (epoch {epoch_num})" if epoch_num is not None else "Train"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for (rgb, freq), labels in pbar:
        rgb = rgb.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(rgb, freq).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = rgb.size(0)
        running_loss += loss.item() * batch_size
        running_correct += _batch_accuracy(logits.unsqueeze(-1), labels)
        total_samples += batch_size

        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / total_samples
    accuracy = running_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch_num=None):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    desc = f"Val (epoch {epoch_num})" if epoch_num is not None else "Val"
    pbar = tqdm(dataloader, desc=desc, leave=False)

    for (rgb, freq), labels in pbar:
        rgb = rgb.to(device, non_blocking=True)
        freq = freq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(rgb, freq).squeeze(-1)
        loss = criterion(logits, labels)

        batch_size = rgb.size(0)
        running_loss += loss.item() * batch_size
        running_correct += _batch_accuracy(logits.unsqueeze(-1), labels)
        total_samples += batch_size

        pbar.set_postfix(loss=loss.item())

    avg_loss = running_loss / total_samples
    accuracy = running_correct / total_samples
    return avg_loss, accuracy