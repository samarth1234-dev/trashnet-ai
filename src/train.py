"""
train.py
--------
Training pipeline for TrashNet classifiers.
Features:
- Baseline CNN vs Transfer Learning (MobileNetV3 / ResNet-18)
- Data augmentation switch for ablation studies
- Class-weighted CrossEntropyLoss to address dataset imbalance
- Automatic Apple Silicon MPS / CUDA / CPU device selection
- Validation macro-F1 tracking and best-checkpoint saving
- Training history logging and curve plotting
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

from dataset import build_dataloaders
from models import get_model, count_parameters


def get_device(requested_device: str = "auto") -> torch.device:
    """Selects best available compute device (MPS on Mac, CUDA on Linux/Windows, or CPU)."""
    if requested_device == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Executes one training epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Evaluates validation loss, accuracy, and macro F1."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def plot_curves(history: dict, save_path: Path, title: str):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc (%)")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], "r-s", label="Val Acc (%)")
    plt.plot(epochs, [f * 100 for f in history["val_f1"]], "g-^", label="Val Macro F1 (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Percentage (%)")
    plt.title(f"{title} - Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_training(
    model_name: str = "baseline",
    exp_name: str = "baseline_experiment",
    manifest_path: str = "data/splits.json",
    data_root: str = "data/dataset-resized",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment: bool = True,
    use_class_weights: bool = True,
    device_str: str = "auto",
    output_dir: str = "checkpoints"
):
    device = get_device(device_str)
    print(f"\n=======================================================")
    print(f"Starting Training Experiment: {exp_name}")
    print(f"Model: {model_name} | Device: {device} | Augment: {augment} | Epochs: {epochs}")
    print(f"=======================================================")

    loaders, classes, class_weights = build_dataloaders(
        manifest_path=manifest_path,
        data_root=data_root,
        batch_size=batch_size,
        augment_train=augment,
        img_size=224,
        num_workers=0
    )

    print(f"[INFO] Train samples: {len(loaders['train'].dataset)} | Val samples: {len(loaders['val'].dataset)}")
    print(f"[INFO] Classes ({len(classes)}): {classes}")
    if use_class_weights:
        print(f"[INFO] Applying inverse class frequency weights: {[round(w.item(), 3) for w in class_weights]}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    model = get_model(model_name, num_classes=len(classes), pretrained=True).to(device)
    total_p, trainable_p = count_parameters(model)
    print(f"[INFO] Parameters: {total_p:,} total | {trainable_p:,} trainable")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)

    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "epoch_times": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    out_dir = Path(output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_model.pth"

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc, train_f1 = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        epoch_dur = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["epoch_times"].append(epoch_dur)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "augment": augment,
                "config": {
                    "lr": lr, "batch_size": batch_size, "weight_decay": weight_decay
                }
            }, best_ckpt_path)

        star = "★ BEST" if is_best else ""
        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_dur:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% F1: {val_f1*100:5.2f}% {star}")

    total_training_sec = time.time() - start_total_time
    print(f"\n[FINISH] Training completed in {total_training_sec/60:.2f} mins.")
    print(f"[BEST] Epoch {best_epoch} with Val Macro-F1: {best_val_f1*100:.2f}%. Saved to {best_ckpt_path}")

    # Save final history and metadata
    history_file = out_dir / "history.json"
    summary = {
        "exp_name": exp_name,
        "model_name": model_name,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_val_acc": history["val_acc"][best_epoch - 1],
        "total_params": total_p,
        "total_train_time_sec": total_training_sec,
        "history": history
    }
    with open(history_file, "w") as f:
        json.dump(summary, f, indent=2)

    curve_path = Path("reports") / f"{exp_name}_curves.png"
    plot_curves(history, curve_path, title=f"{exp_name} ({model_name})")
    print(f"[INFO] Learning curves plotted to: {curve_path}")

    return best_ckpt_path, summary


def main():
    parser = argparse.ArgumentParser(description="TrashNet Model Training")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenet", "resnet"], help="Model type")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--manifest", type=str, default="data/splits.json", help="Path to splits.json")
    parser.add_argument("--data_root", type=str, default="data/dataset-resized", help="Path to images directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (defaults: 1e-3 for baseline, 3e-4 for transfer)")
    parser.add_argument("--no_augment", action="store_true", help="Disable training data augmentation")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weighting in loss")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, mps, cuda, cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoints root directory")
    args = parser.parse_args()

    default_lr = 1e-3 if args.model == "baseline" else 3e-4
    learning_rate = args.lr if args.lr is not None else default_lr

    aug_str = "noaug" if args.no_augment else "aug"
    exp_name = args.exp_name or f"{args.model}_{aug_str}"

    run_training(
        model_name=args.model,
        exp_name=exp_name,
        manifest_path=args.manifest,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=learning_rate,
        augment=not args.no_augment,
        use_class_weights=not args.no_class_weights,
        device_str=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
