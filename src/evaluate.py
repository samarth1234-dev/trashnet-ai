"""
evaluate.py
-----------
Comprehensive evaluation suite for TrashNet models on the held-out test split.
Features:
- Accuracy, Per-class Precision, Recall, F1-Score, and Support.
- High-resolution Seaborn Confusion Matrix (raw counts + normalized percentages).
- Confidence threshold tuning: Analyzes Accuracy vs Coverage trade-off.
- Failure case extraction, overconfident error ranking, and visual montage plotting.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from dataset import build_dataloaders
from models import get_model


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def load_model_for_eval(checkpoint_path: Path, device: torch.device):
    """Loads a model from a saved checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", "baseline")
    classes = ckpt.get("classes")
    
    model = get_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes, ckpt


@torch.no_grad()
def run_test_inference(model, test_loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Runs forward pass on entire test set.
    Returns:
    - true_labels: 1D numpy array
    - pred_labels: 1D numpy array
    - probabilities: 2D numpy array [N, C]
    - rel_paths: list of relative filepaths
    """
    all_targets = []
    all_preds = []
    all_probs = []
    all_paths = []

    softmax = nn.Softmax(dim=1)

    for images, targets, paths in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = softmax(outputs).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_targets.extend(targets.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)
        all_paths.extend(paths)

    return (
        np.array(all_targets),
        np.array(all_preds),
        np.vstack(all_probs),
        all_paths
    )


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: Path, title: str):
    """Generates dual-annotated confusion matrix (raw count and row-normalized percentage)."""
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Text annotation formatting: "Count\n(Pct%)"
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j] * 100
            annot[i, j] = f"{c}\n({p:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Normalized Recall Proportion'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title, fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=250)
    plt.close()


def tune_confidence_thresholds(true_labels: np.ndarray, pred_labels: np.ndarray, probs: np.ndarray, save_plot_path: Path):
    """
    Evaluates model performance across a spectrum of confidence thresholds.
    When confidence < threshold, sample is routed to 'Uncertain / Human Review'.
    """
    max_confidences = np.max(probs, axis=1)
    thresholds = [0.0, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    records = []

    for tau in thresholds:
        accepted_mask = max_confidences >= tau
        num_accepted = int(np.sum(accepted_mask))
        num_rejected = len(true_labels) - num_accepted
        coverage = num_accepted / len(true_labels)

        if num_accepted > 0:
            acc = accuracy_score(true_labels[accepted_mask], pred_labels[accepted_mask])
            p, r, f1, _ = precision_recall_fscore_support(
                true_labels[accepted_mask], pred_labels[accepted_mask], average="macro", zero_division=0
            )
        else:
            acc, p, r, f1 = 0.0, 0.0, 0.0, 0.0

        records.append({
            "threshold": tau,
            "accepted_samples": num_accepted,
            "rejected_samples": num_rejected,
            "coverage_pct": round(coverage * 100, 2),
            "accuracy_pct": round(acc * 100, 2),
            "macro_f1_pct": round(f1 * 100, 2)
        })

    # Plot Accuracy vs Coverage curve
    df_thresh = pd.DataFrame(records)
    plt.figure(figsize=(9, 5))
    plt.plot(df_thresh["threshold"], df_thresh["accuracy_pct"], "b-o", label="Accepted Accuracy (%)", linewidth=2)
    plt.plot(df_thresh["threshold"], df_thresh["coverage_pct"], "r--s", label="Coverage / Retention (%)", linewidth=2)
    plt.plot(df_thresh["threshold"], df_thresh["macro_f1_pct"], "g-^", label="Macro F1 (%)", linewidth=1.5)
    plt.axvline(0.70, color="gray", linestyle=":", label="Recommended Threshold (tau=0.70)")
    plt.xlabel("Confidence Rejection Threshold (tau)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.title("Confidence Threshold Tuning: Accuracy vs Coverage Trade-off", fontsize=13, fontweight="bold")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=250)
    plt.close()

    return records


def analyze_failure_cases(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    probs: np.ndarray,
    rel_paths: List[str],
    classes: List[str],
    data_root: Path,
    save_json_path: Path,
    save_montage_path: Path,
    top_k: int = 12
):
    """
    Extracts all misclassified samples, identifies the most confident mistakes,
    and creates a visual image montage with true vs predicted annotations.
    """
    mistakes = []
    max_confs = np.max(probs, axis=1)

    for i in range(len(true_labels)):
        if true_labels[i] != pred_labels[i]:
            mistakes.append({
                "rel_path": rel_paths[i],
                "true_label": classes[true_labels[i]],
                "true_idx": int(true_labels[i]),
                "pred_label": classes[pred_labels[i]],
                "pred_idx": int(pred_labels[i]),
                "confidence": float(max_confs[i]),
                "probabilities": {classes[c]: round(float(probs[i, c]), 4) for c in range(len(classes))}
            })

    # Sort descending by confidence (most overconfident mistakes first)
    mistakes.sort(key=lambda x: -x["confidence"])

    with open(save_json_path, "w") as f:
        json.dump({
            "total_test_samples": len(true_labels),
            "total_mistakes": len(mistakes),
            "error_rate_pct": round(len(mistakes) / len(true_labels) * 100, 2),
            "top_failures": mistakes
        }, f, indent=2)

    # Plot Visual Montage
    if mistakes:
        n_show = min(top_k, len(mistakes))
        cols = 4
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for idx in range(len(axes)):
            if idx < n_show:
                m = mistakes[idx]
                img_path = data_root / m["rel_path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    axes[idx].imshow(img)
                except Exception as e:
                    axes[idx].text(0.5, 0.5, "Image Error", ha="center")
                title = f"True: {m['true_label']}\nPred: {m['pred_label']} ({m['confidence']*100:.1f}%)"
                axes[idx].set_title(title, fontsize=10, color="crimson", fontweight="bold")
                axes[idx].axis("off")
            else:
                axes[idx].axis("off")

        plt.suptitle("Top Failure Cases (High-Confidence Misclassifications)", fontsize=15, y=0.99, fontweight="bold")
        plt.tight_layout()
        save_montage_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_montage_path, dpi=250)
        plt.close()

    return mistakes


def evaluate_checkpoint(
    checkpoint_path: str,
    manifest_path: str = "data/splits.json",
    data_root: str = "data/dataset-resized",
    exp_name: str = None,
    device_str: str = "auto",
    reports_dir: str = "reports"
):
    device = get_device(device_str)
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, classes, ckpt = load_model_for_eval(ckpt_path, device)
    model_name = ckpt.get("model_name", "unknown")
    experiment_name = exp_name or ckpt_path.parent.name

    print(f"\n=======================================================")
    print(f"Evaluating Model: {model_name} ({experiment_name}) on Test Split")
    print(f"Checkpoint: {ckpt_path}")
    print(f"=======================================================")

    loaders, _, _ = build_dataloaders(
        manifest_path=manifest_path,
        data_root=data_root,
        batch_size=32,
        augment_train=False,
        img_size=224,
        num_workers=0
    )

    true_labels, pred_labels, probs, rel_paths = run_test_inference(model, loaders["test"], device)

    # 1. Overall Metrics
    acc = accuracy_score(true_labels, pred_labels)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)
    weight_p, weight_r, weight_f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)

    # 2. Per-class report
    clf_report = classification_report(true_labels, pred_labels, target_names=classes, output_dict=True, zero_division=0)
    clf_text = classification_report(true_labels, pred_labels, target_names=classes, digits=4, zero_division=0)
    print("\n--- TEST SET CLASSIFICATION REPORT ---")
    print(clf_text)

    # 3. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    rep_dir = Path(reports_dir)
    cm_path = rep_dir / f"confusion_matrix_{experiment_name}.png"
    plot_confusion_matrix(cm, classes, cm_path, title=f"Confusion Matrix: {experiment_name} ({model_name})")
    print(f"[INFO] Confusion matrix plot saved to: {cm_path}")

    # 4. Confidence Threshold Tuning
    tuning_plot_path = rep_dir / f"confidence_tuning_{experiment_name}.png"
    tuning_records = tune_confidence_thresholds(true_labels, pred_labels, probs, tuning_plot_path)
    print(f"[INFO] Confidence tuning plot saved to: {tuning_plot_path}")

    # 5. Failure Case Analysis
    failures_json_path = rep_dir / f"failure_cases_{experiment_name}.json"
    failures_plot_path = rep_dir / f"failure_cases_{experiment_name}.png"
    mistakes = analyze_failure_cases(
        true_labels, pred_labels, probs, rel_paths, classes,
        Path(data_root), failures_json_path, failures_plot_path
    )
    print(f"[INFO] Failure cases JSON saved to: {failures_json_path}")
    print(f"[INFO] Failure cases montage saved to: {failures_plot_path}")

    eval_summary = {
        "experiment_name": experiment_name,
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "test_samples": len(true_labels),
        "overall_accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_p * 100, 2),
        "macro_recall": round(macro_r * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weight_f1 * 100, 2),
        "per_class": {
            cls: {
                "precision": round(clf_report[cls]["precision"] * 100, 2),
                "recall": round(clf_report[cls]["recall"] * 100, 2),
                "f1_score": round(clf_report[cls]["f1-score"] * 100, 2),
                "support": clf_report[cls]["support"]
            }
            for cls in classes
        },
        "confusion_matrix": cm.tolist(),
        "confidence_threshold_tuning": tuning_records,
        "total_failures": len(mistakes)
    }

    metrics_save_path = rep_dir / f"metrics_{experiment_name}.json"
    with open(metrics_save_path, "w") as f:
        json.dump(eval_summary, f, indent=2)
    print(f"[INFO] Full evaluation metrics saved to: {metrics_save_path}")

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrashNet Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--manifest", type=str, default="data/splits.json", help="Path to splits manifest")
    parser.add_argument("--data_root", type=str, default="data/dataset-resized", help="Path to dataset directory")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name override")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, mps, cuda, cpu")
    parser.add_argument("--reports_dir", type=str, default="reports", help="Reports destination folder")
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        data_root=args.data_root,
        exp_name=args.exp_name,
        device_str=args.device,
        reports_dir=args.reports_dir
    )


if __name__ == "__main__":
    main()
