"""
run_experiments.py
------------------
Master pipeline orchestrator that executes:
1. Baseline CNN without Data Augmentation (Ablation)
2. Baseline CNN with Data Augmentation
3. Transfer Learning with MobileNetV3-Small
4. Transfer Learning with ResNet-18
5. Comprehensive Test Set Evaluation for each model:
   - Accuracy, Precision, Recall, Macro F1, Weighted F1
   - Confusion Matrix generation
   - Confidence threshold tuning
   - Failure case analysis
6. Latency & Throughput Benchmarking (CPU & MPS)
7. Generates master comparison report & JSON
"""

import sys
import json
import time
from pathlib import Path

# Add src to python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd

from train import run_training
from evaluate import evaluate_checkpoint
from benchmark import benchmark_model

MODELS_CONFIG = [
    {
        "model": "baseline",
        "exp_name": "baseline_noaug",
        "epochs": 8,
        "batch_size": 32,
        "lr": 1e-3,
        "augment": False,
        "use_class_weights": True
    },
    {
        "model": "baseline",
        "exp_name": "baseline_aug",
        "epochs": 8,
        "batch_size": 32,
        "lr": 1e-3,
        "augment": True,
        "use_class_weights": True
    },
    {
        "model": "mobilenet",
        "exp_name": "mobilenet_aug",
        "epochs": 8,
        "batch_size": 32,
        "lr": 3e-4,
        "augment": True,
        "use_class_weights": True
    },
    {
        "model": "resnet",
        "exp_name": "resnet_aug",
        "epochs": 8,
        "batch_size": 32,
        "lr": 2e-4,
        "augment": True,
        "use_class_weights": True
    }
]


def run_all():
    print("=" * 70)
    print("TRASHNET COMPUTER VISION PIPELINE: FULL EXPERIMENTAL SUITE")
    print("=" * 70)

    results = []

    for cfg in MODELS_CONFIG:
        exp_name = cfg["exp_name"]
        model_name = cfg["model"]
        print(f"\n\n{'#' * 60}")
        print(f"STEP: Running Experiment [{exp_name}]")
        print(f"{'#' * 60}")

        # 1. Train
        ckpt_path, train_summary = run_training(
            model_name=model_name,
            exp_name=exp_name,
            manifest_path="data/splits.json",
            data_root="data/dataset-resized",
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            augment=cfg["augment"],
            use_class_weights=cfg["use_class_weights"],
            device_str="auto",
            output_dir="checkpoints"
        )

        # 2. Evaluate on Test Set
        eval_summary = evaluate_checkpoint(
            checkpoint_path=str(ckpt_path),
            manifest_path="data/splits.json",
            data_root="data/dataset-resized",
            exp_name=exp_name,
            device_str="auto",
            reports_dir="reports"
        )

        # 3. Benchmark Latency
        bench_summary = benchmark_model(
            model_name=model_name,
            checkpoint_path=str(ckpt_path),
            reports_dir="reports"
        )

        # Gather summary row
        mps_lat = bench_summary["benchmarks"].get("mps", {}).get("single_latency", {}).get("mean_ms", "N/A")
        cpu_lat = bench_summary["benchmarks"].get("cpu", {}).get("single_latency", {}).get("mean_ms", "N/A")
        mps_fps = bench_summary["benchmarks"].get("mps", {}).get("throughput_fps", {}).get("32", "N/A")

        row = {
            "Experiment": exp_name,
            "Architecture": model_name.upper(),
            "Data Augmentation": cfg["augment"],
            "Class Weighted": cfg["use_class_weights"],
            "Parameters": f"{train_summary['total_params']:,}",
            "Test Accuracy (%)": eval_summary["overall_accuracy"],
            "Macro F1 (%)": eval_summary["macro_f1"],
            "Weighted F1 (%)": eval_summary["weighted_f1"],
            "Single Latency MPS (ms)": mps_lat,
            "Single Latency CPU (ms)": cpu_lat,
            "Batch-32 FPS (MPS)": mps_fps,
            "Total Errors": eval_summary["total_failures"]
        }
        results.append(row)

    # Master Table Export
    df_results = pd.DataFrame(results)
    print("\n\n" + "=" * 70)
    print("MASTER EXPERIMENTAL COMPARISON TABLE")
    print("=" * 70)
    print(df_results.to_string(index=False))

    master_json = Path("reports") / "MASTER_COMPARISON.json"
    master_csv = Path("reports") / "MASTER_COMPARISON.csv"
    with open(master_json, "w") as f:
        json.dump(results, f, indent=2)
    df_results.to_csv(master_csv, index=False)
    print(f"\n[INFO] Master comparison exported to: {master_json} and {master_csv}")


if __name__ == "__main__":
    run_all()
