"""
benchmark.py
------------
Inference latency and throughput benchmarking across devices (CPU & Apple Silicon MPS).
Evaluates:
- Single-sample latency: Mean, Median (p50), 95th Percentile (p95), 99th Percentile (p99).
- Throughput across batch sizes (1, 8, 16, 32) in Frames Per Second (FPS).
- Memory parameter footprint and model disk size.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

from models import get_model, count_parameters


def measure_single_inference_latency(model, device: torch.device, num_warmup: int = 20, num_runs: int = 100) -> Dict[str, float]:
    """Measures single-image forward-pass latency stats in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

    times_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "median_p50_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2)
    }


def measure_batched_throughput(model, device: torch.device, batch_sizes: List[int] = [1, 8, 16, 32], num_batches: int = 30) -> Dict[int, float]:
    """Measures inference throughput (FPS) across different batch sizes."""
    model.eval()
    throughput = {}

    for bs in batch_sizes:
        dummy_input = torch.randn(bs, 3, 224, 224, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
                if device.type == "mps":
                    torch.mps.synchronize()

        # Timed loop
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_batches):
                _ = model(dummy_input)
                if device.type == "mps":
                    torch.mps.synchronize()
        t1 = time.perf_counter()

        total_time = t1 - t0
        total_frames = bs * num_batches
        fps = round(total_frames / total_time, 1)
        throughput[bs] = fps

    return throughput


def benchmark_model(model_name: str, checkpoint_path: str = None, reports_dir: str = "reports"):
    """Runs latency and throughput benchmarks for a model on CPU and MPS (if available)."""
    devices_to_test = [torch.device("cpu")]
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        devices_to_test.append(torch.device("mps"))

    print(f"\n=======================================================")
    print(f"Benchmarking Model: {model_name}")
    print(f"Devices to test: {[d.type for d in devices_to_test]}")
    print(f"=======================================================")

    results = {
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "benchmarks": {}
    }

    for dev in devices_to_test:
        print(f"\n>>> Running on {dev.type.upper()}...")
        model = get_model(model_name, num_classes=6, pretrained=False)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)

        total_p, trainable_p = count_parameters(model)
        single_stats = measure_single_inference_latency(model, dev)
        throughput_stats = measure_batched_throughput(model, dev)

        print(f"  Single image latency: {single_stats['mean_ms']} ms (p50: {single_stats['median_p50_ms']} ms, p95: {single_stats['p95_ms']} ms)")
        print(f"  Throughput (FPS): {throughput_stats}")

        results["parameters"] = total_p
        results["benchmarks"][dev.type] = {
            "single_latency": single_stats,
            "throughput_fps": throughput_stats
        }

    rep_path = Path(reports_dir) / f"benchmark_{model_name}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Benchmark saved to: {rep_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Model Inference Latency")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "mobilenet", "resnet"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--reports_dir", type=str, default="reports")
    args = parser.parse_args()

    benchmark_model(args.model, args.checkpoint, args.reports_dir)


if __name__ == "__main__":
    main()
