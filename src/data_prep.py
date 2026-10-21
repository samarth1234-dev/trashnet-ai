"""
data_prep.py
-------------
TrashNet dataset preparation, inspection, duplicate detection, and reproducible splitting.

Requirements addressed:
- Document dataset source, version, and number of images per class.
- Inspect dataset and analyze class imbalance.
- Create reproducible train/val/test splits (70/15/15, fixed seed=42).
- Ensure strict zero-leakage: verify no duplicate image hashes across splits.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

DATASET_METADATA = {
    "name": "TrashNet",
    "source_url": "https://github.com/garythung/trashnet",
    "hf_mirror": "https://huggingface.co/datasets/garythung/trashnet",
    "authors": "Gary Thung, Mindy Yang (Stanford CS 229)",
    "paper": "Classification of Trash for Recyclability Status (2016)",
    "version": "v1.0 (dataset-resized)",
    "dimensions_standard": [512, 384],
    "channels": 3,
    "color_space": "RGB"
}


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file for content verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_and_inspect_dataset(data_dir: Path):
    """
    Scans the dataset directory, collects metadata, dimensions, modes,
    and identifies duplicate images via SHA-256 hash.
    """
    print(f"[INFO] Scanning dataset at: {data_dir.resolve()}")
    records = []
    hash_to_files = defaultdict(list)
    dimensions_hist = defaultdict(int)
    modes_hist = defaultdict(int)

    for class_name in CLASSES:
        class_folder = data_dir / class_name
        if not class_folder.exists() or not class_folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_folder}")

        files = sorted([f for f in class_folder.iterdir() if f.is_file() and not f.name.startswith(".")])
        for f in files:
            sha = compute_sha256(f)
            rel_path = f"{class_name}/{f.name}"
            
            with Image.open(f) as img:
                w, h = img.size
                mode = img.mode
                dimensions_hist[f"{w}x{h}"] += 1
                modes_hist[mode] += 1

            records.append({
                "rel_path": rel_path,
                "abs_path": str(f.resolve()),
                "filename": f.name,
                "class_name": class_name,
                "class_idx": CLASS_TO_IDX[class_name],
                "sha256": sha,
                "width": w,
                "height": h,
                "mode": mode,
                "size_bytes": f.stat().st_size
            })
            hash_to_files[sha].append(rel_path)

    df = pd.DataFrame(records)
    print(f"[INFO] Total images scanned: {len(df)}")

    # Detect duplicates
    duplicates = {sha: files for sha, files in hash_to_files.items() if len(files) > 1}
    print(f"[INFO] Number of unique hashes: {len(hash_to_files)}")
    print(f"[INFO] Exact duplicate hash groups detected: {len(duplicates)}")
    for sha, files in duplicates.items():
        print(f"       Duplicate hash {sha[:10]}... -> {files}")

    return df, duplicates, dict(dimensions_hist), dict(modes_hist)


def analyze_class_imbalance(df: pd.DataFrame):
    """Computes distribution statistics, class imbalance ratios, and weights."""
    total_samples = len(df)
    class_counts = df["class_name"].value_counts().to_dict()
    
    # Calculate imbalance relative to maximum and minimum classes
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = round(max_count / min_count, 2)
    
    stats = {}
    for cls_name in CLASSES:
        count = class_counts.get(cls_name, 0)
        pct = round((count / total_samples) * 100, 2)
        # Standard inverse frequency weight
        weight = round(total_samples / (len(CLASSES) * count), 4) if count > 0 else 0.0
        stats[cls_name] = {
            "count": count,
            "percentage": pct,
            "inverse_freq_weight": weight
        }

    return {
        "total_images": total_samples,
        "imbalance_ratio_max_to_min": imbalance_ratio,
        "majority_class": max(class_counts, key=class_counts.get),
        "minority_class": min(class_counts, key=class_counts.get),
        "per_class": stats
    }


def create_reproducible_splits(df: pd.DataFrame, duplicates: dict, drop_duplicates: bool = True, seed: int = 42):
    """
    Creates stratified 70% train / 15% validation / 15% test splits with fixed random seed.
    Guarantees no duplicate images across splits.
    """
    cleaned_df = df.copy()
    dropped_files = []

    if drop_duplicates and duplicates:
        print("[INFO] Resolving cross-class duplicate conflicts...")
        for sha, files in duplicates.items():
            # In TrashNet, duplicates are cross-class anomalies:
            # e.g., ('glass/glass389.jpg', 'plastic/plastic332.jpg').
            # To prevent contradictory supervision and cross-split contamination,
            # drop all occurrences of ambiguous cross-class duplicate pairs.
            for rel_path in files:
                dropped_files.append(rel_path)
        
        cleaned_df = cleaned_df[~cleaned_df["rel_path"].isin(dropped_files)].reset_index(drop=True)
        print(f"[INFO] Removed {len(dropped_files)} ambiguous duplicate images. Remaining clean images: {len(cleaned_df)}")

    # Stratified Split 1: 70% Train, 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        cleaned_df,
        test_size=0.30,
        random_state=seed,
        stratify=cleaned_df["class_name"]
    )

    # Stratified Split 2: Temp 30% -> 15% Val, 15% Test (50/50 of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["class_name"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Verification: Strict Hash Disjointness
    train_hashes = set(train_df["sha256"])
    val_hashes = set(val_df["sha256"])
    test_hashes = set(test_df["sha256"])

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    assert len(leak_train_val) == 0, f"Data leakage between train and val: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Data leakage between train and test: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Data leakage between val and test: {leak_val_test}"
    print("[SUCCESS] Zero-leakage verified: Disjoint hash sets between train, val, and test splits!")

    total_clean = len(cleaned_df)
    split_summary = {
        "seed": seed,
        "clean_total": total_clean,
        "dropped_duplicates_count": len(dropped_files),
        "dropped_files": dropped_files,
        "train": {
            "count": len(train_df),
            "percentage": round(len(train_df) / total_clean * 100, 2),
            "per_class": train_df["class_name"].value_counts().to_dict()
        },
        "val": {
            "count": len(val_df),
            "percentage": round(len(val_df) / total_clean * 100, 2),
            "per_class": val_df["class_name"].value_counts().to_dict()
        },
        "test": {
            "count": len(test_df),
            "percentage": round(len(test_df) / total_clean * 100, 2),
            "per_class": test_df["class_name"].value_counts().to_dict()
        }
    }

    return train_df, val_df, test_df, split_summary


def export_splits_and_manifests(train_df, val_df, test_df, split_summary, imbalance_stats, data_dir: Path, output_dir: Path):
    """Exports manifests (JSON & CSV) for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_csv_dir = output_dir / "splits"
    splits_csv_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(splits_csv_dir / "train.csv", index=False)
    val_df.to_csv(splits_csv_dir / "val.csv", index=False)
    test_df.to_csv(splits_csv_dir / "test.csv", index=False)

    manifest = {
        "dataset_metadata": DATASET_METADATA,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "imbalance_analysis": imbalance_stats,
        "split_summary": split_summary,
        "splits": {
            "train": train_df[["rel_path", "class_name", "class_idx", "sha256"]].to_dict(orient="records"),
            "val": val_df[["rel_path", "class_name", "class_idx", "sha256"]].to_dict(orient="records"),
            "test": test_df[["rel_path", "class_name", "class_idx", "sha256"]].to_dict(orient="records")
        }
    }

    splits_json_path = output_dir / "splits.json"
    with open(splits_json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    summary_json_path = output_dir / "dataset_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump({
            "dataset_metadata": DATASET_METADATA,
            "classes": CLASSES,
            "imbalance_analysis": imbalance_stats,
            "split_summary": split_summary
        }, f, indent=2)

    print(f"[INFO] Exported splits manifest to: {splits_json_path}")
    print(f"[INFO] Exported split CSVs to: {splits_csv_dir}")


def main():
    parser = argparse.ArgumentParser(description="TrashNet Data Preparation and Reproducible Splitting")
    parser.add_argument("--data_dir", type=str, default="data/dataset-resized", help="Path to unzipped TrashNet dataset")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for manifests and splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 70/15/15 reproducible split")
    parser.add_argument("--keep_duplicates", action="store_true", help="Keep duplicate images instead of filtering")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    df, duplicates, dims, modes = scan_and_inspect_dataset(data_dir)
    imbalance_stats = analyze_class_imbalance(df)
    
    print("\n--- DATASET INSPECTION SUMMARY ---")
    print(f"Total Images: {imbalance_stats['total_images']}")
    print(f"Image Resolution: {dims}")
    print(f"Color Modes: {modes}")
    print(f"Imbalance Ratio: {imbalance_stats['imbalance_ratio_max_to_min']}x (Majority: {imbalance_stats['majority_class']}, Minority: {imbalance_stats['minority_class']})")
    print("\nClass Breakdown:")
    for c, stat in imbalance_stats["per_class"].items():
        print(f"  {c:10s}: {stat['count']:4d} images ({stat['percentage']:5.2f}%) | Weight: {stat['inverse_freq_weight']:.3f}")

    train_df, val_df, test_df, split_summary = create_reproducible_splits(
        df=df,
        duplicates=duplicates,
        drop_duplicates=not args.keep_duplicates,
        seed=args.seed
    )

    print("\n--- SPLIT SIZES ---")
    print(f"Train : {len(train_df)} ({split_summary['train']['percentage']}%)")
    print(f"Val   : {len(val_df)} ({split_summary['val']['percentage']}%)")
    print(f"Test  : {len(test_df)} ({split_summary['test']['percentage']}%)")

    export_splits_and_manifests(train_df, val_df, test_df, split_summary, imbalance_stats, data_dir, output_dir)
    print("\n[SUCCESS] Dataset preparation complete.")


if __name__ == "__main__":
    main()
