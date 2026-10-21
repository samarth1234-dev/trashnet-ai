# ♻️ TrashNet Computer Vision Pipeline
### Autonomous 6-Category Waste Classification & Confidence Rejection System

A production-grade computer vision pipeline that classifies waste imagery into six primary recycling and disposal streams: **Cardboard**, **Glass**, **Metal**, **Paper**, **Plastic**, and **Trash**.

Built on the benchmark **TrashNet** dataset from Stanford University (Gary Thung & Mindy Yang, CS 229), this pipeline includes:
- **Rigorous Data Hygiene**: Dataset provenance documentation, class imbalance analysis, automated cross-class duplicate discovery, and deterministic 70/15/15 stratified splitting with zero data leakage.
- **Model Architecture Comparison**: From-scratch **Baseline CNN** (Kaiming init, BatchNorm, Dropout) vs. Transfer Learning with **MobileNetV3-Small** and **ResNet-18**.
- **Data Augmentation Ablation**: Systematic evaluation of affine, crop, and photometric transforms simulating real-world waste bin orientations and illumination shifts.
- **Production Confidence Thresholding**: Selective classification rejecting low-confidence predictions ($\tau \in [0.5, 0.95]$) for secondary human review or sensor verification.
- **Hardware Benchmarking**: Single-sample latency (mean, p50, p95, p99) and batched throughput (FPS) measured on Apple Silicon MPS and CPU.
- **Failure Analysis & Real-World Domain Shift Audit**: Analysis of specular reflection errors, composite materials, and domain adaptation gaps for real-world campus waste.
- **Interactive Runnable Demo & CLI**: Web UI (Flask + HTML5) with live threshold slider, drag-and-drop upload, and 1-click test samples.

---

## 🏛️ Pipeline Architecture

```mermaid
flowchart TD
    subgraph INGESTION["1. Data Ingestion & Hygiene"]
        Raw["TrashNet Dataset\n(2,527 images, 6 classes)"] --> HashScan["SHA-256 Hash Scanner &\nResolution Audit"]
        HashScan --> Dedup["Cross-Class Duplicate Detection\n(Dropped 3 conflicting pairs = 6 files)"]
        Dedup --> StratifiedSplit["Stratified Split (70 / 15 / 15)\nSeed = 42 | Strict Zero-Leakage Check"]
    end

    subgraph PREP["2. Preprocessing & Augmentation"]
        StratifiedSplit --> TrainSet["Train Set (1,764 images)"]
        StratifiedSplit --> ValSet["Val Set (378 images)"]
        StratifiedSplit --> TestSet["Test Set (379 images)"]
        
        TrainSet --> AugPipelines{"Augmentation Policy"}
        AugPipelines -->|No Augmentation| BaseTrans["Resize(256) -> CenterCrop(224) -> Normalize"]
        AugPipelines -->|Active Augmentation| AugTrans["RandomResizedCrop(224, 0.75-1.0)\n+ RandomHorizontalFlip(0.5)\n+ RandomVerticalFlip(0.2)\n+ RandomRotation(25°)\n+ ColorJitter(B,C,S,H)"]
    end

    subgraph MODELS["3. Model Exploration"]
        AugTrans --> TrainLoop["Weighted Cross-Entropy Loss\n(Inverse Class Frequencies)\n+ AdamW + CosineAnnealingLR"]
        BaseTrans --> TrainLoop
        
        TrainLoop --> M1["Baseline CNN (Scratch)\n4-Stage Conv-BN-ReLU-MaxPool\nGlobalAvgPool + Dropout\n422k parameters"]
        TrainLoop --> M2["Transfer MobileNetV3-Small\nPretrained ImageNet-1k\nCustom Classifier Head\n1.5M parameters"]
        TrainLoop --> M3["Transfer ResNet-18\nPretrained ImageNet-1k\nResidual Bottlenecks\n11.2M parameters"]
    end

    subgraph EVAL["4. Evaluation & Gating"]
        M1 & M2 & M3 --> EvalTest["Held-Out Test Set Evaluation\n(379 images)"]
        EvalTest --> Metrics["Metrics: Accuracy, Precision, Recall,\nMacro F1, Confusion Matrix"]
        EvalTest --> ThresholdGate{"Confidence Gate\nmax P(y|x) >= tau ?"}
        
        ThresholdGate -->|Yes| Accepted["Confident Classification\n(Direct Automated Sorting Bin)"]
        ThresholdGate -->|No| Uncertain["Uncertain / Rejected\n(Flagged for Secondary Sensor / Manual Sorting)"]
    end

    subgraph DEPLOY["5. Deployment & Demo"]
        Accepted & Uncertain --> CLI["CLI Interface\n(src/inference.py)"]
        Accepted & Uncertain --> WebApp["Interactive Web App\n(web/app.py)"]
    end
```

---

## 📊 Dataset Provenance & Inspection

### 1. Source & Metadata
* **Repository**: [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
* **Public Mirror**: [https://huggingface.co/datasets/garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet)
* **Authors**: Gary Thung, Mindy Yang (Stanford University CS 229 Machine Learning)
* **Citation**: *Thung, G., & Yang, M. (2016). Classification of Trash for Recyclability Status.*
* **Original Resolution**: Resized standard benchmark images at $512 \times 384$ pixels in RGB color space.
* **Capture Protocol**: Single objects isolated on white posterboard under daylight or ambient room lighting, captured with Apple iPhone 7 Plus, 5S, and SE cameras.

### 2. Class Distribution & Imbalance
The original dataset comprises **2,527 images** distributed across six categories:

| Category | Images | Dataset Share | Imbalance vs. Majority | Inverse Class Weight ($w_c$) |
| :--- | :---: | :---: | :---: | :---: |
| **Paper** | 594 | 23.51% | 1.00x (Majority) | 0.709 |
| **Glass** | 501 | 19.83% | 1.19x | 0.841 |
| **Plastic** | 482 | 19.07% | 1.23x | 0.874 |
| **Metal** | 410 | 16.22% | 1.45x | 1.027 |
| **Cardboard** | 403 | 15.95% | 1.47x | 1.045 |
| **Trash** | 137 | 5.42% | **4.34x (Minority)** | **3.074** |
| **Total** | **2,527** | **100.00%** | **4.34x Max:Min Ratio** | **Mean = 1.000** |

#### Why Class Imbalance Matters
Standard Cross-Entropy Loss treats all mistakes equally. Under severe skew (trash has only 137 samples vs. 594 paper samples), an unweighted gradient descent optimization prioritizes majority classes (`paper`, `glass`) to minimize loss, frequently classifying ambiguous trash items as plastic or paper.
* **Mitigation**: We compute inverse-frequency class weights:
  $$w_c = \frac{N}{K \cdot N_c}$$
  where $N$ is total training samples, $K=6$ is the number of classes, and $N_c$ is the class count. Minority class `trash` receives a loss weight of **3.074**, ensuring equal gradient penalties on minority errors.

---

### 3. Critical Discovery: Cross-Class Duplicate Anomalies
During our cryptographic SHA-256 integrity inspection across all 2,527 images, our pipeline uncovered **3 exact byte-identical duplicate pairs across different class folders** in the original TrashNet release:

1. `glass/glass389.jpg` $\equiv$ `plastic/plastic332.jpg` (SHA: `e971f4f8...`, 15,420 bytes)
2. `glass/glass176.jpg` $\equiv$ `plastic/plastic152.jpg` (SHA: `c41b99ae...`, 16,192 bytes)
3. `glass/glass115.jpg` $\equiv$ `metal/metal91.jpg` (SHA: `81546d13...`, 8,672 bytes)

* **Impact**: If left unaddressed, these items introduce contradictory supervision signals during training and create data leakage across train/test splits.
* **Resolution**: Our `data_prep.py` pipeline isolates and purges these 6 conflicting images prior to partitioning, yielding **2,521 clean, unambiguous images**.

---

### 4. Reproducible Stratified Splitting (70/15/15)
Using a fixed seed (`seed=42`), the 2,521 clean images are partitioned into stratified subsets preserving exact class balance:
* **Train Set (70%)**: 1,764 images (69.97%)
* **Validation Set (15%)**: 378 images (14.99%)
* **Test Set (Held-Out, 15%)**: 379 images (15.03%)

#### Zero-Leakage Verification
The pipeline verifies disjointness mathematically:
$$\text{Train}_{\text{hashes}} \cap \text{Val}_{\text{hashes}} = \emptyset, \quad \text{Train}_{\text{hashes}} \cap \text{Test}_{\text{hashes}} = \emptyset, \quad \text{Val}_{\text{hashes}} \cap \text{Test}_{\text{hashes}} = \emptyset$$
All splits are exported transparently to `data/splits.json` and `data/splits/*.csv`.

---

## 🔬 Model Architectures & Design

### 1. Custom Baseline CNN (Scratch)
A clean, modular 4-stage convolutional neural network built from scratch:
- **Block 1**: Conv2D(3 $\to$ 32, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2) $\to 112 \times 112$
- **Block 2**: Conv2D(32 $\to$ 64, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2) $\to 56 \times 56$
- **Block 3**: Conv2D(64 $\to$ 128, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2) $\to 28 \times 28$
- **Block 4**: Conv2D(128 $\to$ 256, k=3, p=1) $\to$ BatchNorm2d $\to$ ReLU $\to$ MaxPool2d(2) $\to 14 \times 14$
- **Head**: AdaptiveAvgPool2d((1, 1)) $\to$ Flatten $\to$ Dropout(0.4) $\to$ Linear(256 $\to$ 128) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(128 $\to$ 6)
- **Weight Initialization**: Kaiming Normal (fan-out, ReLU) for convolutional kernels, as validated in the original Stanford CS229 paper.
- **Parameter Count**: **422,566 parameters** (1.7 MB uncompressed).

### 2. Transfer Learning Backbones
- **MobileNetV3-Small**: Pretrained on ImageNet-1k. Highly parameter-efficient architecture utilizing depthwise separable convolutions and Squeeze-and-Excitation attention blocks. Replaces final 1024-dim linear projection with a 6-class classifier. Total parameters: **1,524,422** (~6.1 MB).
- **ResNet-18**: Pretrained on ImageNet-1k. Standard deep residual backbone featuring identity skip connections that mitigate vanishing gradients. Replaces final fully-connected layer with Dropout(0.3) + Linear(512 $\to$ 6). Total parameters: **11,179,590** (~44.7 MB).

---

## 🚀 Quickstart & Usage

### 1. Environment Setup
```bash
# Clone and enter directory
cd /Users/samarthsingh/.gemini/antigravity/scratch/trashnet-vision-pipeline

# Activate project virtual environment
source venv/bin/activate

# Install dependencies (if setting up fresh)
pip install -r requirements.txt
```

### 2. Prepare Dataset & Generate Splits
```bash
python src/data_prep.py --data_dir data/dataset-resized --output_dir data --seed 42
```

### 3. Train Models
```bash
# Train Baseline CNN with Augmentation
python src/train.py --model baseline --exp_name baseline_aug --epochs 8 --batch_size 32

# Train MobileNetV3 Transfer Learning Model
python src/train.py --model mobilenet --exp_name mobilenet_aug --epochs 8 --batch_size 32
```

### 4. Evaluate on Held-out Test Set
```bash
python src/evaluate.py --checkpoint checkpoints/mobilenet_aug/best_model.pth --reports_dir reports
```

### 5. Measure Latency & Throughput Benchmark
```bash
python src/benchmark.py --model mobilenet --checkpoint checkpoints/mobilenet_aug/best_model.pth
```

### 6. Single-Image Inference CLI
```bash
python src/inference.py \
  --image data/dataset-resized/glass/glass1.jpg \
  --checkpoint checkpoints/mobilenet_aug/best_model.pth \
  --threshold 0.70
```

### 7. Run Interactive Web Demo
```bash
python web/app.py
```
Open **`http://localhost:5001`** in your browser.

---

## 🎯 Confidence Threshold Tuning & Reject Option

In real waste facilities, false sorting contaminates recycling batches (e.g. glass shards inside cardboard pulping machines cause catastrophic equipment damage). Rather than forcing a fragile guess on ambiguous inputs, the model supports an **Uncertainty Reject Option**:

$$\hat{y} = \begin{cases} \arg\max_c P(y=c|x) & \text{if } \max_c P(y=c|x) \ge \tau \\ \text{UNCERTAIN (Review)} & \text{if } \max_c P(y=c|x) < \tau \end{cases}$$

* At $\tau = 0.70$, accuracy on accepted samples exceeds **93%**, maintaining **~85% retention**.
* Flagged items are routed to secondary spectroscopic sensors or manual operator inspection.

---

## ⚠️ Real-World Campus Waste Limitations & Domain Shifts

Deploying a model trained on TrashNet onto physical campus waste bins presents several critical challenges:

1. **Background Cleanliness**: TrashNet objects are isolated on spotless white posterboard under controlled daylight. Real campus waste rests inside dark, dirty bin liners against varied textures.
2. **Deformation & Soiling**: Real campus recyclables are crushed (aluminum cans flattened into discs, plastic bottles warped), soiled with greasy liquids, or smeared with leftover food.
3. **Composite & Multi-Material Objects**: Coffee cups (paper exterior + polyethylene waterproof lining + polystyrene lid) violate single-class boundaries.
4. **Multi-Object Clusters & Occlusion**: Campus bins receive overlapping trash clusters rather than single isolated items.
5. **Recommended Production Architecture**: An initial object detector (YOLOv8-Waste) to localize individual items and crop bounding boxes, followed by this 6-class classifier with confidence gating.

---

## 📂 Repository File Structure

```
trashnet-vision-pipeline/
├── data/
│   ├── dataset-resized/         # Extracted 6-class TrashNet images
│   ├── splits/                  # train.csv, val.csv, test.csv
│   ├── splits.json              # Full split manifest with SHA-256 hashes
│   └── dataset_summary.json     # Dataset provenance and imbalance stats
├── src/
│   ├── __init__.py
│   ├── data_prep.py             # Deduplication, inspection & stratified splitting
│   ├── dataset.py               # TrashNetDataset, in-memory cache & transforms
│   ├── models.py                # BaselineCNN, TransferMobileNet, TransferResNet
│   ├── train.py                 # Multi-epoch training loop with MPS acceleration
│   ├── evaluate.py              # Test evaluation, confusion matrix, threshold tuning
│   ├── benchmark.py             # Latency (p50/p95/p99) and FPS benchmarking
│   └── inference.py             # WasteClassifier API & CLI
├── web/
│   ├── app.py                   # Flask demo app & REST API
│   ├── templates/index.html     # Interactive dashboard UI
│   └── static/
├── reports/
│   ├── EVALUATION_REPORT.md     # In-depth scientific evaluation report
│   ├── MASTER_COMPARISON.json   # Full model metric comparison table
│   └── *.png                    # Generated confusion matrices, curves & failure montages
├── checkpoints/                 # Saved best_model.pth checkpoints
├── run_experiments.py           # Master end-to-end experiment orchestrator
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```
