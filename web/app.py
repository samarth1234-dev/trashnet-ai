"""
web/app.py
----------
Interactive Flask demo application & REST API for TrashNet Vision Pipeline.
Features:
- Web GUI with drag-and-drop, camera upload, and interactive confidence threshold slider.
- Curated sample library from the test set for instant 1-click evaluation.
- REST API endpoint: POST /api/predict (supports multipart/form-data & base64 JSON).
- Health check: GET /health
- Production-ready for local execution and cloud deployment (Render, Vercel, Railway, Docker).
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from inference import WasteClassifier

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable cross-origin resource sharing for external frontend clients

app.config["UPLOAD_FOLDER"] = str(ROOT_DIR / "data" / "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Select best available model checkpoint
DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "mobilenet_aug" / "best_model.pth"
if not DEFAULT_CHECKPOINT.exists():
    DEFAULT_CHECKPOINT = ROOT_DIR / "checkpoints" / "baseline_aug" / "best_model.pth"

classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        ckpt = DEFAULT_CHECKPOINT
        if not ckpt.exists():
            ckpts = list(ROOT_DIR.glob("checkpoints/**/best_model.pth"))
            if ckpts:
                ckpt = ckpts[0]
            else:
                return None
        classifier = WasteClassifier(checkpoint_path=str(ckpt), confidence_threshold=0.65)
    return classifier


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint for cloud monitoring."""
    clf = get_classifier()
    return jsonify({
        "status": "healthy",
        "service": "trashnet-vision-api",
        "model_loaded": clf is not None,
        "model_name": clf.model_name if clf else None,
        "classes": clf.classes if clf else []
    })


@app.route("/api/samples")
def get_samples():
    """Returns a curated list of test sample images for 1-click testing."""
    # First priority: static bundled samples (works in minimal cloud deploy)
    bundled_samples_file = ROOT_DIR / "web" / "static" / "samples" / "samples.json"
    if bundled_samples_file.exists():
        with open(bundled_samples_file, "r") as f:
            return jsonify(json.load(f))

    # Second priority: full dataset splits.json if present
    splits_path = ROOT_DIR / "data" / "splits.json"
    if not splits_path.exists():
        return jsonify([])

    with open(splits_path, "r") as f:
        data = json.load(f)

    test_items = data.get("splits", {}).get("test", [])
    by_class = {}
    for item in test_items:
        c = item["class_name"]
        if c not in by_class:
            by_class[c] = []
        if len(by_class[c]) < 2:
            by_class[c].append(item)

    samples = []
    for c, items in by_class.items():
        for it in items:
            samples.append({
                "class_name": c,
                "rel_path": it["rel_path"],
                "url": f"/api/image/{it['rel_path']}"
            })
    return jsonify(samples)


@app.route("/api/image/<path:img_path>")
def serve_image(img_path):
    """Serves dataset images to web UI."""
    data_dir = ROOT_DIR / "data" / "dataset-resized"
    if (data_dir / img_path).exists():
        return send_from_directory(data_dir, img_path)
    # Check fallback in static samples
    static_samples_dir = ROOT_DIR / "web" / "static" / "samples"
    for candidate in static_samples_dir.glob("*"):
        if candidate.name.endswith(Path(img_path).name):
            return send_from_directory(static_samples_dir, candidate.name)
    return jsonify({"error": "Sample image not found"}), 404


@app.route("/api/uploads/<filename>")
def serve_upload(filename):
    """Serves uploaded images."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    clf = get_classifier()
    if clf is None:
        return jsonify({"error": "No trained model checkpoint found yet."}), 503

    # Parse threshold
    threshold = 0.65
    if request.is_json:
        data = request.get_json() or {}
        threshold = float(data.get("threshold", 0.65))
    else:
        threshold = float(request.form.get("threshold", 0.65))

    img_source = None
    image_url = None

    # Case 1: Direct multipart file upload
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        fname = secure_filename(file.filename)
        save_path = Path(app.config["UPLOAD_FOLDER"]) / fname
        file.save(save_path)
        img_source = save_path
        image_url = f"/api/uploads/{fname}"

    # Case 2: Base64 JSON payload
    elif request.is_json and "image_base64" in request.json:
        b64_str = request.json["image_base64"]
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_source = img
        image_url = "data:image/jpeg;base64," + b64_str[:30] + "..."

    # Case 3: Test sample relative path
    elif "rel_path" in request.form:
        rel_path = request.form["rel_path"]
        # Check bundled static samples first
        bundled_path = ROOT_DIR / "web" / "static" / "samples" / rel_path
        dataset_path = ROOT_DIR / "data" / "dataset-resized" / rel_path
        if bundled_path.exists():
            img_source = bundled_path
            image_url = f"/static/samples/{rel_path}"
        elif dataset_path.exists():
            img_source = dataset_path
            image_url = f"/api/image/{rel_path}"
        else:
            return jsonify({"error": f"Sample image not found: {rel_path}"}), 404
    else:
        return jsonify({"error": "No image provided. Send a file or base64 JSON."}), 400

    try:
        result = clf.predict(img_source, threshold=threshold)
        result["image_url"] = image_url
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting TrashNet Web Service on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
