"""
inference.py
------------
Modular inference API and CLI for waste classification.
Supports:
- Single-image and batch inference from file path or PIL image.
- Softmax probability distributions across all 6 TrashNet classes.
- Confidence threshold tuning & reject option ("uncertain" flag).
- CLI for rapid local testing.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Union

import torch
import torch.nn as nn
from PIL import Image

from dataset import get_transforms
from models import get_model


class WasteClassifier:
    """Production-ready inference wrapper for TrashNet waste classification models."""

    def __init__(
        self,
        checkpoint_path: str,
        confidence_threshold: float = 0.65,
        device: str = "auto"
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.confidence_threshold = confidence_threshold

        if not self.checkpoint_path.exists():
            repo_root = Path(__file__).resolve().parent.parent
            alt_path = repo_root / checkpoint_path
            if alt_path.exists():
                self.checkpoint_path = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = ckpt.get("model_name", "baseline")
        self.classes = ckpt.get("classes", ["cardboard", "glass", "metal", "paper", "plastic", "trash"])

        self.model = get_model(self.model_name, num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = get_transforms(split="test", augment=False, img_size=224)
        self.softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, Image.Image], threshold: float = None) -> Dict[str, Any]:
        """
        Runs inference on a single image.
        
        Args:
            image_input: File path or PIL Image object.
            threshold: Optional override for confidence rejection threshold.
            
        Returns:
            Dict containing:
            - predicted_class: string
            - confidence: float (0.0 - 1.0)
            - is_confident: bool
            - status: "accepted" | "uncertain"
            - recommended_action: string
            - probabilities: dict of {class_name: probability}
        """
        thresh = threshold if threshold is not None else self.confidence_threshold

        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                repo_root = Path(__file__).resolve().parent.parent
                alt_path = repo_root / image_input
                if alt_path.exists():
                    img_path = alt_path
                else:
                    raise FileNotFoundError(f"Image not found at: {image_input}")
            img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL.Image instance.")

        tensor = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = self.softmax(logits).squeeze(0).cpu().numpy()

        best_idx = int(probs.argmax())
        best_prob = float(probs[best_idx])
        best_class = self.classes[best_idx]
        is_confident = (best_prob >= thresh)

        status = "accepted" if is_confident else "uncertain"
        if is_confident:
            action = f"Route item to the [{best_class.upper()}] recycling/disposal bin."
        else:
            action = f"Confidence ({best_prob*100:.1f}%) is below threshold ({thresh*100:.1f}%). Route to secondary optical inspection or manual sorting."

        prob_dict = {
            self.classes[i]: round(float(probs[i]), 4)
            for i in range(len(self.classes))
        }
        # Sort probabilities descending
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

        return {
            "predicted_class": best_class,
            "confidence": round(best_prob, 4),
            "confidence_percentage": round(best_prob * 100, 2),
            "threshold_applied": thresh,
            "is_confident": is_confident,
            "status": status,
            "recommended_action": action,
            "all_probabilities": sorted_probs,
            "model_name": self.model_name
        }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    default_ckpt = str(repo_root / "checkpoints" / "mobilenet_aug" / "best_model.pth")

    parser = argparse.ArgumentParser(description="Waste Classification Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=default_ckpt,
        help="Path to trained model checkpoint (default: checkpoints/mobilenet_aug/best_model.pth)"
    )
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold for acceptance")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, mps, cuda, cpu")
    args = parser.parse_args()

    classifier = WasteClassifier(
        checkpoint_path=args.checkpoint,
        confidence_threshold=args.threshold,
        device=args.device
    )

    result = classifier.predict(args.image)
    print("\n" + "="*50)
    print(f"TRASHNET WASTE CLASSIFICATION RESULT")
    print("="*50)
    print(f"Input Image        : {args.image}")
    print(f"Model Architecture : {result['model_name']}")
    print(f"Predicted Category : {result['predicted_class'].upper()}")
    print(f"Confidence Score   : {result['confidence_percentage']}%")
    print(f"Status             : {result['status'].upper()} (Threshold: {result['threshold_applied']*100:.0f}%)")
    print(f"Action             : {result['recommended_action']}")
    print("\nProbability Distribution:")
    for cls_name, prob in result["all_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:10s}: {prob*100:5.1f}%  {bar}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
