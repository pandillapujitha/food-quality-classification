"""
predict.py
----------
Load the trained ResNet18 model and predict Fresh/Rotten for a single
image, either from the command line or by importing `predict_image()`
from other scripts (e.g. app.py).

USAGE (command line):
    python predict.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "food_quality_model.pth"
DEVICE = torch.device("cpu")


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_model(model_path: Path = MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Please run 'python train.py' first to train and save the model."
        )
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((checkpoint["image_size"], checkpoint["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["mean"], checkpoint["std"]),
        ]
    )
    return model, tfms, class_names


def preprocess_pil_image(pil_image: Image.Image, tfms) -> torch.Tensor:
    pil_image = pil_image.convert("RGB")
    tensor = tfms(pil_image)
    return tensor.unsqueeze(0)  # add batch dimension


@torch.no_grad()
def predict_image(image_path: str = None, pil_image: Image.Image = None, model=None, tfms=None, class_names=None):
    """
    Predict Fresh/Rotten for a single image.

    Provide either `image_path` (str/Path) OR an already-loaded `pil_image`.
    If `model`/`tfms`/`class_names` are not provided, the saved model on
    disk will be loaded automatically (slower if called repeatedly - for
    a UI like Streamlit, load once and pass them in).

    Returns:
        dict with keys: label, confidence (0-100 float), probabilities (dict)
    """
    if model is None or tfms is None or class_names is None:
        model, tfms, class_names = load_model()

    if pil_image is None:
        if image_path is None:
            raise ValueError("Provide either image_path or pil_image.")
        pil_image = Image.open(image_path)

    input_tensor = preprocess_pil_image(pil_image, tfms).to(DEVICE)
    outputs = model(input_tensor)
    probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    label = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    probabilities = {class_names[i]: float(probs[i]) * 100.0 for i in range(len(class_names))}

    return {"label": label, "confidence": confidence, "probabilities": probabilities}


def read_image_with_opencv(image_path: str) -> Image.Image:
    """Read an image with OpenCV (BGR) and convert to a PIL RGB image.
    Used to satisfy the OpenCV requirement / to support additional
    image-processing steps (e.g. resizing/denoising) before inference."""
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"OpenCV could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description="Predict Fresh/Rotten for a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to an image file")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image = read_image_with_opencv(image_path)
    result = predict_image(pil_image=pil_image)

    print("\n" + "=" * 40)
    print(f"Image        : {image_path}")
    print(f"Prediction   : {result['label'].upper()}")
    print(f"Confidence   : {result['confidence']:.2f}%")
    print("Class probabilities:")
    for cls, p in result["probabilities"].items():
        print(f"   {cls:10s}: {p:.2f}%")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
