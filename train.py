"""
train.py
--------
Trains a ResNet18 (PyTorch, ImageNet-pretrained) model to classify food
images as "fresh" or "rotten". Runs entirely on CPU.

Outputs:
    models/food_quality_model.pth   -> trained model weights + metadata
    plots/accuracy_curve.png        -> training/validation accuracy per epoch
    plots/loss_curve.png            -> training/validation loss per epoch
    plots/confusion_matrix.png      -> confusion matrix on the held-out test set
    plots/training_history.json     -> raw numbers behind the plots

USAGE:
    python train.py
    python train.py --epochs 15 --batch-size 32 --lr 0.0005
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------------
# Paths / constants
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODEL_PATH = MODELS_DIR / "food_quality_model.pth"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cpu")  # explicitly CPU as required


def get_transforms():
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def build_model(num_classes: int = 2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def check_dataset():
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        raise FileNotFoundError(
            f"No data found in {TRAIN_DIR}.\n"
            "Please run 'python download_dataset.py' first to download and "
            "organize the dataset."
        )
    class_dirs = [d for d in TRAIN_DIR.iterdir() if d.is_dir()]
    empty = [d.name for d in class_dirs if not any(d.glob("*.*"))]
    if len(class_dirs) < 2 or empty:
        raise FileNotFoundError(
            f"Expected at least 2 non-empty class folders inside {TRAIN_DIR} "
            f"(e.g. 'fresh' and 'rotten'). Found empty/missing: {empty or class_dirs}.\n"
            "Please run 'python download_dataset.py' first."
        )


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        total += inputs.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = running_loss / total if total else 0.0
    avg_acc = running_correct / total if total else 0.0
    return avg_loss, avg_acc, all_preds, all_labels


def plot_curves(history, out_dir: Path):
    epochs = range(1, len(history["train_acc"]) + 1)

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, out_dir: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Food Quality Classifier (ResNet18)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    check_dataset()

    train_tfms, eval_tfms = get_transforms()

    full_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tfms)
    class_names = full_dataset.classes  # e.g. ['fresh', 'rotten']
    print(f"Detected classes: {class_names}")
    print(f"Total training images found: {len(full_dataset)}")

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    # Validation subset should use eval transforms (no augmentation)
    val_subset.dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=eval_tfms)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Held-out test set (created by download_dataset.py)
    test_loader = None
    if TEST_DIR.exists() and any(TEST_DIR.rglob("*.*")):
        test_dataset = datasets.ImageFolder(str(TEST_DIR), transform=eval_tfms)
        if len(test_dataset) > 0:
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    print(f"\nTraining on device: {DEVICE} for {args.epochs} epochs\n")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"({elapsed:.1f}s)"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes. Best val acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model + metadata needed by predict.py / app.py
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "architecture": "resnet18",
        },
        MODEL_PATH,
    )
    print(f"Model saved to: {MODEL_PATH}")

    # Plots for accuracy / loss
    plot_curves(history, PLOTS_DIR)
    with open(PLOTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Accuracy/loss curves saved to: {PLOTS_DIR}")

    # Confusion matrix on the held-out test set (fallback: validation set)
    eval_loader = test_loader if test_loader is not None else val_loader
    eval_name = "test" if test_loader is not None else "validation"
    _, eval_acc, preds, labels = evaluate(model, eval_loader, criterion)
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names, PLOTS_DIR)
    print(f"Confusion matrix ({eval_name} set, acc={eval_acc:.4f}) saved to: {PLOTS_DIR}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=class_names))


if __name__ == "__main__":
    main()
