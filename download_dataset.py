"""
download_dataset.py
--------------------
Automatically downloads and organizes the "Fruits fresh and rotten for
classification" dataset (Kaggle: sriramr/fruits-fresh-and-rotten-for-classification)
into the folder structure expected by train.py:

    dataset/
        train/
            fresh/
            rotten/
        test/
            fresh/
            rotten/

The original dataset ships with fine-grained folders such as
"freshapples", "freshbanana", "freshoranges", "rottenapples", etc.
This script merges all "fresh*" folders into a single "fresh" class and
all "rotten*" folders into a single "rotten" class, which is exactly
the binary Fresh vs Rotten setup this project trains on.

USAGE:
    python download_dataset.py

REQUIREMENTS:
    pip install kagglehub
    A free Kaggle account + API token (kaggle.json), OR simply let
    kagglehub prompt you to log in interactively the first time.

If you cannot use Kaggle in your environment, see the "Manual dataset
setup" section printed at the bottom of this script's output -- you can
point MANUAL_DATASET_DIR at any folder that already contains
"fresh"/"rotten" style subfolders and this script will reorganize it.
"""

import os
import shutil
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

KAGGLE_DATASET_SLUG = "sriramr/fruits-fresh-and-rotten-for-classification"

# If Kaggle download is not possible in your environment, set this to a
# local folder path that contains train/ and test/ subfolders (or any
# fresh*/rotten* subfolders) and re-run this script.
MANUAL_DATASET_DIR = None  # e.g. r"C:\Users\me\Downloads\dataset"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def ensure_dirs():
    for split in (TRAIN_DIR, TEST_DIR):
        for cls in ("fresh", "rotten"):
            (split / cls).mkdir(parents=True, exist_ok=True)


def classify_folder_name(name: str):
    """Return 'fresh', 'rotten', or None based on a folder name."""
    lname = name.lower()
    if "fresh" in lname:
        return "fresh"
    if "rotten" in lname or "spoiled" in lname or "stale" in lname:
        return "rotten"
    return None


def copy_images(src_folder: Path, dest_folder: Path):
    count = 0
    for f in src_folder.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            dest = dest_folder / f"{src_folder.name}_{f.name}"
            if not dest.exists():
                shutil.copy2(f, dest)
                count += 1
    return count


def organize_raw_dataset(raw_root: Path):
    """
    Walk a raw dataset root (which may or may not already have
    train/test splits) and merge fresh*/rotten* folders into
    dataset/train/{fresh,rotten} and dataset/test/{fresh,rotten}.
    """
    ensure_dirs()
    total_copied = 0

    # Case 1: raw_root already has train/ and test/ (or Train/ Test/) splits
    split_map = {}
    for child in raw_root.iterdir():
        if child.is_dir() and child.name.lower() in ("train", "training"):
            split_map["train"] = child
        elif child.is_dir() and child.name.lower() in ("test", "testing", "val", "validation"):
            split_map["test"] = child

    if split_map:
        for split_name, split_path in split_map.items():
            dest_split = TRAIN_DIR if split_name == "train" else TEST_DIR
            for sub in split_path.iterdir():
                if not sub.is_dir():
                    continue
                cls = classify_folder_name(sub.name)
                if cls is None:
                    continue
                copied = copy_images(sub, dest_split / cls)
                total_copied += copied
                print(f"  [{split_name}] {sub.name} -> {cls}  ({copied} images)")
    else:
        # Case 2: raw_root directly contains fresh*/rotten* folders with
        # no explicit train/test split -> put everything in train, we
        # will create a validation split automatically inside train.py.
        for sub in raw_root.iterdir():
            if not sub.is_dir():
                continue
            cls = classify_folder_name(sub.name)
            if cls is None:
                continue
            copied = copy_images(sub, TRAIN_DIR / cls)
            total_copied += copied
            print(f"  [train] {sub.name} -> {cls}  ({copied} images)")

    return total_copied


def download_with_kagglehub():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Install it with: pip install kagglehub")
        return None

    print(f"Downloading dataset '{KAGGLE_DATASET_SLUG}' via kagglehub ...")
    try:
        path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
        print(f"Downloaded to: {path}")
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        print(f"kagglehub download failed: {exc}")
        return None


def dataset_already_populated():
    for split in (TRAIN_DIR, TEST_DIR):
        for cls in ("fresh", "rotten"):
            folder = split / cls
            if folder.exists():
                images = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTS]
                if images:
                    return True
    return False


def main():
    print("=" * 70)
    print("Food Quality Dataset Downloader")
    print("=" * 70)

    ensure_dirs()

    if dataset_already_populated():
        print("Dataset already appears to be populated under dataset/train "
              "and dataset/test. Skipping download.")
        print("Delete the images inside dataset/train/* and dataset/test/* "
              "if you want to force a re-download.")
        return

    raw_root = None

    if MANUAL_DATASET_DIR:
        candidate = Path(MANUAL_DATASET_DIR)
        if candidate.exists():
            raw_root = candidate
            print(f"Using manually specified dataset directory: {raw_root}")
        else:
            print(f"MANUAL_DATASET_DIR '{MANUAL_DATASET_DIR}' does not exist.")

    if raw_root is None:
        raw_root = download_with_kagglehub()

    if raw_root is None:
        print("\n" + "=" * 70)
        print("AUTOMATIC DOWNLOAD FAILED")
        print("=" * 70)
        print(
            "Manual dataset setup:\n"
            "  1. Create a free Kaggle account: https://www.kaggle.com\n"
            "  2. Go to https://www.kaggle.com/settings -> 'Create New Token'\n"
            "     to download kaggle.json, then place it at:\n"
            "       Linux/Mac : ~/.kaggle/kaggle.json\n"
            "       Windows   : C:\\Users\\<you>\\.kaggle\\kaggle.json\n"
            "  3. Re-run: python download_dataset.py\n"
            "\n"
            "  OR, download the dataset manually from:\n"
            "     https://www.kaggle.com/datasets/sriramr/"
            "fruits-fresh-and-rotten-for-classification\n"
            "  extract it anywhere, set MANUAL_DATASET_DIR in this file to\n"
            "  that folder's path, and re-run this script.\n"
        )
        sys.exit(1)

    print("\nOrganizing dataset into dataset/train/{fresh,rotten} and "
          "dataset/test/{fresh,rotten} ...")
    total = organize_raw_dataset(raw_root)

    if total == 0:
        print("\nNo images were copied. Please check the downloaded dataset "
              "structure and adjust classify_folder_name() if needed.")
        sys.exit(1)

    # If no separate test split was found, carve one out of train (10%)
    make_test_split_if_missing()

    print(f"\nDone! Copied {total} images in total.")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Test dir : {TEST_DIR}")
    print("\nYou can now run: python train.py")


def make_test_split_if_missing():
    """If dataset/test/{fresh,rotten} are empty, move ~10% of train images
    into test so evaluation / confusion matrix has held-out data."""
    import random

    random.seed(42)
    for cls in ("fresh", "rotten"):
        test_folder = TEST_DIR / cls
        train_folder = TRAIN_DIR / cls
        existing_test = [f for f in test_folder.iterdir() if f.suffix.lower() in VALID_EXTS]
        if existing_test:
            continue
        train_images = [f for f in train_folder.iterdir() if f.suffix.lower() in VALID_EXTS]
        if not train_images:
            continue
        n_move = max(1, int(0.1 * len(train_images)))
        random.shuffle(train_images)
        for f in train_images[:n_move]:
            shutil.move(str(f), str(test_folder / f.name))
        print(f"  Created test split for '{cls}': moved {n_move} images from train -> test")


if __name__ == "__main__":
    main()
