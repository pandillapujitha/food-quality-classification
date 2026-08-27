# 🍎 Deep Learning-Based Food Quality Classification System Using PyTorch

A complete end-to-end deep learning project that classifies food images as
**Fresh** or **Rotten** using a **ResNet18** convolutional neural network
(transfer learning), trained on **CPU** with **PyTorch**, and served through
a **Streamlit** web application.

---

## 📌 Project Overview

Food spoilage is a major concern in agriculture, retail, and household food
management. This project builds an automated **Food Quality Classification
System** that takes a photo of a fruit/vegetable and predicts whether it is
**Fresh** or **Rotten**, along with a confidence score.

The system covers the full ML lifecycle:

1. **Dataset acquisition** — automatic download & organization
2. **Model training** — ResNet18 fine-tuned on CPU
3. **Evaluation** — accuracy/loss curves + confusion matrix
4. **Inference** — single-image prediction via CLI
5. **Deployment** — an interactive Streamlit web app

---

## ✨ Features

- ✅ Automatic dataset download & folder organization (no manual setup)
- ✅ ResNet18 transfer learning (PyTorch, ImageNet-pretrained backbone)
- ✅ CPU-only training (no GPU required)
- ✅ Command-line training script with configurable epochs/batch size/LR
- ✅ Command-line single-image prediction script
- ✅ Streamlit web app:
  - Upload an image
  - View the uploaded image
  - Get a Fresh / Rotten prediction
  - See the confidence percentage
  - Full per-class probability breakdown
  - Sidebar view of training accuracy curve, loss curve & confusion matrix
- ✅ Training accuracy & loss curves (Matplotlib)
- ✅ Confusion matrix (Scikit-learn + Matplotlib)
- ✅ Clean, professional, responsive UI
- ✅ OpenCV used in the preprocessing pipeline

---

## 🛠️ Technologies Used

| Category            | Technology                          |
|----------------------|--------------------------------------|
| Language             | Python 3.11                          |
| Deep Learning        | PyTorch, Torchvision (ResNet18)      |
| Web App / Frontend   | Streamlit                            |
| Image Processing     | OpenCV, PIL (Pillow)                 |
| Numerical Computing  | NumPy                                |
| Visualization        | Matplotlib                           |
| Evaluation Metrics   | Scikit-learn (confusion matrix, report) |
| Dataset Download     | KaggleHub                            |

> **Note:** This project uses **PyTorch only**. TensorFlow and Keras are
> **not** used anywhere in this codebase.

---

## 📂 Dataset Information

This project uses the public Kaggle dataset:

**"Fruits fresh and rotten for classification"**
🔗 https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

The original dataset contains fine-grained folders such as `freshapples`,
`freshbanana`, `freshoranges`, `rottenapples`, `rottenbanana`,
`rottenoranges`, etc.

`download_dataset.py` automatically:

1. Downloads the dataset via `kagglehub` (prompts a one-time Kaggle login
   if needed).
2. Merges all `fresh*` folders into a single **`fresh`** class.
3. Merges all `rotten*` folders into a single **`rotten`** class.
4. Organizes everything into:

```
dataset/
├── train/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
```

5. If no explicit test split exists in the source data, it automatically
   carves out ~10% of the training images into `dataset/test/` so you
   always get a proper held-out evaluation set (used for the confusion
   matrix).

You never need to manually create folders or move images.

### If you can't access Kaggle from your environment
Open `download_dataset.py` and set `MANUAL_DATASET_DIR` to the path of any
dataset folder you already have (it just needs subfolders containing
"fresh" or "rotten" in their names) — the script will auto-organize it for
you. Instructions for creating a free Kaggle API token are also printed by
the script if the automatic download fails.

---

## 💻 Installation

### 1. Clone or extract this project
```bash
git clone <your-repo-url>
cd food_quality_classification
```
*(If you extracted this from a ZIP file, just `cd` into the extracted folder.)*

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download & organize the dataset (automatic)
```bash
python download_dataset.py
```

---

## 🏋️ Training

Train the ResNet18 model on CPU:

```bash
python train.py
```

Optional arguments:
```bash
python train.py --epochs 15 --batch-size 32 --lr 0.0001 --val-split 0.2
```

This will:
- Split the training data into train/validation subsets
- Fine-tune a pretrained ResNet18 on your CPU
- Save the best model to `models/food_quality_model.pth`
- Save `plots/accuracy_curve.png`, `plots/loss_curve.png`,
  `plots/confusion_matrix.png`, and `plots/training_history.json`
- Print a full classification report (precision/recall/F1) in the terminal

> Training on CPU is slower than on GPU. For a quick first run, try a
> smaller number of epochs (e.g. `--epochs 5`) to confirm everything works
> end-to-end before running a longer training session.

---

## 🔍 Prediction (Command Line)

Classify a single image from the terminal:

```bash
python predict.py --image path/to/your/image.jpg
```

Example output:
```
========================================
Image        : sample_apple.jpg
Prediction   : FRESH
Confidence   : 96.42%
Class probabilities:
   fresh     : 96.42%
   rotten    : 3.58%
========================================
```

---

## 🌐 Running the Streamlit Web App

Once you have a trained model (`models/food_quality_model.pth` exists):

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`)
in your browser. From there you can:

1. Upload a food image (JPG/PNG)
2. View the uploaded image
3. Click **Classify Image**
4. See the Fresh/Rotten prediction and confidence percentage
5. View training accuracy/loss curves and the confusion matrix in the sidebar

---

## 📤 GitHub Upload

To push this project to your own GitHub repository:

```bash
# Initialize git (skip if already a repo)
git init

# Add all files (dataset/ and models/*.pth are excluded via .gitignore)
git add .
git commit -m "Initial commit: Food Quality Classification System"

# Create a new repository on GitHub first, then:
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```

> `.gitignore` is already configured to exclude the (potentially large)
> dataset images and trained model weights from version control. If you
> want to share the trained model, consider using
> [Git LFS](https://git-lfs.com/) or uploading it to a release/cloud
> storage link and referencing it in this README.

---

## ☁️ Streamlit Deployment (Streamlit Community Cloud)

1. Push this project to a **public GitHub repository** (see above).
2. Make sure `requirements.txt` is present at the repo root (it already is).
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Click **"New app"**, select your repository, branch, and set the main
   file path to `app.py`.
5. Click **Deploy**.

**Important:** Since `models/food_quality_model.pth` is excluded from git
by default, you have two options for deployment:
- **Option A (recommended):** Remove `models/*.pth` from `.gitignore` and
  commit the trained model file directly (works fine if the model is a
  few tens of MB).
- **Option B:** Host the `.pth` file externally (e.g. Google Drive,
  Hugging Face Hub, GitHub Releases) and add a small download step at the
  top of `app.py` that fetches it into `models/` on first run.

Similarly, the `dataset/` folder is **not required** at deployment time —
only the trained model is needed to run the Streamlit app.

---

## 📁 Folder Structure

```
food_quality_classification/
│
├── app.py                     # Streamlit web application
├── train.py                   # Model training script
├── predict.py                 # Single-image prediction script (CLI + importable)
├── download_dataset.py        # Automatic dataset download & organization
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Git ignore rules
│
├── models/                    # Trained model weights
│   └── food_quality_model.pth # (created after running train.py)
│
├── dataset/                   # Dataset (auto-created by download_dataset.py)
│   ├── train/
│   │   ├── fresh/
│   │   └── rotten/
│   └── test/
│       ├── fresh/
│       └── rotten/
│
├── plots/                     # Generated evaluation plots
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── training_history.json
│
└── assets/                    # Static assets (icons/screenshots for README/app)
```

---

## 🚀 Future Improvements

- Add support for multi-class classification (specific fruit/vegetable +
  freshness level, e.g. "slightly rotten", "very rotten")
- Add Grad-CAM visualizations to explain model predictions
- Support batch prediction (upload multiple images at once)
- Add REST API (FastAPI) alongside the Streamlit UI for programmatic access
- Add data augmentation experiments (mixup, cutmix) to improve robustness
- Add model quantization / ONNX export for faster CPU inference
- Add automated hyperparameter tuning (e.g. Optuna)
- Add Dockerfile for containerized deployment
- Add unit tests for data pipeline and inference functions

---

## 📄 License

This project is provided for educational purposes. The dataset used is
subject to its own license on Kaggle — please review the dataset page for
usage terms before any commercial use.
