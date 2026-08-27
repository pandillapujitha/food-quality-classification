from pathlib import Path

import streamlit as st
from PIL import Image

from predict import load_model, predict_image, MODEL_PATH


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"


st.set_page_config(
    page_title="Food Quality Classifier",
    page_icon="🍎",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🍎 Food Quality Classification System")

st.write(
    "Deep Learning (PyTorch · ResNet18) — "
    "Detect Fresh vs Rotten Food from an Image"
)

st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ About this App")

    st.write(
        "This application uses a **ResNet18** convolutional "
        "neural network trained with PyTorch."
    )

    st.write("The model classifies food as:")

    st.success("🟢 Fresh")

    st.error("🔴 Rotten")

    st.write(
        "**Pipeline:**\n\n"
        "PIL → Preprocessing → ResNet18 → Softmax → Prediction"
    )

    st.markdown("---")

    st.header("📊 Training Performance")

    acc_path = PLOTS_DIR / "accuracy_curve.png"
    loss_path = PLOTS_DIR / "loss_curve.png"
    cm_path = PLOTS_DIR / "confusion_matrix.png"

    if acc_path.exists():
        st.image(
            str(acc_path),
            caption="Training vs Validation Accuracy",
            width=300,
        )

    if loss_path.exists():
        st.image(
            str(loss_path),
            caption="Training vs Validation Loss",
            width=300,
        )

    if cm_path.exists():
        st.image(
            str(cm_path),
            caption="Confusion Matrix",
            width=300,
        )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():

    return load_model()


if not MODEL_PATH.exists():

    st.error(
        "❌ Model not found.\n\n"
        "Please run:\n\n"
        "`.\\venv\\Scripts\\python.exe train.py --epochs 5`"
    )

    st.stop()


try:

    model, tfms, class_names = get_model()

except Exception as e:

    st.error("❌ Could not load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.header("📤 Upload a Food Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "jfif",
    ],
)


# ============================================================
# IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

    except Exception as e:

        st.error("❌ Unable to open this image.")

        st.exception(e)

        st.stop()


    st.subheader("🖼️ Uploaded Image")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.image(
            image,
            caption=uploaded_file.name,
            width=400,
        )


    st.markdown("---")


    # ========================================================
    # BUTTON
    # ========================================================

    classify = st.button(
        "🔍 Classify Image",
        type="primary",
    )


    if classify:

        with st.spinner("🔎 Analyzing image..."):

            try:

                result = predict_image(
                    pil_image=image,
                    model=model,
                    tfms=tfms,
                    class_names=class_names,
                )

            except Exception as e:

                st.error("❌ Prediction failed.")

                st.exception(e)

                st.stop()


        # ====================================================
        # GET RESULT
        # ====================================================

        label = result.get("label", "Unknown")

        confidence = result.get("confidence", 0)

        probabilities = result.get(
            "probabilities",
            {},
        )


        try:

            confidence = float(confidence)

        except:

            confidence = 0.0


        confidence = max(
            0.0,
            min(
                confidence,
                100.0,
            ),
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.header("🎯 Prediction")


        if label.lower() == "fresh":

            st.success(
                f"🟢 FRESH\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        elif label.lower() == "rotten":

            st.error(
                f"🔴 ROTTEN\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.warning(
                f"Prediction: {label}\n\n"
                f"Confidence: {confidence:.2f}%"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        st.subheader("📈 Prediction Confidence")

        st.progress(
            int(confidence)
        )

        st.write(
            f"**{confidence:.2f}% confident**"
        )


        # ====================================================
        # PROBABILITIES
        # ====================================================

        st.subheader(
            "📊 Class Probabilities"
        )


        if isinstance(probabilities, dict):

            for cls, probability in sorted(
                probabilities.items(),
                key=lambda x: float(x[1]),
                reverse=True,
            ):

                try:

                    probability = float(
                        probability
                    )

                except:

                    probability = 0.0


                probability = max(
                    0.0,
                    min(
                        probability,
                        100.0,
                    ),
                )


                st.write(
                    f"**{str(cls).capitalize()}**"
                )

                st.progress(
                    int(probability)
                )

                st.caption(
                    f"{probability:.2f}%"
                )


else:

    st.info(
        "👆 Upload an image above, "
        "then click **Classify Image**."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Built with PyTorch, Streamlit, OpenCV, PIL, NumPy, "
    "Matplotlib & Scikit-learn. "
    "Model: ResNet18 (Transfer Learning), trained on CPU."
)
