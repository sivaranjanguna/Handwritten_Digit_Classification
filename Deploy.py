from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# --------------------------------------------------
# Minimal Dark Theme Styling
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main {
    background-color: #0e1117;
}

h1 {
    text-align: center;
    color: #7c3aed;
    font-weight: 800;
}

h3 {
    color: #e5e7eb;
}

.prediction-text {
    text-align: center;
    font-size: 40px;
    font-weight: 800;
    color: #22c55e;
    margin-top: 20px;
}

.confidence-text {
    text-align: center;
    font-size: 18px;
    color: #c7c7c7;
}

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 13px;
    margin-top: 40px;
}

.stButton>button {
    width: 100%;
    background-color: #7c3aed;
    color: white;
    font-size: 18px;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Handwritten_Digit_Model.keras")

model = load_model()

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def is_canvas_blank(img_array, threshold=10):
    return np.sum(img_array > threshold) < 50

# --------------------------------------------------
# App Header
# --------------------------------------------------
st.markdown("# ✍️ Handwritten Digit Recognition")
st.markdown(
    "<p style='text-align:center; color:#c7c7c7;'>"
    "Draw a digit (0–9) on the canvas or upload an image, then click <b>Predict Digit</b>"
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# Canvas Section
# --------------------------------------------------
st.markdown("### 🎨 Draw Digit")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

predict_clicked = st.button("🔍 Predict Digit")

# --------------------------------------------------
# Prediction from Canvas
# --------------------------------------------------
if predict_clicked:

    if canvas_result.image_data is None:
        st.warning("✏️ Please draw a digit first")
    else:
        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        ).convert("L")

        img = img.resize((28, 28))
        img_array = np.array(img)

        if is_canvas_blank(img_array):
            st.warning("✏️ Please draw a digit before predicting")
        else:
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 28, 28)

            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.markdown(
                f"<div class='prediction-text'>The written digit is {predicted_digit}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='confidence-text'>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

st.divider()

# --------------------------------------------------
# Image Upload Section
# --------------------------------------------------
st.markdown("### 📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image (JPG / PNG)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    st.markdown(
        f"<div class='prediction-text'>The written digit is {predicted_digit}</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='confidence-text'>Confidence: {confidence:.2f}%</div>",
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    "<div class='footer'>Built with ❤️ using TensorFlow & Streamlit</div>",
    unsafe_allow_html=True
)
