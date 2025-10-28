import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import io
import base64
from streamlit_drawable_canvas import st_canvas

# --- Page Configuration ---
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Custom Dark Theme CSS ---
dark_theme_css = """
<style>
body {
    background-color: #0f1116;
    color: #ffffff;
}
header, .stApp {
    background: #0f1116;
}
h1, h2, h3, h4 {
    color: #00FFFF;
    text-align: center;
}
.sidebar .sidebar-content {
    background: #1c1c1c;
}
.stButton>button {
    background-color: #00FFFF;
    color: black;
    border-radius: 10px;
    font-size: 18px;
    padding: 10px 24px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #0099cc;
    color: white;
}
.uploadedFile {
    background-color: #1c1c1c;
}
div[data-testid="stToolbar"] { display: none; }
footer {visibility: hidden;}
</style>
"""
st.markdown(dark_theme_css, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()

# --- App Title ---
st.title("🧠 Handwritten Digit Recognition (MNIST)")
st.markdown("### Upload an image or draw a digit below (0–9):")

# --- Tabs for Upload or Draw ---
option = st.radio("Choose Input Method:", ("🖼 Upload Image", "✍️ Draw Digit"), index=1)

# --- Image Upload Section ---
img = None
if option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload a 28x28 pixel digit image (PNG or JPG):", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Image", use_container_width=True)

# --- Drawing Canvas Section ---
elif option == "✍️ Draw Digit":
    st.markdown("**Draw your digit below (white on black):**")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))

# --- Prediction Function ---
def predict_digit(image):
    image = ImageOps.invert(image)  # Invert colors (white bg → black bg)
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return predicted_label, confidence

# --- Predict Button ---
if st.button("🔍 Predict Digit"):
    if img is not None:
        label, conf = predict_digit(img)
        st.success(f"### ✅ Predicted Digit: **{label}**")
        st.markdown(f"**Confidence:** {conf:.2f}%")
    else:
        st.warning("⚠️ Please upload or draw a digit first!")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Developed by Parth Biru ✨</p>", unsafe_allow_html=True)
