import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.detector import predict_emotion

# Streamlit app layout
st.set_page_config(page_title="Facial Emotion Detection", layout="centered")

st.title("😃 Facial Emotion Detection using Deep Learning")
st.markdown("Upload a face image to detect the **emotion** expressed.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Predict emotion(s)
    results = predict_emotion(image_cv)

    if results:
        st.success(f"✅ Detected {len(results)} face(s)")
        for i, res in enumerate(results, 1):
            label = res['label']
            confidence = res['confidence']
            st.markdown(f"**Face {i}**: `{label}` (Confidence: `{confidence * 100:.2f}%`)")
    else:
        st.warning("⚠️ No face detected. Please try a different image.")
