import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("eye_cnn_model.h5")
    return model

model = load_model()

# UI
st.title("🧠 Eye State Classifier (Awake vs Drowsy)")

uploaded_file = st.file_uploader("Upload an eye image (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))  # Convert to grayscale and resize
    st.image(image, caption='Uploaded Image', width=150)

    # Preprocess image
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0

    # Predict
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]

    # Labels (Adjust depending on your training labels)
    class_names = ["Awake", "Drowsy", "Other"] if model.output_shape[-1] == 3 else ["Awake", "Drowsy"]

    st.markdown(f"### Prediction: **{class_names[class_id]}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")
