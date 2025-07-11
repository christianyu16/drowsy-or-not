import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Drowsiness Detection ")

model = tf.keras.models.load_model("eye_cnn_model.h5")
class_names = ["Awake", "Drowsy"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).resize((224, 224)).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = class_names[int(prediction[0][0] > 0.5)]

    st.write(f"### Prediction: {label}")
