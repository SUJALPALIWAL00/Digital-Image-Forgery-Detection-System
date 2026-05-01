import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os

def convert_to_ela(image, quality=90):
    original = image.convert('RGB')
    temp_filename = "temp_inference.jpg"
    original.save(temp_filename, 'JPEG', quality=quality)
    resaved = Image.open(temp_filename)
    ela_image = ImageChops.difference(original, resaved)
    
    scale = 20.0 # Match your new training scale
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

st.set_page_config(page_title="Forensic Image Detector", layout="wide")
st.title("🔬 Digital Image Forgery Detection System")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    ela_img = convert_to_ela(img)
    
    try:
        if os.path.exists('trained_model.h5'):
            model = tf.keras.models.load_model('trained_model.h5')
            test_img = ela_img.resize((128, 128))
            test_arr = np.array(test_img).astype('float32') / 255.0
            test_arr = np.expand_dims(test_arr, axis=0)
            
            # Prediction line (CLEANED)
            prediction = model.predict(test_arr)[0][0]
        else:
            st.error("Model file 'trained_model.h5' not found.")
            prediction = None
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        prediction = None

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(ela_img, caption="ELA Map", use_container_width=True)

    if prediction is not None:
        st.divider()
        if prediction > 0.5:
            st.error(f"🚩 RESULT: FORGED (Confidence: {prediction*100:.2f}%)")
        else:
            st.success(f"✅ RESULT: AUTHENTIC (Confidence: {(1-prediction)*100:.2f}%)")