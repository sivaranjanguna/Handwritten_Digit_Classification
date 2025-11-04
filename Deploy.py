import streamlit as st
import tensorflow as tf
import numpy as np 
import cv2
from PIL import Image

# load pre-trained model
model = tf.keras.models.load_model('Handwritten_Digit_Model.keras')

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will predict the digit.")

# upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # preprocess image
    image = Image.open(uploaded_file)
    image = image.convert("L")  # convert to grayscale
    image = image.resize((28, 28))  # resize to 28x28
    image = np.array(image) / 255.0  # normalize pixel values
    image = np.expand_dims(image, axis=0)  # add batch dimension

    # make prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Digit: {predicted_digit}")