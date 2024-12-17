import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Define paths and URLs
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model_url = "https://drive.google.com/uc?id=1rKh-IElSdHTqax7XdfSdZTn-r8T_qWPf"  # Direct download link for the model

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    print("Downloading model...")
    response = requests.get(model_url, stream=True)
    with open(model_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download complete!")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load the class indices from the JSON file
class_indices_path = f"{working_dir}/class_indices.json"
class_indices = json.load(open(class_indices_path))

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, uploaded_image, class_indices):
    preprocessed_img = load_and_preprocess_image(uploaded_image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Disease Classifier')

# File uploader for user to upload an image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
