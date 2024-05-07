import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load classifier
model = load_model('/content/drive/MyDrive/Colab Notebooks/Saved Model/classification.h5')

# Load labels
with open('/content/drive/MyDrive/Colab Notebooks/Saved Model/labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Define classification function
def classify(image, model, labels):
    # Resize and preprocess image
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    image_array = np.array(image)
    normalized_image_array = image_array.astype(np.float32) / 255.0
    data = np.expand_dims(normalized_image_array, axis=0)

    # Perform prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    label = labels[index]
    confidence_score = prediction[0][index]

    return label, confidence_score

# Set title
st.title('Brain Tumour Classification')

# Set header
st.header('Please upload the MRI Image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Display image and classify
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    label, confidence_score = classify(image, model, labels)
    st.write(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence_score}")
