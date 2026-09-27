import streamlit as st
from fastai.vision.all import *
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🌿 Image Classifier")
st.write("Classify images as: Bird, Forest, Mountain, or River")

# Load model
@st.cache_resource
def load_model():
    return load_learner('notebooks/Chapter1/export.pkl')

learn = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    st.write("Classifying...")
    pred, idx, probs = learn.predict(image)
    
    # Show results
    st.success(f"Predicted: **{pred}**")
    
    # Show probabilities
    st.write("Confidence scores:")
    for class_name, prob in zip(learn.dls.vocab, probs):
        st.write(f"- {class_name}: {prob:.4f}")