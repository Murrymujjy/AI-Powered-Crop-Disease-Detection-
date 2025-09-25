import torch
import json
from torchvision import transforms
from torchvision.models import efficientnet_b0
from collections import OrderedDict
import streamlit as st
from PIL import Image

# --- Model & Data Loading (all in one place for easy access) ---

# Define the custom transformation class
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# Load the class names from the saved JSON file
@st.cache_resource
def load_class_names():
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        st.error("Class names file not found. Please ensure 'class_names.json' is in the same directory.")
        st.stop()
        
class_names = load_class_names()

# Define the preprocessing pipeline, exactly as used for training
mean = [0.4728, 0.5311, 0.3941]
std = [0.2185, 0.2149, 0.2605]

transform_pipeline = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load the trained PyTorch model
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    
    state_dict = torch.load('crop_disease_model.pth', map_location=torch.device('cpu'))
    
    num_classes = len(class_names)
    
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 256),
        torch.nn.Linear(256, num_classes)
    )

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    model.eval()
    return model

model = load_model()

# The main prediction function
def classify_crop_disease(image):
    image_tensor = transform_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)
        
    confidences = torch.nn.functional.softmax(predictions[0], dim=0)
    
    confidences_dict = {
        class_names[i]: float(confidences[i]) for i in range(len(class_names))
    }
    
    sorted_confidences = sorted(confidences_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_confidences
