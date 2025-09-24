import streamlit as st
import torch
import json
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# Set the title of the app
st.title("AI-Powered Crop Disease Detection")
st.write("Upload an image of a crop leaf (cashew, cassava, maize, or tomato) to detect potential diseases.")

# 1. Define the custom transformation class
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# 2. Load the class names from the saved JSON file
@st.cache_resource
def load_class_names():
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    return class_names

class_names = load_class_names()

# 3. Define the preprocessing pipeline, exactly as used for training
mean = [0.4728, 0.5311, 0.3941]
std = [0.2185, 0.2149, 0.2605]

transform_pipeline = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 4. Load the trained PyTorch model architecture and its state_dict
@st.cache_resource
def load_model():
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    
    # Load the saved state_dict
    model.load_state_dict(torch.load('crop_disease_model.pth', map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    return model

model = load_model()

# 5. Create the prediction function
def classify_crop_disease(image):
    # Apply transformations
    image_tensor = transform_pipeline(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        predictions = model(image_tensor)
        
    confidences = torch.nn.functional.softmax(predictions[0], dim=0)
    
    # Create a dictionary of results
    confidences_dict = {
        class_names[i]: float(confidences[i]) for i in range(len(class_names))
    }
    
    # Sort the dictionary by confidence for better display
    sorted_confidences = sorted(confidences_dict.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_confidences

# Streamlit UI for file upload and display
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Make a prediction
    with st.spinner('Analyzing the image...'):
        results = classify_crop_disease(image)

    # Display the results
    st.subheader("Prediction Results:")
    
    # Display the top prediction
    top_class, top_confidence = results[0]
    st.success(f"**Predicted Class:** {top_class} (Confidence: {top_confidence:.2f})")

    st.markdown("---")
    
    # Display the full list of confidences
    st.subheader("All Confidence Scores:")
    for class_name, confidence in results:
        st.write(f"- {class_name}: {confidence:.4f}")
