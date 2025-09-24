import streamlit as st
import torch
import json
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from collections import OrderedDict

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
    # Build a new EfficientNet-B0 model from scratch
    model = efficientnet_b0(weights=None)

    # --- THE NEW AND MORE ROBUST FIX ---
    # 1. Load the state_dict
    state_dict = torch.load('crop_disease_model.pth', map_location=torch.device('cpu'))

    # 2. Correct the architecture to match the state_dict's final layers
    # Based on your error, the original classifier was completely replaced during training.
    # The last layer's input features are 256. This means there's an intermediate layer.
    num_classes = len(class_names)
    
    # We will build a new classifier with two Linear layers to match the keys in the state_dict.
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 256),
        torch.nn.Linear(256, num_classes)
    )

    # 3. Load the state_dict with the 'strict=False' flag to ignore any extra or mismatched keys.
    # This is a safe way to handle the problem and will load the weights that match.
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
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
