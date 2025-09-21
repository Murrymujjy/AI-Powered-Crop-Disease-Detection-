import gradio as gr
import torch
import json
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

# 1. Define your custom transformation class
class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

# 2. Load your class names from the saved JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# The mean and std values you calculated from your dataset
mean = [0.4728, 0.5311, 0.3941]
std = [0.2185, 0.2149, 0.2605]

# Your preprocessing pipeline, exactly as used for training
transform_pipeline = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# 3. Load the trained PyTorch model architecture and then its state_dict
def load_model():
    # Define the model architecture
    model = efficientnet_b0(weights=None)
    
    # Replace the final layer to match the number of your classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    
    # Load the saved state_dict
    model.load_state_dict(torch.load('crop_disease_model.pth', map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode
    model.eval()
    return model

model = load_model()

# 4. Create the prediction function
def classify_crop_disease(image):
    if image is None:
        return None

    # Apply the transformations
    image_tensor = transform_pipeline(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        predictions = model(image_tensor)
        
    confidences = torch.nn.functional.softmax(predictions[0], dim=0)
    predicted_class_index = torch.argmax(confidences).item()
    predicted_class = class_names[predicted_class_index]
    
    # Sort confidences for a more user-friendly display
    confidences_dict = {
        class_names[i]: float(confidences[i]) for i in range(len(class_names))
    }
    
    return predicted_class, confidences_dict

# 5. Create the Gradio interface
interface = gr.Interface(
    fn=classify_crop_disease, 
    inputs=gr.Image(type="pil", label="Upload a leaf image"), 
    outputs=[
        gr.Textbox(label="Predicted Disease"), 
        gr.Label(label="Confidence Scores")
    ],
    title="AI-Powered Crop Disease Detection",
    description="""
    Upload an image of a crop leaf (cashew, cassava, maize, or tomato) to detect potential diseases. 
    The model is an EfficientNetB0 fine-tuned on a custom dataset.
    """,
    theme="huggingface",
    examples=[] # Add example image paths to this list
)

if __name__ == "__main__":
    interface.launch()
