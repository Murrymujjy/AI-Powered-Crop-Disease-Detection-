**AI-Powered Crop Disease Detection 🌿**
This repository contains the code for a crop disease detection application built using a PyTorch-based CNN model. The model is deployed as an interactive web application using Gradio, allowing users to upload images of plant leaves to identify potential diseases.

The project is designed to assist farmers and agricultural enthusiasts in quickly diagnosing common crop diseases.

📝 **Table of Contents**
Project Overview

Model & Dataset

Deployment

How to Use

Files in this Repository

Technologies

**🌱 Project Overview**
The application uses an EfficientNetB0 model, which was fine-tuned on a custom dataset of 22 different crop diseases and healthy plant conditions. The model can accurately classify diseases for four major crops: cashew, cassava, maize, and tomato.

The application interface, built with Gradio, provides a simple drag-and-drop mechanism for image uploads and displays the predicted disease along with confidence scores.

**📊 Model & Dataset**
**Model Architecture**: EfficientNetB0 (PyTorch)

**Number of Classes**: 22 (See class_names.json for the full list)

**Validation Accuracy**: 78%

**Dataset**: The model was trained on an undersampled dataset with a total of **18,260 images**. The dataset was split into training (**14,608 images**) and validation (**3,652 images**) sets.

**Preprocessing**: All images were resized to 224x224 pixels and normalized using mean and standard deviation values calculated from the training data.

**🚀 Deployment**
This application is designed for easy deployment on **Hugging Face Spaces**. The app.py script, along with the model weights and class names, can be directly uploaded to a **Gradio Space**, which handles the environment setup and web hosting.

**🧑‍🌾 How to Use**
Navigate to the deployed application link (e.g., on Hugging Face Spaces).

Click on the "Upload a leaf image" box or drag and drop an image file.

The application will automatically run the image through the model and display the predicted disease and a breakdown of confidence scores for each class.

**📂 Files in this Repository**
**app.py**: The main Python script that defines the Gradio interface and model inference logic.

**requirements.txt**: A list of all necessary Python libraries for the application to run.

**crop_disease_model.pth**: The trained model weights in PyTorch's .pth format.

**class_names.json**: A JSON file containing the list of 22 class names in the correct order.

**💻 Technologies**
Python

PyTorch

Gradio

Hugging Face Spaces

requirements.txt
This file is crucial for telling Gradio or any other environment what libraries to install.
