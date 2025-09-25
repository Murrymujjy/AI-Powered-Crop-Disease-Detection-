import streamlit as st
from PIL import Image
from model import classify_crop_disease

def show_detector_page():
    st.title("🌿 Crop Disease Detection")
    st.write("Upload an image of a crop leaf to detect potential diseases.")
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        
        with st.spinner('Analyzing the image...'):
            predictions = classify_crop_disease(image)

        st.subheader("Prediction Results:")
        
        top_class, top_confidence = predictions[0]
        st.success(f"**Predicted Class:** {top_class} (Confidence: {top_confidence:.2f})")

        st.markdown("---")
        
        st.subheader("All Confidence Scores:")
        for class_name, confidence in predictions:
            st.write(f"- {class_name}: {confidence:.4f}")
