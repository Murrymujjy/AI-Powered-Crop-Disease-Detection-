import streamlit as st

# Import the UI and logic for each page
from detector import show_detector_page
from chatbot import show_agribot_page
from recommendations import show_recommendations_page

# --- Global App Configuration and Custom CSS ---
st.set_page_config(
    page_title="AI Crop Disease Detector",
    page_icon="🌱",
    layout="wide"
)

# Custom HTML and CSS for a professional navigation bar
st.markdown("""
<style>
    .nav-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: #f0f2f6;
        padding: 10px 0;
        display: flex;
        justify-content: center;
        border-bottom: 2px solid #e0e2e5;
    }
    .nav-link {
        color: #4CAF50;
        padding: 10px 20px;
        text-decoration: none;
        font-weight: bold;
        font-size: 18px;
        margin: 0 10px;
        border-radius: 8px;
        transition: background-color 0.3s, color 0.3s;
        cursor: pointer;
    }
    .nav-link:hover {
        background-color: #4CAF50;
        color: white;
    }
    .nav-link.active {
        background-color: #4CAF50;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Navigation Logic ---
# Use st.radio in the sidebar for easy, clean navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Go to", ["Home", "Detector", "AgriBot", "Recommendations"])

# --- Page Router ---
if page_selection == "Home":
    st.title("Welcome to the AI-Powered Crop Disease Detection System 🌱")
    st.markdown("""
    This application uses a deep learning model to identify diseases in images of plant leaves.
    Navigate to the different sections using the sidebar on the left.
    """)
    st.info("""
    **Sections:**
    - **Detector:** Upload a leaf image to get a disease diagnosis.
    - **AgriBot:** Chat with an AI assistant for general plant care advice.
    - **Recommendations:** Get tailored recommendations for common crop issues.
    """)
elif page_selection == "Detector":
    show_detector_page()
elif page_selection == "AgriBot":
    show_agribot_page()
elif page_selection == "Recommendations":
    show_recommendations_page()
