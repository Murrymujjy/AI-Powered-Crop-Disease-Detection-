import streamlit as st

# Import the UI and logic for all features
from detector import show_detector_page
from chatbot import show_agribot_page
from recommendations import show_recommendations_page
from weather import show_weather_page  
from market import show_market_page  
from soil import show_soil_page      


# --- Global App Configuration and Custom CSS for Professional Look ---
st.set_page_config(
    page_title="AI Agricultural Dashboard",
    page_icon="🌱",
    layout="wide"
)

# Custom, flat-design CSS for a professional, modern dashboard
st.markdown("""
<style>
    /* Main body background and text color */
    body {
        color: #333;
        background-color: #f7f9fb;
    }
    /* Fixed, clean navigation bar */
    .nav-bar {
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: white; /* Clean white background */
        padding: 10px 0;
        display: flex;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Subtle shadow for depth */
        border-radius: 10px;
    }
    .nav-link {
        color: #007bff; /* Primary blue color */
        padding: 10px 20px;
        text-decoration: none;
        font-weight: 600;
        font-size: 16px;
        margin: 0 5px;
        border-radius: 6px;
        transition: background-color 0.2s, color 0.2s;
        cursor: pointer;
    }
    .nav-link:hover {
        background-color: #f0f8ff; /* Light hover effect */
        color: #0056b3;
    }
    /* Streamlit sidebar styling (to match the professional look) */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* Custom Card Styling for Metrics/Sections */
    div.st-emotion-cache-1r6i7z3 {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Stronger shadow for "pop" */
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
    }
    /* Main title styling */
    h1 {
        color: #1e8449; /* Green color for agriculture focus */
        font-size: 2.5rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Navigation Logic ---
st.sidebar.title("Dashboard Menu")
# Added new features to the menu
menu_items = ["Home", "Detector", "Weather", "AgriBot", "Recommendations", "Market Prices", "Soil Monitoring"]
page_selection = st.sidebar.radio("Navigate", menu_items)


# --- Page Router ---
if page_selection == "Home":
    st.title("AI-Powered Agricultural Intelligence Platform 🌱")
    st.markdown("### Your solution for precision farming and maximizing yield.")
    st.markdown("""
    Welcome to your unified agricultural dashboard. Use the sidebar to access advanced features:
    
    - **Detector:** AI-driven disease identification.
    - **Weather:** Real-time weather data and advisory.
    - **AgriBot:** Your AI chatbot for quick questions.
    - **Recommendations:** Tailored crop care advice.
    - **Market Prices:** Track crop value across regions (requires data integration).
    - **Soil Monitoring:** Real-time sensor data display (requires IoT integration).
    """)

elif page_selection == "Detector":
    show_detector_page()

elif page_selection == "Weather":
    show_weather_page()

elif page_selection == "AgriBot":
    show_agribot_page()

elif page_selection == "Recommendations":
    show_recommendations_page()

elif page_selection == "Market Prices":
    # show_market_page() # Uncomment when ready
    st.title("📈 Crop Market Prices")
    st.error("Feature requires data integration (uncomment 'show_market_page()' in app.py when ready).")

elif page_selection == "Soil Monitoring":
    # show_soil_page() # Uncomment when ready
    st.title("🔬 Soil Parameter Monitoring")
    st.error("Feature requires IoT sensor API integration (uncomment 'show_soil_page()' in app.py when ready).")
