import streamlit as st
from streamlit_option_menu import option_menu

import detector
import weather
import chatbot
import recommendations
import market  # Placeholder module
import soil   # Placeholder module

# --- 2. Custom Styling (CSS) ---
def set_custom_style():
    st.markdown("""
    <style>
    /* Light blue background for the main content */
    .main {
        background-color: #e6f0ff;
        padding: 20px;
        border-radius: 10px;
    }
    /* Custom button styling */
    div.stButton > button {
        color: white;
        background-color: #1f77b4; /* Primary Blue */
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.2s;
    }
    div.stButton > button:hover {
        background-color: #1a5c8e; /* Darker blue on hover */
    }
    /* Ensure Streamlit's primary element color matches the button */
    :root {
        --primary-color: #1f77b4; 
    }
    </style>
    """, unsafe_allow_html=True)

set_custom_style()

# --- 3. Session Setup ---
# Define all navigation labels upfront for clarity
NAV_PAGES = {
    "🏠 Home": "Home",
    "🌿 Disease Detector": "Detector",
    "☀️ Weather Forecast": "Weather",
    "🤖 AgriBot Chat": "Chatbot",
    "💡 Recommendations": "Recommendations",
    "📈 Market Prices": "Market",
    "🔬 Soil Monitoring": "Soil",
}

if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "🏠 Home"

# --- 4. Sidebar Navigation using streamlit_option_menu ---
with st.sidebar:
    st.title("Agricultural Dashboard")
    
    # Use the keys from the NAV_PAGES dictionary for the menu options
    selected_key = option_menu(
        "Navigation",
        list(NAV_PAGES.keys()),
        icons=["house", "leaf", "sun", "robot", "lightbulb", "graph-up", "microscope"],
        default_index=0 # Start on the Home page
    )
    # Update the session state to the simpler, non-emoji name
    st.session_state.selected_nav = NAV_PAGES[selected_key]


# --- 5. Page Router (Calling the imported module functions) ---

if st.session_state.selected_nav == "Home":
    # 1. Centered Title (using HTML for styling)
    st.markdown("<h1 style='text-align: center;'>🌾 AI-Powered Agricultural Intelligence Platform</h1>", unsafe_allow_html=True)
    
    # 2. Centered Introductory Text
    st.markdown(
        """
        <div style='text-align: center; font-size:1.1rem; margin-bottom: 20px;'>
        Welcome to the **AI Agricultural Dashboard**, your intelligent tool for maximizing crop health and profitability.<br>
        Built with deep learning and real-time data, this platform supports **farmers** in making data-driven decisions.
        </div>
        <hr>
        """, unsafe_allow_html=True
    )
    
    # 3. Call-to-Action for Navigation
    st.warning("👉 **Discover More:** Use the **Navigation Menu** in the sidebar to explore advanced features like the Disease Detector, Weather Forecast, and AgriBot Chat!")


    # Creating a Home Page grid with links/buttons to other features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌿 Detector")
        st.markdown("AI-driven disease identification.")
        if st.button("Go to Detector"):
            st.session_state.selected_nav = "Detector"
            st.rerun()

    with col2:
        st.markdown("### ☀️ Weather")
        st.markdown("Real-time forecast for planting decisions.")
        if st.button("Go to Weather"):
            st.session_state.selected_nav = "Weather"
            st.rerun()
            
    with col3:
        st.markdown("### 🤖 AgriBot")
        st.markdown("Your personal crop advisory chatbot.")
        if st.button("Go to Chatbot"):
            st.session_state.selected_nav = "Chatbot"
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='text-align: center;'>📌 Made with ❤️ by <strong>AI Farm Solutions</strong></div>", unsafe_allow_html=True)


# --- ROUTING to the Imported Modules ---

elif st.session_state.selected_nav == "Detector":
    detector.show_detector_page()

elif st.session_state.selected_nav == "Weather":
    weather.show_weather_page()

elif st.session_state.selected_nav == "Chatbot":
    chatbot.show_agribot_page()

elif st.session_state.selected_nav == "Recommendations":
    recommendations.show_recommendations_page()

elif st.session_state.selected_nav == "Market":
    # Call the function from the imported market module
    market.show_market_page()

elif st.session_state.selected_nav == "Soil":
    # Call the function from the imported soil module
    soil.show_soil_page()
