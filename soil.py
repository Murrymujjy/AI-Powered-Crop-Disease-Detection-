import streamlit as st

def show_soil_page():
    st.title("🔬 Soil Parameter Monitoring")
    st.markdown("Real-time data on soil pH, moisture, and nutrients for precision farming.")
    st.info("Requires integration with an IoT sensor platform API to display real-time metrics.")
    
    st.subheader("Real-Time Data (Placeholder)")
    # Placeholder for actual real-time sensor data display
    st.columns(3)[0].metric("Soil Moisture", "65%", "Optimal")
    st.columns(3)[1].metric("Soil pH", "6.2", "Slightly Acidic")
    st.columns(3)[2].metric("Temperature", "25 °C", "Normal")
