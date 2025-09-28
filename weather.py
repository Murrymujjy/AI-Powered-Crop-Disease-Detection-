# weather.py
import streamlit as st
import requests
import pandas as pd

# Placeholder for a real API Key and Base URL
WEATHER_API_KEY = "e81c6d430db790e860eb39cee71b6e07"  
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

def fetch_weather(city_name):
    """Fetches current weather data for a given city."""
    if WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        st.warning("Please get a valid API key (e.g., from OpenWeatherMap) and replace the placeholder in weather.py.")
        return None

    try:
        params = {
            'q': city_name,
            'appid': WEATHER_API_KEY,
            'units': 'metric'  # Get temperature in Celsius
        }
        response = requests.get(WEATHER_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data. Check city name or API key: {e}")
        return None

def show_weather_page():
    st.title("☀️ Real-Time Weather Forecast")
    st.markdown("Enter your location to get real-time weather conditions and advisory.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        city_name = st.text_input("Enter your Farm Location (e.g., 'Lagos, NG')", "Lagos, NG")
        if st.button("Get Weather Update", use_container_width=True):
            if city_name:
                weather_data = fetch_weather(city_name)
                if weather_data:
                    display_weather(weather_data)

def display_weather(data):
    """Displays fetched weather data in a professional card format."""
    
    main_weather = data['weather'][0]['main']
    temp = data['main']['temp']
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    city = data['name']
    country = data['sys']['country']

    st.subheader(f"Current Conditions in {city}, {country}")
    
    col_temp, col_wind, col_humidity = st.columns(3)

    with col_temp:
        st.metric(label="Temperature", value=f"{temp:.1f} °C", delta=f"{main_weather}")
    
    with col_wind:
        st.metric(label="Wind Speed", value=f"{wind_speed} m/s")

    with col_humidity:
        st.metric(label="Humidity", value=f"{humidity} %")

    st.markdown("---")
    
    # Simple Advisory based on weather
    if temp > 30 and humidity > 80:
        st.warning("⚠️ Heat and high humidity are favorable for fungal diseases. Monitor crops closely.")
    elif temp < 10:
        st.info("❄️ Cold weather advisory: Protect sensitive crops from frost.")
    else:
        st.success("✅ Favorable weather conditions for most crops. Proceed with routine care.")
