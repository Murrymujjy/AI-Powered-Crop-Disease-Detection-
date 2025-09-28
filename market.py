import pandas as pd
import streamlit as st
from datetime import datetime

# --- Configuration Constants ---
# NOTE: Replace 'default-app-id' with __app_id when integrating into the canvas environment
# appId = typeof __app_id !== 'undefined' ? __app_id : 'default-app-id';

# Placeholder exchange rates relative to 1 USD
# IMPORTANT: These are static placeholders. For live rates, you would need to integrate 
# an external financial API (e.g., ExchangeRate-API) here.
EXCHANGE_RATES = {
    "USD - US Dollar": {"rate": 1.0, "symbol": "$"},
    "NGN - Nigerian Naira": {"rate": 1500.00, "symbol": "₦"}, 
    "EUR - Euro": {"rate": 0.93, "symbol": "€"},
    "GBP - Pound Sterling": {"rate": 0.80, "symbol": "£"},
    "JPY - Japanese Yen": {"rate": 150.00, "symbol": "¥"},
}

def show_market_page():
    """
    Displays the market data, price simulation, and IoT metrics page.
    """
    st.title("🌾 Agricultural Market Insights & Farm Monitoring")
    st.markdown("---")

    # --- Real-Time Farm Metrics (IoT Placeholder) ---
    st.subheader("Real-Time Farm Metrics")
    st.info("The metrics below are placeholders. Integration with an IoT sensor platform API is required to display live farm data.")
    
    # Placeholder data for IoT metrics
    # In a real app, you would fetch this data from an API endpoint
    col_iot1, col_iot2, col_iot3 = st.columns(3)
    
    with col_iot1:
        st.metric("Soil Moisture", "45%", "-5% (Dry)", help="Data from Placeholder Soil Sensor 1")
    with col_iot2:
        st.metric("Air Temperature", "28°C", "+1.2°C", help="Data from Placeholder Weather Station")
    with col_iot3:
        st.metric("Humidity", "65%", "0%", help="Data from Placeholder Weather Station")

    st.markdown("---")


    # --- Currency Selection (Added) ---
    st.sidebar.header("Currency Settings")
    selected_currency_key = st.sidebar.selectbox(
        "Select Display Currency", 
        list(EXCHANGE_RATES.keys())
    )
    
    selected_currency = EXCHANGE_RATES[selected_currency_key]
    conversion_rate = selected_currency["rate"]
    currency_symbol = selected_currency["symbol"]
    
    # =========================================================================
    # REAL MARKET DATA PIPELINE INTEGRATION NEEDED HERE
    # =========================================================================

    try:
        # Base data in USD (This is what you'd get from your API)
        st.subheader(f"Current Commodity Prices ({selected_currency_key})")
        
        # Base prices are stored in USD
        data_usd = {
            'Commodity': ['Maize (Corn)', 'Wheat', 'Soybeans', 'Rice'],
            'Base USD Price': [4.50, 6.75, 12.80, 0.55], # Price in USD
            'Change (24h)': ['+1.2%', '-0.5%', '+0.1%', '+0.0%'],
            'Last Updated': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 4
        }
        
        df = pd.DataFrame(data_usd)
        
        # Dynamic Column Name
        new_col_name = f"Current Price ({currency_symbol}/bu or /kg)"
        
        # Apply conversion rate to the base USD prices
        df[new_col_name] = df['Base USD Price'] * conversion_rate
        
        # Remove the internal USD column and reorder the displayed columns
        df = df.drop(columns=['Base USD Price'])
        df = df[['Commodity', new_col_name, 'Change (24h)', 'Last Updated']]
        
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                # Dynamic format based on selected currency symbol
                new_col_name: st.column_config.NumberColumn(format=f"{currency_symbol}%.2f")
            }
        )

    except NameError:
        st.error("Market data could not be loaded. A required variable 'data' was not defined.")
    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")

    # =========================================================================
    # Price Simulation (Placeholder)
    # =========================================================================
    
    st.subheader("Price Simulation & Forecast")
    st.info("Market data pipeline integration is required to run real-time simulations.")
    
    st.markdown("""
        The price simulation model is a **placeholder** designed to demonstrate 
        the potential impact of crop disease on future market prices.
        
        Once connected to live market data (API/Scraping), this section will:
        1.  Fetch historical price volatility.
        2.  Allow users to input potential yield reduction (e.g., 10% loss due to disease).
        3.  Run a basic demand-supply simulation to forecast the new price point.
    """)
    
    # Simple simulation controls placeholder
    col1, col2 = st.columns(2)
    
    # We use the list of commodities from the data_usd dictionary for the selectbox
    with col1:
        st.selectbox("Select Crop for Simulation", data_usd['Commodity'])
    with col2:
        # This is where the crop disease detection result could feed in
        st.slider("Simulated Yield Reduction (%)", 0, 50, 10, help="Simulate the impact of a confirmed disease on overall crop yield.")

    if st.button("Run Price Forecast"):
        # Placeholder result converted to the selected currency
        usd_forecast_price = 5.10
        forecast_price = usd_forecast_price * conversion_rate
        
        st.success(f"Forecast Run. Simulated Price: {currency_symbol}{forecast_price:.2f}/bu (25% increase due to 10% yield reduction).")

# End of market.py
