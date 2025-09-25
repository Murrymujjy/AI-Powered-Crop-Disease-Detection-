import streamlit as st
import pandas as pd

def show_recommendations_page():
    st.title("💡 Plant Care & Recommendations")
    st.markdown("Get tailored advice and recommendations for your crops based on common issues.")
    
    data = {
        'Crop': ['Tomato', 'Tomato', 'Maize', 'Maize', 'Cassava', 'Cassava', 'Cashew', 'Cashew'],
        'Common Issue': ['Late Blight', 'Nutrient Deficiency', 'Gray Leaf Spot', 'Pest Infestation', 'Mosaic Disease', 'Root Rot', 'Powdery Mildew', 'Leaf Spot'],
        'Recommendation': [
            'Apply fungicides immediately. Ensure good air circulation and avoid overhead watering.',
            'Use a balanced NPK fertilizer and consider adding compost to the soil.',
            'Rotate crops to prevent buildup. Use a foliar fungicide and plant resistant varieties.',
            'Apply an insecticide. Introduce beneficial insects like ladybugs.',
            'Use disease-free planting material. Uproot and burn infected plants to prevent spread.',
            'Ensure proper soil drainage. Avoid overwatering and plant on raised beds.',
            'Apply a sulfur-based fungicide. Prune infected branches to promote air flow.',
            'Use copper-based fungicides. Clear debris around the plant to reduce disease sources.'
        ]
    }
    recommendations_df = pd.DataFrame(data)

    selected_crop = st.selectbox("Select your crop:", recommendations_df['Crop'].unique())
    
    if selected_crop:
        filtered_df = recommendations_df[recommendations_df['Crop'] == selected_crop]
        st.subheader(f"Common issues for {selected_crop}:")
        selected_issue = st.selectbox("Select an issue:", filtered_df['Common Issue'].unique())

        if selected_issue:
            recommendation = filtered_df[filtered_df['Common Issue'] == selected_issue]['Recommendation'].iloc[0]
            st.header(f"Recommendation for {selected_crop} - {selected_issue}")
            st.info(recommendation)

    st.markdown("---")
    st.subheader("Explore all recommendations")
    st.dataframe(recommendations_df, use_container_width=True)
