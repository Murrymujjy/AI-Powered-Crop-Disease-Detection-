import streamlit as st
import time

def show_agribot_page():
    st.title("🤖 AgriBot: Your Personal Plant Expert")
    st.markdown("Ask me anything about crop diseases, plant care, or general agriculture advice!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    def generate_bot_response(prompt):
        if any(keyword in prompt.lower() for keyword in ["hello", "hi", "greetings"]):
            return "Hello there! How can I assist you with your crops today?"
        elif any(keyword in prompt.lower() for keyword in ["tomato", "tomato leaf"]):
            return "Tomato plants are susceptible to diseases like late blight and leaf mold. Good ventilation and proper watering can help prevent them."
        else:
            return "I'm still learning! For now, I can provide information on diseases affecting tomatoes, maize, cashew, and cassava. Please try asking about one of those."

    if prompt := st.chat_input("Ask a question about plants..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                time.sleep(1)
                response = generate_bot_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
