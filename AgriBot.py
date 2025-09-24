import streamlit as st
import time

st.set_page_config(
    page_title="AgriBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AgriBot: Your Personal Plant Expert")
st.markdown("Ask me anything about crop diseases, plant care, or general agriculture advice!")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to generate bot response
def generate_bot_response(prompt):
    # A simple, rule-based response for now.
    # Replace this with an actual LLM API call (e.g., OpenAI, Gemini, Hugging Face)
    if any(keyword in prompt.lower() for keyword in ["hello", "hi", "greetings"]):
        return "Hello there! How can I assist you with your crops today?"
    elif any(keyword in prompt.lower() for keyword in ["tomato", "tomato leaf"]):
        return "Tomato plants are susceptible to diseases like late blight and leaf mold. Good ventilation and proper watering can help prevent them."
    elif any(keyword in prompt.lower() for keyword in ["cashew", "cashew tree"]):
        return "For cashew trees, common diseases include powdery mildew. Fungicides can be effective, and pruning can improve air circulation."
    elif any(keyword in prompt.lower() for keyword in ["maize", "corn"]):
        return "Maize plants can suffer from diseases like gray leaf spot. Using resistant varieties and crop rotation are key strategies."
    elif any(keyword in prompt.lower() for keyword in ["cassava", "yuca"]):
        return "Cassava mosaic disease is a major concern. The best way to manage it is to use disease-free planting material."
    elif any(keyword in prompt.lower() for keyword in ["thank you", "thanks"]):
        return "You're welcome! I'm here to help. Feel free to ask another question."
    else:
        return "I'm still learning! For now, I can provide information on diseases affecting tomatoes, maize, cashew, and cassava. Please try asking about one of those."

# Accept user input
if prompt := st.chat_input("Ask a question about plants..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            time.sleep(1) # Simulate a delay for a more natural feel
            response = generate_bot_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
