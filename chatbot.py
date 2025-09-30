import streamlit as st
import requests
import json
import time

# --- Hugging Face API Configuration ---

HUGGING_FACE_API_KEY = "YOUR_HF_TOKEN"
# Using a common chat-optimized model as a placeholder.
# REPLACE with the actual model ID if you have a specific one (e.g., Llama-3-8B-Instruct).
MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
API_URL = MODEL_ENDPOINT

# System instruction to define the AgriBot's persona and rules
SYSTEM_PROMPT = (
    "You are AgriBot, a world-class agricultural consultant and helpful AI assistant. "
    "Your goal is to provide accurate, specific, and actionable advice to farmers based "
    "on scientific knowledge and common best practices. Keep answers concise but comprehensive. "
    "If the user asks about real-time farm data (like soil moisture or price forecasts), "
    "remind them that they should refer to the main dashboard metrics, as you cannot access live data."
)

def format_chat_history(chat_history):
    """Formats the Streamlit chat history into a simple, single prompt string
    for the Hugging Face model to maintain context."""
    
    # Start with the system prompt
    prompt_string = f"[SYSTEM]: {SYSTEM_PROMPT}\n\n"
    
    # Add previous turns
    for message in chat_history:
        role = "User" if message["role"] == "user" else "AgriBot"
        # Sanitize content to prevent internal formatting issues
        content = message["content"].replace("\n", " ")
        prompt_string += f"[{role}]: {content}\n"
        
    # Append the final turn marker for the model to continue
    prompt_string += "[AgriBot]:"
    return prompt_string

def generate_agri_response(prompt, chat_history):
    """
    Calls the Hugging Face Inference API to get a response using exponential backoff.
    """
    
    # Append the current user prompt to the history for formatting
    temp_history = chat_history + [{"role": "user", "content": prompt}]
    
    # Format the entire history for the model
    formatted_prompt = format_chat_history(temp_history)

    # Hugging Face Inference API payload structure
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.5,
            "return_full_text": False, # Important: we only want the model's generated response
        }
    }

    # API authentication using the provided key
    headers = {
        'Authorization': f'Bearer {HUGGING_FACE_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # Implement exponential backoff for API robustness
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            if not isinstance(result, list) or not result or 'generated_text' not in result[0]:
                 return "AgriBot: Sorry, I received an empty or unexpected response from the AI model."

            # Hugging Face usually returns a list of one result with a 'generated_text' key
            text = result[0]['generated_text'].strip()
            
            # Remove any leading AgriBot tag if the model accidentally generates it
            if text.startswith("[AgriBot]:"):
                text = text[len("[AgriBot]:"):].strip()
                
            return text

        except requests.exceptions.HTTPError as e:
            # Handle model loading (503) or rate limits (429) specific to HF
            if response.status_code in [429, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # print(f"API Rate limit hit or temporary error. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 401:
                 return "AgriBot: Authentication failed. Please check your Hugging Face API key."
            else:
                return f"AgriBot: An API error occurred (Status {response.status_code}). Please check the model endpoint. Error: {e}"
        except requests.exceptions.RequestException as e:
            return f"AgriBot: A network or connection error occurred: {e}"
        except json.JSONDecodeError:
            return "AgriBot: Error processing the AI response (JSON decode error)."
            
    return "AgriBot: Maximum API retry attempts reached. Please check your connection."


def show_agribot_page():
    """
    Sets up the Streamlit interface for the AgriBot chatbot.
    """
    st.title("👨‍🌾 AgriBot: Your AI Consultant (Hugging Face Model)")
    st.markdown("---")
    st.info("Ask me anything about crop diseases, farming techniques, or pest control! Note: I cannot access real-time web data or live metrics from the dashboard.")

    # Initialize chat history
    if "agri_chat_history" not in st.session_state:
        st.session_state.agri_chat_history = []
        # Add an initial welcome message from the bot
        st.session_state.agri_chat_history.append({
            "role": "model", 
            "content": "Hello! I am AgriBot. I am running on a Hugging Face model. How can I assist you with your farming decisions today?"
        })

    # Display chat messages from history
    for message in st.session_state.agri_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask AgriBot a question about agriculture..."):
        # Display user message
        st.session_state.agri_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display model response
        with st.chat_message("model"):
            with st.spinner("AgriBot is consulting the knowledge base..."):
                full_response = generate_agri_response(prompt, st.session_state.agri_chat_history)
            
            st.markdown(full_response)
        
        # Add model response to history
        # We only store the final response text, not the temporary formatted prompt
        st.session_state.agri_chat_history.append({"role": "model", "content": full_response})
