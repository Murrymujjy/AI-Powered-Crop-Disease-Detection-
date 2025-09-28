import requests
import os
import json
from time import sleep

# ==============================================================================
# CONFIGURATION

# ==============================================================================
HF_TOKEN = "YOUR_HF_TOKEN"
# We use a powerful but relatively small instruction model for fast inference.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Construct the full API URL
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query_model(prompt):
    """
    Sends a text generation request to the Hugging Face Inference API.

    The parameters include 'wait_for_model': True, which is CRITICAL for the
    free serverless tier. This ensures the function waits for the model to load
    if it was sleeping due to inactivity, preventing a '503 Service Unavailable' error.
    """
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,  # Limit response length for faster results
            "temperature": 0.7,
            "return_full_text": False # Only return the generated response, not the prompt
        },
        # Crucial for the free tier to wait if the model is loading
        "options": {
            "wait_for_model": True
        }
    }

    try:
        print("Sending request to Hugging Face...")
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        if isinstance(data, list) and data and 'generated_text' in data[0]:
            return data[0]['generated_text']
        elif 'error' in data:
            return f"API Error: {data.get('error')}"
        else:
            return "Unexpected response format from API."

    except requests.exceptions.HTTPError as errh:
        # Specific handling for the 503 error, which often means the model is loading
        if response.status_code == 503:
            print("\nModel is currently loading (503 error). Please wait a moment and try again.")
            # On the free tier, waiting and retrying is often necessary.
            return "Please wait 30 seconds and try again."
        return f"HTTP Error: {errh}"
    except requests.exceptions.ConnectionError as errc:
        return f"Connection Error: {errc}"
    except requests.exceptions.Timeout as errt:
        return f"Timeout Error: {errt}"
    except requests.exceptions.RequestException as err:
        return f"An unexpected error occurred: {err}"


# --- Main Chat Loop ---
def run_chatbot():
    """Simple command-line loop for the chatbot."""
    if HF_TOKEN == "YOUR_HF_TOKEN":
        print("ERROR: Please update the HF_TOKEN variable in the script with your actual token.")
        return

    print("--- Hugging Face Inference API Chatbot ---")
    print(f"Model: {MODEL_ID}")
    print("Type 'quit' or 'exit' to end the chat.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        response_text = query_model(user_input)
        print(f"AI: {response_text}")

if __name__ == "__main__":
    run_chatbot()
