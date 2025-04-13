import requests
import json
import streamlit as st
from config import OLLAMA_BASE_URL

# --- Ollama API Interaction ---

def check_ollama_connection():
    """Checks connection to the Ollama server."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=5)
        response.raise_for_status()
        return True, "Connected"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error: Ollama server not found or not running."
    except requests.exceptions.Timeout:
        return False, "Connection Timeout: Server took too long to respond."
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {e}"

def get_available_models():
    """Fetches the list of available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        models_data = response.json()
        # Filter out potentially problematic model names if needed, or just return all
        return sorted([model['name'] for model in models_data.get('models', [])])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}", icon="🚨")
        return [] # Return empty list on error

def get_ollama_stream(model, messages, system_prompt=None, temperature=0.7, top_p=1.0, top_k=40, seed=None):
    """
    Sends a request to Ollama's /api/chat endpoint and streams the response.

    Args:
        model (str): The Ollama model name.
        messages (list): List of message dictionaries (role, content, optional images).
        system_prompt (str, optional): The system prompt.
        temperature (float): Model temperature.
        top_p (float): Model top_p.
        top_k (int): Model top_k.
        seed (int, optional): Model seed.

    Yields:
        tuple(str, bool): Chunks of the response content and a boolean indicating if the chunk is final.
    """
    api_url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
    }
    if system_prompt:
        payload["system"] = system_prompt
    if seed is not None:
        payload["options"]["seed"] = seed

    last_chunk = False
    try:
        with requests.post(api_url, json=payload, stream=True, timeout=180) as response:
            response.raise_for_status()
            buffer = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    try:
                        json_chunk = json.loads(decoded_line)
                        content_chunk = json_chunk.get('message', {}).get('content', '')
                        is_done = json_chunk.get('done', False)

                        if content_chunk:
                            yield content_chunk, False # Yield content chunk

                        if is_done:
                            last_chunk = True
                            # Optional: Yield metadata if needed from final chunk
                            # final_data = json_chunk.get('total_duration', None)
                            break # Exit loop once done

                    except json.JSONDecodeError:
                        st.warning(f"Received non-JSON line, skipping: {decoded_line}")
                        print(f"Received non-JSON line: {decoded_line}") # Debugging
                    except Exception as e_inner:
                        st.error(f"Error processing stream chunk: {e_inner}", icon="🔥")
                        print(f"Error processing chunk: {e_inner}\nChunk: {decoded_line}")
                        break
            yield "", last_chunk # Ensure a final yield indicating done status

    except requests.exceptions.Timeout:
        st.error("Error: Ollama request timed out.", icon="⏳")
        yield "Error: Request Timed Out.", True
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Ollama at {OLLAMA_BASE_URL}. Is it running?", icon="🔌")
        yield f"Error: Connection Failed ({OLLAMA_BASE_URL}).", True
    except requests.exceptions.RequestException as e:
        error_message = f"Ollama API request failed: {e}"
        # Attempt to parse error details from Ollama if available
        try:
            error_details = response.json() # This might fail if response isn't valid JSON
            error_message += f"\nDetails: {error_details.get('error', 'No details provided')}"
        except:
            pass # Ignore if error response is not JSON or response object isn't available
        st.error(error_message, icon="🔥")
        yield f"Error: API Request Failed.", True # Yield generic error for UI
    except Exception as e:
        st.error(f"An unexpected error occurred during streaming: {e}", icon="💥")
        yield f"Error: Unexpected Streaming Error.", True

def generate_text_ollama(model, prompt, system_prompt=None, temperature=0.2,):
    """Generates a single text completion using Ollama's /api/generate."""
    api_url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False, # We want the full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 100 # Limit prediction length (adjust as needed)
        }
    }
    if system_prompt:
        payload["system"] = system_prompt

    try:
        response = requests.post(api_url, json=payload, timeout=45) # Adjust timeout
        response.raise_for_status()
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        print(f"Error during Ollama generate call: {e}") # Log for debugging
        return None # Indicate failure