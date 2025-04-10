import streamlit as st
import requests
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime
import time
import base64

# Page config
st.set_page_config(
    page_title="Ollama Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model" not in st.session_state:
    st.session_state.model = "llama2"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_chat_name" not in st.session_state:
    st.session_state.current_chat_name = "New Chat"

if "system_instruction" not in st.session_state:
    st.session_state.system_instruction = "You are a helpful AI assistant. Please provide clear and concise responses."

# Function to stop current model
def stop_current_model():
    try:
        # Try to stop the model
        response = requests.post("http://localhost:11434/api/stop")
        
        # Try to kill any running processes
        try:
            requests.post("http://localhost:11434/api/kill")
        except:
            pass  # Ignore if kill endpoint is not available
            
    except Exception as e:
        pass  # Silently handle any errors

# Function to check if model is running
def is_model_running():
    try:
        response = requests.get("http://localhost:11434/api/status")
        if response.status_code == 200:
            status = response.json()
            return status.get("status", "") == "running"
        return False
    except:
        return False

# Function to fetch available models
def fetch_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return ["llama2", "mistral", "codellama"]  # Default models if API fails
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return ["llama2", "mistral", "codellama"]

# Function to save chat history
def save_chat_history(chat_name=None):
    if not os.path.exists("chat_history"):
        os.makedirs("chat_history")
    
    if chat_name is None:
        chat_name = st.session_state.current_chat_name
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history/{chat_name}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump({
            "chat_name": chat_name,
            "model": st.session_state.model,
            "messages": st.session_state.messages,
            "timestamp": timestamp
        }, f, indent=2)
    
    return filename

# Function to load chat history
def load_chat_history():
    if not os.path.exists("chat_history"):
        return []
    
    chat_files = [f for f in os.listdir("chat_history") if f.endswith(".json")]
    return chat_files

# Function to get chat details
def get_chat_details(filename):
    with open(f"chat_history/{filename}", "r") as f:
        chat_data = json.load(f)
        return {
            "name": chat_data.get("chat_name", "Unnamed Chat"),
            "model": chat_data.get("model", "Unknown Model"),
            "timestamp": chat_data.get("timestamp", ""),
            "message_count": len(chat_data.get("messages", []))
        }

# Function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to handle file upload
def handle_file_upload(uploaded_file):
    if uploaded_file is None:
        return None, None
    
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        file_path = tmp_file.name
    
    # Determine file type and handle accordingly
    if uploaded_file.type.startswith("image/"):
        try:
            # For images, encode to base64
            image_base64 = encode_image_to_base64(file_path)
            return file_path, {"type": "image", "data": image_base64}
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None
    elif uploaded_file.type == "application/pdf":
        try:
            # For PDFs, we'll extract text content
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(file_path)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            return file_path, {"type": "pdf", "data": text_content}
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None, None
    else:
        st.warning(f"Unsupported file type: {uploaded_file.type}")
        return None, None

# Sidebar
with st.sidebar:
    st.title("Ollama Chat Settings")
    
    # System instruction input
    system_instruction = st.text_area(
        "System Instruction",
        value=st.session_state.system_instruction,
        help="Set the behavior and personality of the AI assistant"
    )
    if system_instruction != st.session_state.system_instruction:
        st.session_state.system_instruction = system_instruction
        st.success("System instruction updated!")
    
    # New Chat button
    if st.button("New Chat"):
        stop_current_model()  # Stop any running model
        st.session_state.messages = []
        st.session_state.current_chat_name = "New Chat"
        st.success("Started a new chat!")
    
    # Chat name input
    chat_name = st.text_input(
        "Chat Name",
        value=st.session_state.current_chat_name,
        help="Rename the current chat"
    )
    if chat_name != st.session_state.current_chat_name:
        st.session_state.current_chat_name = chat_name
        st.success(f"Chat renamed to: {chat_name}")
    
    # Model selection
    available_models = fetch_available_models()
    new_model = st.selectbox(
        "Select Model",
        available_models,
        index=available_models.index(st.session_state.model) if st.session_state.model in available_models else 0,
        help="Select a model to use for chat"
    )
    
    # Stop current model and switch to new one
    if new_model != st.session_state.model:
        # First stop the current model
        stop_current_model()
        
        # Wait a moment to ensure the model is stopped
        time.sleep(2)  # Give the model time to stop
        
        # Verify the model is stopped and try again if needed
        if is_model_running():
            time.sleep(2)  # Wait a bit longer
            stop_current_model()  # Try stopping again
        
        # Now switch to the new model
        st.session_state.model = new_model
        st.session_state.messages = []  # Clear messages when switching models
        st.success(f"Switched to {new_model} model!")
    
    # Temperature control
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.7, 0.1,
        help="Higher values make the output more random, lower values make it more deterministic"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Upload a PDF or image file to analyze"
    )
    
    # Chat history management
    st.subheader("Chat History")
    chat_files = load_chat_history()
    
    if chat_files:
        # Display chat history with details
        for filename in chat_files:
            chat_details = get_chat_details(filename)
            with st.expander(f"{chat_details['name']} ({chat_details['message_count']} messages)"):
                st.write(f"Model: {chat_details['model']}")
                st.write(f"Created: {chat_details['timestamp']}")
                if st.button("Load", key=f"load_{filename}"):
                    with open(f"chat_history/{filename}", "r") as f:
                        chat_data = json.load(f)
                        st.session_state.messages = chat_data["messages"]
                        st.session_state.model = chat_data["model"]
                        st.session_state.current_chat_name = chat_data.get("chat_name", "Unnamed Chat")
                        st.success("Chat loaded successfully!")
                if st.button("Delete", key=f"delete_{filename}"):
                    os.remove(f"chat_history/{filename}")
                    st.success("Chat deleted successfully!")
                    st.rerun()
    
    # Save chat button
    if st.button("Save Current Chat"):
        filename = save_chat_history()
        st.success(f"Chat saved as: {st.session_state.current_chat_name}")

# Main chat interface
st.title(f"Ollama Chat - {st.session_state.current_chat_name}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    message_content = {"role": "user", "content": prompt}
    
    # Handle file upload if present
    file_path = None
    file_data = None
    if uploaded_file:
        file_path, file_data = handle_file_upload(uploaded_file)
        if file_path:
            message_content["file_path"] = file_path
            if file_data["type"] == "image":
                message_content["image"] = uploaded_file
            elif file_data["type"] == "pdf":
                message_content["pdf_content"] = file_data["data"]
    
    st.session_state.messages.append(message_content)
    with st.chat_message("user"):
        st.markdown(prompt)
        if uploaded_file and file_data:
            if file_data["type"] == "image":
                st.image(uploaded_file)
            elif file_data["type"] == "pdf":
                st.info("PDF content has been processed and will be used in the conversation.")

    # Generate response
    with st.chat_message("assistant"):
        try:
            # Prepare the request payload
            payload = {
                "model": st.session_state.model,
                "prompt": f"System: {st.session_state.system_instruction}\n\nUser: {prompt}",
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            # If there's a file, add it to the context
            if file_data:
                if file_data["type"] == "image":
                    payload["images"] = [file_data["data"]]
                elif file_data["type"] == "pdf":
                    # Add PDF content to the prompt
                    payload["prompt"] = f"System: {st.session_state.system_instruction}\n\nPDF Content:\n{file_data['data']}\n\nUser: {prompt}"
            
            # Make the API request
            response = requests.post("http://localhost:11434/api/generate", json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_message = response_data.get("response", "")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
                st.markdown(assistant_message)
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary files
            if file_path:
                try:
                    os.unlink(file_path)
                except:
                    pass

# Add clear chat button at the bottom
if st.button("Clear Chat", key="clear_chat_bottom"):
    stop_current_model()  # Stop any running model
    st.session_state.messages = []
    st.success("Chat cleared!")
    st.rerun() 