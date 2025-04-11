# appv3.py

import streamlit as st
import requests
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime
import time
import base64
import html
import uuid # For unique chat IDs
import traceback # For error details

# --- Constants ---
HISTORY_DIR = Path("ollama_chat_history") # Directory to store chat files
LAST_CHAT_ID_FILE = HISTORY_DIR / ".last_ollama_chat_id" # Hidden file for last chat ID
DEFAULT_OLLAMA_MODEL = "llama2" # Default model if API fetch fails
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Please provide clear and concise responses."
OLLAMA_BASE_URL = "http://localhost:11434" # Make base URL configurable if needed

# Ensure history directory exists
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# --- Session State Initialization --- (Using setdefault for safety)
def initialize_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("current_chat_id", str(uuid.uuid4()))
    st.session_state.setdefault("current_chat_name", "New Chat")
    st.session_state.setdefault("model", DEFAULT_OLLAMA_MODEL)
    st.session_state.setdefault("system_instruction", DEFAULT_SYSTEM_PROMPT)
    st.session_state.setdefault("temperature", 0.7)
    st.session_state.setdefault("chat_history_list", [])
    st.session_state.setdefault("renaming_chat_id", None)
    st.session_state.setdefault("autoload_last_chat", True)
    st.session_state.setdefault("app_just_started", True)
    st.session_state.setdefault("loaded_on_start", False)
    st.session_state.setdefault("ollama_reachable", None)
    st.session_state.setdefault("pending_file_parts", [])
    st.session_state.setdefault("last_uploaded_file_names", set())
    st.session_state.setdefault("response_count", 0)
    st.session_state.setdefault("max_tokens", 2048) # Consider matching Ollama defaults if needed
    st.session_state.setdefault("top_p", 0.95) # Add missing defaults if used


# --- Ollama Interaction Functions ---
def check_ollama_connection():
    """Checks if the Ollama server is reachable."""
    if st.session_state.ollama_reachable is None:
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=2)
            st.session_state.ollama_reachable = response.status_code == 200
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            st.session_state.ollama_reachable = False
        except Exception as e:
            print(f"Unexpected error checking Ollama connection: {e}")
            st.session_state.ollama_reachable = False
    return st.session_state.ollama_reachable

def fetch_available_models():
    """Fetches available models from Ollama API."""
    if not check_ollama_connection(): return [DEFAULT_OLLAMA_MODEL]
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models if "name" in model]
        return model_names if model_names else [DEFAULT_OLLAMA_MODEL]
    except requests.exceptions.RequestException as e: st.error(f"Error fetching models: {e}. Using default."); return [DEFAULT_OLLAMA_MODEL]
    except json.JSONDecodeError: st.error("Error decoding model list from Ollama. Using default."); return [DEFAULT_OLLAMA_MODEL]

def stop_ollama_model(model_name):
     """Attempts to stop/unload a specific model."""
     if not check_ollama_connection(): return
     try:
         print(f"Requesting unload for model: {model_name} (via empty generate - experimental)")
         requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={"model": model_name, "keep_alive": "0s"}, timeout=5)
     except Exception as e: print(f"Could not explicitly unload model {model_name}: {e}")


# --- History Management Functions ---
def get_last_chat_id():
    if LAST_CHAT_ID_FILE.exists():
        try: return LAST_CHAT_ID_FILE.read_text().strip()
        except Exception: return None
    return None

def set_last_chat_id(chat_id):
    try: LAST_CHAT_ID_FILE.write_text(str(chat_id))
    except Exception as e: print(f"Warning: Could not write last chat ID: {e}")

def get_chat_filepath(chat_id):
    return HISTORY_DIR / f"ollama_chat_{chat_id}.json"

def create_save_data():
    """Packages current chat state for saving."""
    messages_to_save = []
    for msg in st.session_state.get("messages", []):
        messages_to_save.append({ "role": msg.get("role"), "content": msg.get("content", "") })
    return {
        "chat_id": st.session_state.get("current_chat_id", str(uuid.uuid4())),
        "chat_name": st.session_state.get("current_chat_name", "New Chat"),
        "model": st.session_state.get("model", DEFAULT_OLLAMA_MODEL),
        "system_instruction": st.session_state.get("system_instruction", DEFAULT_SYSTEM_PROMPT),
        "temperature": st.session_state.get("temperature", 0.7),
        "messages": messages_to_save,
        "saved_at": datetime.now().isoformat()
    }

def save_current_chat_to_file(show_toast=False):
    """Saves the current chat state to its JSON file."""
    chat_id = st.session_state.get("current_chat_id")
    if not chat_id: print("Warning: Save attempt without chat ID."); return
    filepath = get_chat_filepath(chat_id)
    try:
        data_to_save = create_save_data()
        with open(filepath, "w", encoding="utf-8") as f: json.dump(data_to_save, f, indent=2)
        set_last_chat_id(chat_id)
        if show_toast: st.toast(f"Chat '{st.session_state.current_chat_name}' saved.", icon="💾")
    except Exception as e: st.error(f"Error auto-saving chat {chat_id}: {e}", icon="💾")

def save_specific_chat_data(chat_id, chat_data):
    """Saves provided chat data dictionary."""
    if not chat_id or not chat_data: return False
    filepath = get_chat_filepath(chat_id)
    try:
        chat_data["saved_at"] = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f: json.dump(chat_data, f, indent=2)
        if chat_id == st.session_state.get("current_chat_id"): set_last_chat_id(chat_id)
        return True
    except Exception as e: st.error(f"Error saving specific chat data for {chat_id}: {e}", icon="💾"); return False

def load_chat_data(chat_id):
    """Loads raw chat data dictionary from file."""
    filepath = get_chat_filepath(chat_id)
    if not filepath.exists(): return None
    try:
        with open(filepath, "r", encoding="utf-8") as f: return json.load(f)
    except Exception as e: st.error(f"Error loading chat file {filepath.name}: {e}", icon="🚫"); return None

def reset_chat_session_state(new_chat_id=None):
    """Resets state variables specific to a single chat session."""
    st.session_state.messages = []
    st.session_state.current_chat_id = new_chat_id or str(uuid.uuid4())
    st.session_state.current_chat_name = "New Chat"
    st.session_state.renaming_chat_id = None
    # Cannot directly modify widget state
    # st.session_state.file_uploader_sidebar = None

def _load_chat_data_into_state(data, source_description):
    """Loads parsed chat data dictionary into session state."""
    if not data: return False
    try:
        reset_chat_session_state(new_chat_id=data.get("chat_id")) # Reset first

        st.session_state.current_chat_name = data.get("chat_name", "Loaded Chat")
        st.session_state.model = data.get("model", DEFAULT_OLLAMA_MODEL)
        st.session_state.system_instruction = data.get("system_instruction", DEFAULT_SYSTEM_PROMPT)
        st.session_state.temperature = data.get("temperature", 0.7)

        loaded_messages_simple = data.get("messages", [])
        st.session_state.messages = [] # Ensure it's empty before loading
        for msg_data in loaded_messages_simple:
             st.session_state.messages.append({ "role": msg_data.get("role"), "content": msg_data.get("content", "") })

        print(f"Chat '{st.session_state.current_chat_name}' loaded from {source_description}!")
        set_last_chat_id(st.session_state.current_chat_id)
        st.session_state.available_models = fetch_available_models()
        return True
    except Exception as e: st.error(f"Error applying loaded chat data: {e}"); traceback.print_exc(); return False

def load_chat_from_id(chat_id):
    """Loads chat data from a history file ID into session state."""
    chat_data = load_chat_data(chat_id)
    if chat_data: return _load_chat_data_into_state(chat_data, f"history (ID: {chat_id[:8]}...)")
    return False

def list_saved_chats():
    """Lists saved chats metadata from the history directory."""
    chat_files_meta = []
    for filepath in HISTORY_DIR.glob("ollama_chat_*.json"):
        file_chat_id = filepath.stem.replace("ollama_chat_", "")
        chat_data = load_chat_data(file_chat_id)
        if chat_data:
            saved_at_str = chat_data.get("saved_at"); saved_at_dt = datetime.min
            try:
                if saved_at_str: saved_at_dt = datetime.fromisoformat(saved_at_str)
            except ValueError: pass
            chat_files_meta.append({ "id": chat_data.get("chat_id", file_chat_id), "name": chat_data.get("chat_name", "Untitled Chat"),
                "saved_at_dt": saved_at_dt, "message_count": len(chat_data.get("messages", [])), "model": chat_data.get("model", "Unknown") })
    chat_files_meta.sort(key=lambda x: x["saved_at_dt"], reverse=True)
    return chat_files_meta

def delete_chat_file(chat_id):
    """Deletes the chat file."""
    filepath = get_chat_filepath(chat_id)
    try:
        if filepath.exists():
            filepath.unlink(); st.toast(f"Deleted chat ID {chat_id[:8]}...", icon="🗑️")
            if get_last_chat_id() == chat_id:
                if LAST_CHAT_ID_FILE.exists(): LAST_CHAT_ID_FILE.unlink()
            if st.session_state.get("renaming_chat_id") == chat_id: st.session_state.renaming_chat_id = None
            return True
        else: st.warning(f"Chat file ID {chat_id[:8]}... not found.", icon="⚠️"); return False
    except Exception as e: st.error(f"Error deleting {filepath.name}: {e}", icon="❌"); return False


# --- File Handling ---
def encode_image_to_base64(image_path):
    """Encodes an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e: st.error(f"Error encoding image {image_path}: {e}"); return None

def handle_file_upload(uploaded_file):
    """Processes uploaded file, extracts data, returns temp path and data dict."""
    if not uploaded_file: return None, None
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue()); file_path = tmp_file.name
        if uploaded_file.type.startswith("image/"):
            encoded_data = encode_image_to_base64(file_path)
            if encoded_data: return file_path, {"type": "image", "data": encoded_data, "filename": uploaded_file.name}
            else: return file_path, None # Return path for cleanup
        elif uploaded_file.type == "application/pdf":
            text_content = ""; pdf_file = None
            try:
                import PyPDF2
                pdf_file = open(file_path, "rb")
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                if pdf_reader.is_encrypted: st.warning("Skipping encrypted PDF."); return file_path, None
                text_content = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
            except ImportError: st.error("Install PyPDF2 (`pip install pypdf2`)"); return file_path, None
            except Exception as e: st.error(f"Error processing PDF: {e}"); traceback.print_exc(); return file_path, None
            finally:
                 if pdf_file is not None and not pdf_file.closed: pdf_file.close()
            if text_content: return file_path, {"type": "pdf", "data": text_content, "filename": uploaded_file.name}
            else: st.warning("Could not extract text from PDF."); return file_path, None
        else: st.warning(f"Unsupported file type: {uploaded_file.type}"); return file_path, None
    except Exception as e: st.error(f"Error handling file upload: {e}"); traceback.print_exc(); return file_path, None

# --- Startup Logic ---
def run_startup_logic():
    """Handles autoloading the last chat on app startup."""
    if st.session_state.get('app_just_started', True):
        st.session_state.app_just_started = False; st.session_state.loaded_on_start = False; loaded_existing = False
        if st.session_state.autoload_last_chat:
            last_id = get_last_chat_id()
            if last_id:
                print(f"Attempting to autoload last chat ID: {last_id}")
                if load_chat_from_id(last_id): st.session_state.loaded_on_start = True; loaded_existing = True; print(f"Autoloaded: {st.session_state.current_chat_name}")
                else: print(f"Failed load last chat ID {last_id}"); reset_chat_session_state(new_chat_id=st.session_state.current_chat_id)
        if not loaded_existing: print("Saving initial state."); save_current_chat_to_file()
        check_ollama_connection(); st.session_state.available_models = fetch_available_models()
        if st.session_state.model not in st.session_state.available_models:
             print(f"Model '{st.session_state.model}' not found. Switching."); available_models = st.session_state.available_models
             st.session_state.model = available_models[0] if available_models else DEFAULT_OLLAMA_MODEL
    if st.session_state.get("loaded_on_start", False): st.success(f"Chat '{st.session_state.current_chat_name}' auto-loaded!", icon="📂"); st.session_state.loaded_on_start = False

# --- Helper to remove last message on error ---
def _remove_last_user_message_on_error():
    """Removes the last user message from state if an API call failed."""
    try:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user": st.session_state.messages.pop(); print("Removed last user message due to error.")
    except Exception as e: print(f"Error removing last user message: {e}")


# ===== Main App Execution =====

st.set_page_config(page_title="Ollama Chat", page_icon="🤖", layout="wide")
initialize_session()
run_startup_logic()

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Settings & Chats")
    st.divider()

    with st.expander("🤖 Model & Instructions", expanded=True):
        system_instruction = st.text_area( "System Instruction", value=st.session_state.system_instruction, height=100, key="system_prompt_input", help="Set the behavior of the AI assistant" )
        if system_instruction != st.session_state.system_instruction: st.session_state.system_instruction = system_instruction; save_current_chat_to_file(); st.toast("Instruction updated!", icon="📝")

        available_models = st.session_state.get("available_models", [DEFAULT_OLLAMA_MODEL]); current_model_index = 0
        try:
            if st.session_state.model in available_models: current_model_index = available_models.index(st.session_state.model)
            else:
                 if st.session_state.model not in available_models: available_models.insert(0, st.session_state.model)
                 current_model_index = 0
        except Exception: current_model_index = 0
        selected_model = st.selectbox( "Model", available_models, index=current_model_index, key="model_select", help="Choose the Ollama model to use" )
        if selected_model != st.session_state.model: print(f"Switching model to: {selected_model}"); st.session_state.model = selected_model; save_current_chat_to_file(); st.toast(f"Switched model to {selected_model}!", icon="🔄")

        temperature = st.slider( "Temperature", 0.0, 2.0, value=st.session_state.temperature, step=0.1, key="temperature_slider", help="Controls randomness (0=deterministic, >1 more random)", on_change=save_current_chat_to_file )
        if temperature != st.session_state.temperature: st.session_state.temperature = temperature

    st.divider()

    with st.expander("💬 Chats & History", expanded=True):
        col_manage1, col_manage2 = st.columns(2)
        with col_manage1:
            if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"):
                save_current_chat_to_file(); reset_chat_session_state(); save_current_chat_to_file(); st.success("Started new chat.", icon="✨"); st.rerun()
        with col_manage2:
            try:
                export_data_str = json.dumps(create_save_data(), indent=2)
                export_safe_name = "".join(c if c.isalnum() else "_" for c in st.session_state.current_chat_name)
                export_filename = f"ollama_export_{export_safe_name}_{datetime.now():%Y%m%d_%H%M}.json"
                st.download_button("📥 Export Chat", export_data_str, export_filename, "application/json", use_container_width=True, disabled=not st.session_state.messages, help="Download current chat.")
            except Exception as e_export: st.error(f"Export error: {e_export}", icon="💾")

        st.markdown("---"); st.subheader("Saved Chats")
        st.session_state.chat_history_list = list_saved_chats()
        if not st.session_state.chat_history_list: st.caption("No saved chats yet.")
        else:
            st.caption("Load, rename, or delete past chats.")
            for chat_meta in st.session_state.chat_history_list:
                chat_id = chat_meta['id']; display_name = chat_meta['name']
                is_current = (chat_id == st.session_state.current_chat_id)
                is_renaming = (chat_id == st.session_state.get("renaming_chat_id"))
                col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
                with col1:
                    if is_renaming: st.text_input("New name:", value=display_name, key=f"rename_input_{chat_id}", label_visibility="collapsed")
                    else:
                        load_key = f"load_{chat_id}"; button_type = "primary" if is_current else "secondary"
                        if st.button(f"{display_name}", key=load_key, use_container_width=True, help=f"Load '{display_name}'", type=button_type):
                            if not is_current: save_current_chat_to_file()
                            if load_chat_from_id(chat_id): st.rerun()
                with col2:
                    if is_renaming:
                        save_rename_key = f"save_rename_{chat_id}"
                        if st.button("✅", key=save_rename_key, help="Save new name", use_container_width=True):
                            new_name = st.session_state.get(f"rename_input_{chat_id}", display_name).strip()
                            if new_name and new_name != display_name:
                                chat_data = load_chat_data(chat_id)
                                if chat_data: chat_data["chat_name"] = new_name
                                if save_specific_chat_data(chat_id, chat_data): st.toast(f"Renamed to '{new_name}'", icon="✏️");
                                if is_current: st.session_state.current_chat_name = new_name
                                st.session_state.renaming_chat_id = None; st.rerun()
                            elif new_name == display_name: st.session_state.renaming_chat_id = None; st.rerun()
                            else: st.warning("Enter a valid name.", icon="⚠️")
                    else:
                         rename_icon_key = f"rename_icon_{chat_id}";
                         if st.button("✏️", key=rename_icon_key, help="Rename this chat", use_container_width=True): st.session_state.renaming_chat_id = chat_id; st.rerun()
                with col3:
                    if is_renaming:
                         cancel_rename_key = f"cancel_rename_{chat_id}";
                         if st.button("❌", key=cancel_rename_key, help="Cancel rename", use_container_width=True): st.session_state.renaming_chat_id = None; st.rerun()
                    else:
                         delete_key = f"delete_{chat_id}"
                         if st.button("🗑️", key=delete_key, help=f"Delete chat '{display_name}'", use_container_width=True):
                             if delete_chat_file(chat_id):
                                 if is_current:
                                     remaining_chats = list_saved_chats(); loaded_new = False
                                     if remaining_chats:
                                         if load_chat_from_id(remaining_chats[0]['id']): st.success("Deleted current, loaded recent.", icon="✨"); loaded_new = True
                                     if not loaded_new: reset_chat_session_state(); save_current_chat_to_file(); st.success("Deleted last chat, started new one.", icon="✨")
                                 st.rerun()
    st.divider()

    with st.expander("📎 Attach File", expanded=False):
        uploaded_file = st.file_uploader( "Upload Image or PDF", type=["pdf", "png", "jpg", "jpeg"], key="file_uploader_sidebar", help="Attach a file to your next message." )

    st.divider()
    st.checkbox("Auto-load last chat on startup", key="autoload_last_chat", value=st.session_state.autoload_last_chat, help="Load the most recent chat when the app starts.")
    if st.session_state.ollama_reachable: st.success("Ollama connected", icon="🟢")
    else: st.error("Ollama disconnected", icon="🔴")
    if st.button("Check Connection"): st.session_state.ollama_reachable = None; check_ollama_connection(); st.rerun()


# --- Main Chat Interface ---
st.title(f"Ollama Chat: {st.session_state.current_chat_name}")

chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.get("messages", [])):
        with st.chat_message(message.get("role", "user")):
            if "content" in message and message["content"]:
                st.markdown(message["content"])
                if message.get("role") == "assistant":
                     try:
                         from st_copy_to_clipboard import st_copy_to_clipboard
                         copy_key=f"copy_{st.session_state.current_chat_id}_{i}"
                         st_copy_to_clipboard(message["content"], key=copy_key) # Basic call
                     except ImportError: pass
                     except Exception as e_copy: print(f"Error rendering copy button: {e_copy}")
            if "image_data" in message and message["image_data"]:
                 try: st.image(base64.b64decode(message["image_data"]), caption=message.get("image_filename","Attached Image"))
                 except Exception as e_img: st.error(f"Error displaying image: {e_img}")
            if "pdf_info" in message and message["pdf_info"]:
                st.info(f"📎 Context from: {message['pdf_info']}")


# --- Chat Input Logic ---
if prompt := st.chat_input("Ask Ollama..." if st.session_state.ollama_reachable else "Ollama not connected...", disabled=not st.session_state.ollama_reachable, key="chat_input_main"):

    if st.session_state.renaming_chat_id: st.session_state.renaming_chat_id = None

    temp_file_path_to_delete = None # Initialize path for potential deletion
    processed_file_data = None
    user_attachments_display = {}
    # Get the potential file from the session state key
    current_uploaded_file = st.session_state.get("file_uploader_sidebar")

    if current_uploaded_file:
        # Process file BEFORE adding user msg to state
        temp_file_path_to_delete, processed_file_data = handle_file_upload(current_uploaded_file)

        if processed_file_data:
             user_attachments_display = processed_file_data
        else:
            st.error("Failed to process uploaded file.")
            # If processing failed but a temp file was created, delete it now
            if temp_file_path_to_delete and os.path.exists(temp_file_path_to_delete):
                try:
                    os.unlink(temp_file_path_to_delete)
                    print(f"Cleaned up failed temp file: {temp_file_path_to_delete}")
                except Exception as e_del:
                     print(f"Error deleting failed temp file: {e_del}")
            temp_file_path_to_delete = None # Nullify path as processing failed

        # *** REMOVED THIS LINE AGAIN - Cannot set widget state ***
        # st.session_state.file_uploader_sidebar = None

    # Add user message to state (display happens on rerun)
    user_message = {"role": "user", "content": prompt}
    if processed_file_data:
        if processed_file_data["type"] == "image": user_message["image_data"] = processed_file_data["data"]; user_message["image_filename"] = processed_file_data["filename"]
        elif processed_file_data["type"] == "pdf": user_message["pdf_info"] = processed_file_data["filename"] # Store info for display
    st.session_state.messages.append(user_message)
    save_current_chat_to_file() # Auto-save

    # Prepare and call API
    with st.chat_message("assistant"):
        message_placeholder = st.empty(); message_placeholder.markdown('<div class="thinking-indicator">Thinking... 💭</div>', unsafe_allow_html=True)
        full_response = ""; response = None
        try:
            # --- REVISED PAYLOAD CONSTRUCTION ---
            user_query = prompt.strip() if prompt else ""
            payload = { "model": st.session_state.model, "system": st.session_state.system_instruction, "stream": True, "options": {"temperature": st.session_state.temperature} }

            if processed_file_data:
                filename = processed_file_data['filename']
                file_type = processed_file_data['type']
                if file_type == "image":
                    payload["images"] = [processed_file_data["data"]]
                    if not user_query: final_prompt_for_api = f"Describe the key elements in the uploaded image '{filename}'."
                    else: final_prompt_for_api = user_query
                elif file_type == "pdf":
                    pdf_text = processed_file_data['data']
                    if not user_query: final_prompt_for_api = ( f"Please summarize the main points or provide an overview of the following document named '{filename}'.\n\n" f"--- DOCUMENT CONTENT START ---\n{pdf_text}\n--- DOCUMENT CONTENT END ---" )
                    else: final_prompt_for_api = ( f"Please answer the following user query based *only* on the provided text from the document '{filename}'. Do not use any prior knowledge.\n\n" f"--- DOCUMENT CONTENT START ---\n{pdf_text}\n--- DOCUMENT CONTENT END ---\n\nUser Query: {user_query}" )
                    print(f"Sending PDF content for {filename} with instructions.")
                else: final_prompt_for_api = user_query
            else: final_prompt_for_api = user_query

            if not final_prompt_for_api:
                 st.warning("Please enter a message or upload a file and ask a question about it.")
                 if temp_file_path_to_delete and os.path.exists(temp_file_path_to_delete):
                     try: os.unlink(temp_file_path_to_delete)
                     except Exception as e_del: print(f"Error deleting temp file: {e_del}")
                 _remove_last_user_message_on_error(); st.stop()

            payload["prompt"] = final_prompt_for_api
            # ------------------------------------

            # --- API CALL ---
            response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True, timeout=180)
            response.raise_for_status()

            # --- STREAM PROCESSING ---
            for line in response.iter_lines():
                if line:
                    try:
                        part = json.loads(line.decode('utf-8')); content = part.get("response", "")
                        full_response += content; message_placeholder.markdown(full_response + " ▌")
                        if part.get("done"): break
                    except json.JSONDecodeError: print(f"Warning: JSON Decode Error: {line}")
                    except Exception as e_stream: print(f"Error processing stream line: {e_stream}"); full_response += f"\n*(Stream Error)*"; break
            message_placeholder.markdown(full_response)

        # --- ERROR HANDLING ---
        except requests.exceptions.ConnectionError: st.error("Connection Error."); st.session_state.ollama_reachable = False; full_response = "*Connection Error*"; message_placeholder.error(full_response); _remove_last_user_message_on_error()
        except requests.exceptions.Timeout: st.error("Request timed out."); full_response = "*Request Timed Out.*"; message_placeholder.error(full_response); _remove_last_user_message_on_error()
        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}"); full_response = f"*API Error: {e}*"
            error_detail = ""
            if response is not None:
                 try: error_detail = response.text
                 except Exception: error_detail = "(Could not retrieve details)"
            if error_detail: full_response += f"\nDetails: {error_detail}"; st.error(f"Details: {error_detail}")
            message_placeholder.error(full_response); _remove_last_user_message_on_error()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            traceback.print_exc()
            full_response = f"*Error: {e}*"
            message_placeholder.error(full_response)
            _remove_last_user_message_on_error()
        # --- FINALLY BLOCK ---
        finally:
            # Add assistant response to state
            if not full_response.startswith("*"):
                 st.session_state.messages.append({"role": "assistant", "content": full_response})
                 save_current_chat_to_file() # Save assistant response
            # *** CENTRALIZED DELETION OF TEMP FILE ***
            if temp_file_path_to_delete and os.path.exists(temp_file_path_to_delete):
                try:
                    os.unlink(temp_file_path_to_delete)
                    print(f"Deleted temp file: {temp_file_path_to_delete}")
                except Exception as e_del:
                    print(f"Error deleting temp file {temp_file_path_to_delete}: {e_del}")

    # Rerun to display everything cleanly
    st.rerun()


# --- Footer Buttons ---
st.divider()
if st.button("🗑️ Clear Current Chat", key="clear_chat_main_area"):
    st.session_state.messages = []; save_current_chat_to_file(); st.rerun()