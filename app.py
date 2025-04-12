# --- Keep all imports and functions from the previous version ---
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
DEFAULT_OLLAMA_MODEL = "llama3" # <--- CHANGED DEFAULT TO TRY A MORE ROBUST MODEL
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant. Please provide clear and concise responses."
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434") # Allow overriding via environment variable


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
    st.session_state.setdefault("response_count", 0) # Used for clearing file uploader

# --- All Helper Functions (check_ollama_connection, fetch_available_models, history management, file handling, startup logic, _remove_last_user_message_on_error) ---
# ... PASTE ALL FUNCTIONS FROM THE PREVIOUS VERSION HERE ...
# --- Ollama Interaction Functions ---
def check_ollama_connection():
    """Checks if the Ollama server is reachable. Updates state."""
    is_reachable = False
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/", timeout=2)
        is_reachable = response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        is_reachable = False
    except Exception as e:
        print(f"Unexpected error checking Ollama connection: {e}")
        is_reachable = False
    finally:
        # Update state only if it changes or is None
        if st.session_state.ollama_reachable != is_reachable:
            st.session_state.ollama_reachable = is_reachable
            return True # Indicates state changed
    return False # Indicates state did not change


def fetch_available_models():
    """Fetches available models from Ollama API."""
    # Only fetch if connection is confirmed or hasn't been checked yet
    if st.session_state.ollama_reachable == False:
        return [st.session_state.get("model", DEFAULT_OLLAMA_MODEL)] # Return current/default if not connected

    print("Attempting to fetch models...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) # Longer timeout for fetching
        response.raise_for_status()
        models_data = response.json()
        models = models_data.get("models", [])
        model_names = sorted([model["name"] for model in models if "name" in model])
        print(f"Models fetched: {model_names}")
        # Update connection status based on successful fetch
        if st.session_state.ollama_reachable != True:
             st.session_state.ollama_reachable = True
             st.rerun() # Rerun if connection status changed to True
        return model_names if model_names else [DEFAULT_OLLAMA_MODEL]
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        st.error(f"Connection error fetching models from {OLLAMA_BASE_URL}.")
        if st.session_state.ollama_reachable != False:
            st.session_state.ollama_reachable = False
            st.rerun() # Rerun if connection status changed to False
        return [st.session_state.get("model", DEFAULT_OLLAMA_MODEL)]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching models: {e}. Check Ollama server.")
        if st.session_state.ollama_reachable != False:
            st.session_state.ollama_reachable = False
            st.rerun() # Rerun if connection status changed to False
        return [st.session_state.get("model", DEFAULT_OLLAMA_MODEL)]
    except json.JSONDecodeError:
        st.error("Error decoding model list from Ollama. Using default.")
        return [st.session_state.get("model", DEFAULT_OLLAMA_MODEL)]

# Stop model function might not be reliable or necessary with standard usage
# def stop_ollama_model(model_name): ...


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
    # Only include core message data needed for reload/API
    messages_to_save = []
    for msg in st.session_state.get("messages", []):
        core_msg = { "role": msg.get("role"), "content": msg.get("content", "") }
        # Include image data if present
        if msg.get("image_data"):
            core_msg["image_data"] = msg["image_data"]
            core_msg["image_filename"] = msg.get("image_filename") # Save filename for display
        # Include PDF info if present (just for display)
        if msg.get("pdf_info"):
             core_msg["pdf_info"] = msg.get("pdf_info")
        messages_to_save.append(core_msg)

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
        # print(f"Chat {chat_id} saved.") # Optional console log
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
    st.session_state.response_count = 0 # Reset for file uploader key
    # Don't reset global settings like model, temp, system prompt by default
    # st.session_state.model = DEFAULT_OLLAMA_MODEL # Optionally reset model too
    # st.session_state.system_instruction = DEFAULT_SYSTEM_PROMPT # Optionally reset prompt
    # st.session_state.temperature = 0.7 # Optionally reset temp


def _load_chat_data_into_state(data, source_description):
    """Loads parsed chat data dictionary into session state."""
    if not data: return False
    try:
        # Reset core chat state before loading
        reset_chat_session_state(new_chat_id=data.get("chat_id"))

        # Load chat-specific settings
        st.session_state.current_chat_name = data.get("chat_name", "Loaded Chat")
        st.session_state.model = data.get("model", DEFAULT_OLLAMA_MODEL)
        st.session_state.system_instruction = data.get("system_instruction", DEFAULT_SYSTEM_PROMPT)
        st.session_state.temperature = data.get("temperature", 0.7)

        # Load messages (ensure they are loaded correctly)
        loaded_messages = data.get("messages", [])
        st.session_state.messages = [] # Ensure it's empty before loading
        for msg_data in loaded_messages:
             # Load all relevant keys saved in create_save_data
             st.session_state.messages.append({
                 "role": msg_data.get("role"),
                 "content": msg_data.get("content", ""),
                 "image_data": msg_data.get("image_data"),
                 "image_filename": msg_data.get("image_filename"),
                 "pdf_info": msg_data.get("pdf_info")
             })

        print(f"Chat '{st.session_state.current_chat_name}' (ID: {st.session_state.current_chat_id[:8]}) loaded from {source_description}!")
        set_last_chat_id(st.session_state.current_chat_id)

        # Fetch models again after loading in case the loaded model needs verification
        st.session_state.available_models = fetch_available_models()
        if st.session_state.model not in st.session_state.available_models:
            st.warning(f"Loaded chat uses model '{st.session_state.model}' which is not currently available. Select an available model.", icon="⚠️")
            # Optionally switch to default or first available
            # st.session_state.model = DEFAULT_OLLAMA_MODEL

        return True
    except Exception as e:
        st.error(f"Error applying loaded chat data: {e}")
        traceback.print_exc()
        return False

def load_chat_from_id(chat_id):
    """Loads chat data from a history file ID into session state."""
    chat_data = load_chat_data(chat_id)
    if chat_data:
        if _load_chat_data_into_state(chat_data, f"history (ID: {chat_id[:8]}...)"):
            st.session_state.response_count = 0 # Reset uploader on load
            return True
    return False

def list_saved_chats():
    """Lists saved chats metadata from the history directory."""
    chat_files_meta = []
    for filepath in sorted(HISTORY_DIR.glob("ollama_chat_*.json"), key=os.path.getmtime, reverse=True): # Sort by mod time
        file_chat_id = filepath.stem.replace("ollama_chat_", "")
        chat_data = load_chat_data(file_chat_id) # Use loading function to handle errors
        if chat_data:
            saved_at_str = chat_data.get("saved_at")
            saved_at_dt = datetime.min
            try:
                if saved_at_str: saved_at_dt = datetime.fromisoformat(saved_at_str)
            except (ValueError, TypeError): pass # Ignore invalid date format

            chat_files_meta.append({
                "id": chat_data.get("chat_id", file_chat_id),
                "name": chat_data.get("chat_name", "Untitled Chat"),
                "saved_at_dt": saved_at_dt,
                "message_count": len(chat_data.get("messages", [])),
                "model": chat_data.get("model", "Unknown")
            })
    # Already sorted by file mod time, which often correlates with save time
    # chat_files_meta.sort(key=lambda x: x["saved_at_dt"], reverse=True)
    return chat_files_meta

def delete_chat_file(chat_id):
    """Deletes the chat file."""
    filepath = get_chat_filepath(chat_id)
    try:
        if filepath.exists():
            chat_name = load_chat_data(chat_id).get("chat_name", f"ID {chat_id[:8]}...") # Get name before deleting
            filepath.unlink()
            st.toast(f"Deleted chat '{chat_name}'", icon="🗑️")
            # If deleting the last used chat, remove the pointer
            if get_last_chat_id() == chat_id:
                if LAST_CHAT_ID_FILE.exists():
                    try: LAST_CHAT_ID_FILE.unlink()
                    except OSError as e_unlink: print(f"Error removing last chat ID file: {e_unlink}")
            # If deleting the chat being renamed, cancel rename
            if st.session_state.get("renaming_chat_id") == chat_id:
                st.session_state.renaming_chat_id = None
            return True
        else:
            st.warning(f"Chat file ID {chat_id[:8]}... not found for deletion.", icon="⚠️")
            return False
    except Exception as e:
        st.error(f"Error deleting chat file {filepath.name}: {e}", icon="❌")
        return False


# --- File Handling ---
def encode_image_to_base64(image_path):
    """Encodes an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image {Path(image_path).name}: {e}")
        return None

def handle_file_upload(uploaded_file):
    """Processes uploaded file, extracts data, returns temp path and data dict."""
    if not uploaded_file: return None, None

    file_path = None
    processed_data = None
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    print(f"Handling upload: {uploaded_file.name}, Type: {uploaded_file.type}, Ext: {file_extension}")

    try:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        print(f"Temporary file created at: {file_path}")

        # --- Image Processing ---
        if uploaded_file.type.startswith("image/"):
            print("Processing as image...")
            encoded_data = encode_image_to_base64(file_path)
            if encoded_data:
                processed_data = {"type": "image", "data": encoded_data, "filename": uploaded_file.name}
                print("Image encoded successfully.")
            else:
                st.error("Image encoding failed.")
                # Keep file_path, processed_data remains None

        # --- PDF Processing ---
        elif uploaded_file.type == "application/pdf" or file_extension == ".pdf":
            print("Processing as PDF...")
            text_content = ""
            try:
                # Ensure PyPDF2 is installed: pip install pypdf2
                import PyPDF2
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    if pdf_reader.is_encrypted:
                        try:
                             # Attempt to decrypt with an empty password (common case)
                             password_type = pdf_reader.decrypt('')
                             if password_type == PyPDF2.PasswordType.OWNER_PASSWORD or password_type == PyPDF2.PasswordType.USER_PASSWORD:
                                 print("PDF decrypted with empty password.")
                             else:
                                 # Handle case where decryption fails or needs a password
                                 st.warning(f"Skipping password-protected PDF: {uploaded_file.name}", icon="🔒")
                                 return file_path, None # Return path for cleanup, no data
                        except Exception as decrypt_err:
                            st.warning(f"Could not decrypt PDF '{uploaded_file.name}': {decrypt_err}. Skipping.", icon="🔒")
                            return file_path, None

                    # Proceed with text extraction if not encrypted or decrypted
                    num_pages = len(pdf_reader.pages)
                    print(f"Extracting text from {num_pages} pages...")
                    extracted_texts = []
                    for i, page in enumerate(pdf_reader.pages):
                         try:
                             page_text = page.extract_text()
                             if page_text:
                                 extracted_texts.append(page_text)
                         except Exception as page_err:
                              print(f"Warning: Could not extract text from page {i+1} of {uploaded_file.name}: {page_err}")
                    text_content = "\n\n".join(extracted_texts) # Join pages with double newline
                    print(f"Extracted ~{len(text_content)} characters from PDF.")

            except ImportError:
                st.error("PyPDF2 is required for PDF processing. Please install it (`pip install pypdf2`).")
                return file_path, None # Return path for cleanup, no data
            except Exception as e:
                st.error(f"Error processing PDF '{uploaded_file.name}': {e}")
                traceback.print_exc()
                return file_path, None # Return path for cleanup, no data

            if text_content:
                processed_data = {"type": "pdf", "data": text_content, "filename": uploaded_file.name}
            else:
                st.warning(f"Could not extract any text from PDF: {uploaded_file.name}", icon="⚠️")
                # Keep file_path, processed_data remains None

        # --- Unsupported File Type ---
        else:
            st.warning(f"Unsupported file type: {uploaded_file.type} ({uploaded_file.name})", icon="❓")
            # Keep file_path, processed_data remains None

    except Exception as e:
        st.error(f"Error handling file upload for '{uploaded_file.name}': {e}")
        traceback.print_exc()
        # Ensure temp file path is returned for cleanup if it was created
        return file_path, None

    # Return the path (for potential deletion) and the processed data (or None if failed)
    return file_path, processed_data

# --- Startup Logic ---
def run_startup_logic():
    """Handles initial setup: connection check, model fetch, autoload."""
    if st.session_state.get('app_just_started', True):
        print("\n--- App Startup ---")
        st.session_state.app_just_started = False
        st.session_state.loaded_on_start = False
        loaded_existing_chat = False

        # 1. Check connection and fetch models early
        print("Initial connection check...")
        check_ollama_connection()
        st.session_state.available_models = fetch_available_models()
        print(f"Startup models available: {st.session_state.available_models}")

        # 2. Attempt to autoload last chat if enabled
        if st.session_state.autoload_last_chat:
            last_id = get_last_chat_id()
            if last_id:
                print(f"Attempting to autoload last chat ID: {last_id[:8]}...")
                # Load function now handles state update and model check
                if load_chat_from_id(last_id):
                    st.session_state.loaded_on_start = True
                    loaded_existing_chat = True
                    print(f"Autoloaded chat: '{st.session_state.current_chat_name}'")
                else:
                    print(f"Failed to load last chat ID {last_id[:8]}... starting new.")
                    # Ensure state is reset if load fails
                    reset_chat_session_state(new_chat_id=st.session_state.current_chat_id) # Keep ID if needed
                    save_current_chat_to_file() # Save the fresh state
            else:
                 print("No last chat ID found.")

        # 3. If no chat was loaded, ensure initial state is saved
        if not loaded_existing_chat:
            print("No chat autoloaded, ensuring initial new chat state is saved.")
            # Make sure model is valid before saving initial state
            if st.session_state.model not in st.session_state.available_models:
                print(f"Initial model '{st.session_state.model}' invalid, switching to default/first.")
                st.session_state.model = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in st.session_state.available_models else (st.session_state.available_models[0] if st.session_state.available_models else DEFAULT_OLLAMA_MODEL)
            save_current_chat_to_file()

        print("--- Startup Complete ---\n")

    # Display autoload success message only once after startup
    if st.session_state.get("loaded_on_start", False):
        st.success(f"Chat '{st.session_state.current_chat_name}' auto-loaded!", icon="📂")
        st.session_state.loaded_on_start = False # Prevent showing again


# --- Helper to remove last message on error ---
def _remove_last_user_message_on_error():
    """Removes the last user message from state if an API call failed."""
    try:
        if st.session_state.messages and st.session_state.messages[-1].get("role") == "user":
            removed_msg = st.session_state.messages.pop()
            print(f"Removed last user message due to API error: {removed_msg.get('content', '')[:50]}...")
            # Optionally save the state after removal
            save_current_chat_to_file()
            return True
    except Exception as e:
        print(f"Error removing last user message: {e}")
    return False

# ===== Main App Execution =====

st.set_page_config(page_title="Ollama Chat", page_icon="🤖", layout="wide")

# Initialize session state if not already done (safe to call multiple times)
initialize_session()

# Run startup logic (connection check, autoload) - happens only once
run_startup_logic()

# --- Sidebar UI ---
with st.sidebar:
    st.title("⚙️ Settings & Chats")
    st.divider()

    # --- Connection Status ---
    connection_status = st.empty() # Placeholder for status message
    def update_connection_status():
        if st.session_state.ollama_reachable is None:
            connection_status.warning("Checking connection...", icon="⏳")
        elif st.session_state.ollama_reachable:
            connection_status.success(f"Ollama connected ({OLLAMA_BASE_URL})", icon="🟢")
        else:
            connection_status.error(f"Ollama disconnected ({OLLAMA_BASE_URL})", icon="🔴")
    update_connection_status() # Initial display

    if st.button("🔄 Refresh Connection & Models"):
        st.session_state.ollama_reachable = None # Force re-check
        check_ollama_connection()
        st.session_state.available_models = fetch_available_models()
        update_connection_status() # Update display after check
        st.rerun() # Rerun to update model list selector


    st.divider()

    # --- Model Settings ---
    with st.expander("🤖 Model & Instructions", expanded=True):
        # System Prompt
        system_instruction = st.text_area(
            "System Instruction",
            value=st.session_state.system_instruction,
            height=100,
            key="system_prompt_input",
            help="Set the behavior of the AI assistant (applied when chat starts or instruction changes)."
        )
        if system_instruction != st.session_state.system_instruction:
            print("System instruction changed.")
            st.session_state.system_instruction = system_instruction
            save_current_chat_to_file()
            st.toast("System instruction updated!", icon="📝")
            # Note: This takes effect on the *next* message or if chat is reloaded.

        # Model Selection
        available_models = st.session_state.get("available_models", [DEFAULT_OLLAMA_MODEL])
        current_model = st.session_state.get("model", DEFAULT_OLLAMA_MODEL)
        current_model_index = 0
        # Ensure current model is in the list for the selectbox
        if current_model not in available_models:
            available_models.insert(0, f"{current_model} (unavailable)") # Add placeholder if missing
            current_model_index = 0
        else:
            try:
                 current_model_index = available_models.index(current_model)
            except ValueError:
                 # Should not happen if logic above is correct, but fallback safely
                 available_models.insert(0, f"{current_model} (unavailable)")
                 current_model_index = 0


        selected_model_display = st.selectbox(
            "Model",
            available_models,
            index=current_model_index,
            key="model_select",
            help="Choose the Ollama model. If context is lost, try llama3 or mistral."
        )
        # Handle selection, ignoring the "(unavailable)" tag if present
        selected_model_actual = selected_model_display.replace(" (unavailable)", "")
        if selected_model_actual != st.session_state.model:
            print(f"Switching model from '{st.session_state.model}' to: '{selected_model_actual}'")
            st.session_state.model = selected_model_actual
            save_current_chat_to_file()
            st.toast(f"Switched model to {selected_model_actual}!", icon="🔄")
            # Rerun to ensure UI consistency if model was previously unavailable
            if "(unavailable)" in selected_model_display:
                 st.rerun()

        # Temperature
        temperature = st.slider(
            "Temperature", 0.0, 2.0,
            value=st.session_state.temperature,
            step=0.1,
            key="temperature_slider",
            help="Controls randomness (0=deterministic, >1 more random). Applied on next message.",
            on_change=save_current_chat_to_file # Save immediately on change
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
            # No need to toast or rerun, on_change saves it.

    st.divider()

    # --- Chat Management ---
    with st.expander("💬 Chats & History", expanded=True):
        col_manage1, col_manage2 = st.columns(2)
        with col_manage1:
            if st.button("➕ New Chat", use_container_width=True, key="new_chat_sidebar"):
                # Save current chat *before* resetting if messages exist
                if st.session_state.messages:
                    save_current_chat_to_file(show_toast=False) # Save silently
                reset_chat_session_state()
                save_current_chat_to_file(show_toast=False) # Save the new empty chat state
                st.success("Started new chat.", icon="✨")
                st.rerun()
        with col_manage2:
            try:
                export_data_str = json.dumps(create_save_data(), indent=2)
                export_safe_name = "".join(c if c.isalnum() else "_" for c in st.session_state.current_chat_name)
                export_filename = f"ollama_export_{export_safe_name}_{datetime.now():%Y%m%d_%H%M}.json"
                st.download_button(
                    "📥 Export Chat",
                    export_data_str,
                    export_filename,
                    "application/json",
                    use_container_width=True,
                    disabled=not st.session_state.messages,
                    help="Download current chat as a JSON file."
                )
            except Exception as e_export: st.error(f"Export error: {e_export}", icon="💾")

        # --- Saved Chats List ---
        st.markdown("---")
        st.subheader("Saved Chats")
        st.session_state.chat_history_list = list_saved_chats() # Refresh list

        if not st.session_state.chat_history_list:
            st.caption("No saved chats yet.")
        else:
            st.caption(f"Load, rename, or delete past chats ({len(st.session_state.chat_history_list)} total).")
            # Use a container for scrolling if list gets long
            history_container = st.container(height=300) # Adjust height as needed
            with history_container:
                for chat_meta in st.session_state.chat_history_list:
                    chat_id = chat_meta['id']
                    display_name = chat_meta['name']
                    is_current = (chat_id == st.session_state.current_chat_id)
                    is_renaming = (chat_id == st.session_state.get("renaming_chat_id"))

                    col1, col2, col3 = st.columns([0.7, 0.15, 0.15])

                    with col1:
                        # --- Rename Input ---
                        if is_renaming:
                            st.text_input(
                                "New name:",
                                value=display_name,
                                key=f"rename_input_{chat_id}",
                                label_visibility="collapsed"
                            )
                        # --- Load Button ---
                        else:
                            load_key = f"load_{chat_id}"
                            button_type = "primary" if is_current else "secondary"
                            help_text = f"Load '{display_name}' ({chat_meta.get('message_count', 0)} msgs, {chat_meta.get('model', '?')})"
                            if is_current: help_text = f"'{display_name}' (Current Chat)"

                            if st.button(f"{'▶️' if is_current else ''} {display_name}", key=load_key, use_container_width=True, help=help_text, type=button_type, disabled=is_renaming):
                                if not is_current:
                                    if st.session_state.messages: # Save previous before loading
                                        save_current_chat_to_file(show_toast=False)
                                    if load_chat_from_id(chat_id):
                                        st.session_state.renaming_chat_id = None # Ensure rename state is cleared
                                        st.rerun()
                                    # Else: load_chat_from_id would show an error

                    with col2:
                        # --- Save Rename Button ---
                        if is_renaming:
                            save_rename_key = f"save_rename_{chat_id}"
                            if st.button("✅", key=save_rename_key, help="Save new name", use_container_width=True):
                                new_name = st.session_state.get(f"rename_input_{chat_id}", display_name).strip()
                                if new_name and new_name != display_name:
                                    chat_data = load_chat_data(chat_id)
                                    if chat_data:
                                        chat_data["chat_name"] = new_name
                                        if save_specific_chat_data(chat_id, chat_data):
                                            st.toast(f"Renamed to '{new_name}'", icon="✏️")
                                            if is_current: st.session_state.current_chat_name = new_name # Update current name immediately
                                            st.session_state.renaming_chat_id = None
                                            st.rerun()
                                        # Else: save_specific_chat_data shows error
                                    else: st.error("Could not load chat data to rename.")
                                elif new_name == display_name: # No change, just cancel
                                    st.session_state.renaming_chat_id = None
                                    st.rerun()
                                else: st.warning("Enter a valid name.", icon="⚠️")
                        # --- Rename Icon Button ---
                        else:
                             rename_icon_key = f"rename_icon_{chat_id}";
                             if st.button("✏️", key=rename_icon_key, help="Rename this chat", use_container_width=True, disabled=is_current and st.session_state.renaming_chat_id is not None): # Disable if another rename is active
                                 st.session_state.renaming_chat_id = chat_id
                                 st.rerun()

                    with col3:
                        # --- Cancel Rename Button ---
                        if is_renaming:
                             cancel_rename_key = f"cancel_rename_{chat_id}";
                             if st.button("❌", key=cancel_rename_key, help="Cancel rename", use_container_width=True):
                                 st.session_state.renaming_chat_id = None
                                 st.rerun()
                        # --- Delete Button ---
                        else:
                             delete_key = f"delete_{chat_id}"
                             if st.button("🗑️", key=delete_key, help=f"Delete chat '{display_name}'", use_container_width=True, disabled=is_renaming):
                                 current_chat_was_deleted = (chat_id == st.session_state.current_chat_id)
                                 if delete_chat_file(chat_id): # Shows toast on success
                                     # If deleted chat was the current one, load another or start new
                                     if current_chat_was_deleted:
                                         remaining_chats = list_saved_chats() # Get updated list
                                         loaded_new = False
                                         if remaining_chats:
                                             if load_chat_from_id(remaining_chats[0]['id']):
                                                 st.success("Deleted current chat, loaded most recent.", icon="✨")
                                                 loaded_new = True
                                             else:
                                                 print("Error loading most recent chat after deletion.") # Fallback to new
                                         if not loaded_new:
                                             reset_chat_session_state()
                                             save_current_chat_to_file(show_toast=False)
                                             st.success("Deleted last chat, started new one.", icon="✨")
                                     st.rerun() # Rerun to update the list display

    st.divider()

    # --- File Uploader ---
    with st.expander("📎 Attach File", expanded=False):
        # Use a changing key based on response_count to allow re-uploading the same file
        uploader_key = f"file_uploader_sidebar_{st.session_state.response_count}"
        uploaded_file = st.file_uploader(
            "Upload Image or PDF",
            type=["png", "jpg", "jpeg", "webp", "gif", "pdf"], # Added more image types
            key=uploader_key,
            help="Attach a file to your *next* message. Upload will clear after message is sent."
        )

    st.divider()
    # --- App Settings ---
    st.checkbox(
        "Auto-load last chat on startup",
        key="autoload_last_chat",
        value=st.session_state.autoload_last_chat, # Bind directly to state
        help="Load the most recent chat when the app starts."
    )

# --- Main Chat Interface ---
st.title(f"Ollama Chat: {st.session_state.current_chat_name}")

# --- Display Chat Messages ---
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.get("messages", [])):
        role = message.get("role", "user")
        with st.chat_message(role):
            # Display text content
            if "content" in message and message["content"]:
                st.markdown(message["content"])
                # Add copy button for assistant messages
                if role == "assistant":
                     try:
                         # Make sure you have installed: pip install streamlit-copy-to-clipboard
                         from st_copy_to_clipboard import st_copy_to_clipboard
                         copy_key=f"copy_{st.session_state.current_chat_id}_{i}"
                         # Call without the unsupported 'label' argument
                         st_copy_to_clipboard(message["content"], key=copy_key) # <--- CORRECTED: Removed label
                     except ImportError:
                         if i == 0: # Show warning only once if needed
                            st.caption("Install `streamlit-copy-to-clipboard` for copy buttons.")
                     except Exception as e_copy:
                         print(f"Error rendering copy button: {e_copy}") # Log other errors

            # Display image if present in the message (using saved base64 data)
            if message.get("image_data"):
                 try:
                     # Use a reasonable width for display
                     st.image(base64.b64decode(message["image_data"]),
                              caption=message.get("image_filename","Attached Image"),
                              width=300)
                 except Exception as e_img:
                     st.error(f"Error displaying image '{message.get('image_filename')}': {e_img}")

            # Display PDF info if present
            if message.get("pdf_info"):
                st.info(f"📎 Context included from: {message['pdf_info']}")


# --- Chat Input Logic ---
prompt = st.chat_input(
    "Ask Ollama..." if st.session_state.ollama_reachable else "Ollama connection issue...",
    disabled=not st.session_state.ollama_reachable,
    key="chat_input_main"
)

if prompt:
    # If user types while a chat is being renamed, cancel the rename
    if st.session_state.renaming_chat_id:
        st.session_state.renaming_chat_id = None
        st.rerun() # Rerun briefly to update sidebar before processing prompt

    # --- File Handling ---
    temp_file_path_to_delete = None
    processed_file_data = None
    # Construct the key used in the sidebar uploader to get the file
    uploader_key = f"file_uploader_sidebar_{st.session_state.response_count}"
    current_uploaded_file = st.session_state.get(uploader_key)

    if current_uploaded_file:
        print(f"Processing uploaded file: {current_uploaded_file.name}")
        # Process the file, getting back temp path and data dict (or None on failure)
        temp_file_path_to_delete, processed_file_data = handle_file_upload(current_uploaded_file)

        if not processed_file_data:
            # Error/Warning is shown by handle_file_upload
            # Clean up temp file immediately if processing failed but file was created
            if temp_file_path_to_delete and os.path.exists(temp_file_path_to_delete):
                try:
                    os.unlink(temp_file_path_to_delete)
                    print(f"Cleaned up failed temp file: {temp_file_path_to_delete}")
                except Exception as e_del:
                    print(f"Error deleting failed temp file: {e_del}")
            temp_file_path_to_delete = None # Reset path as processing failed
        # We don't clear the uploader state here; it happens via key change on rerun

    # --- Prepare User Message for State & API ---
    user_message_for_state = {"role": "user"}
    user_prompt_text = prompt.strip() # The text typed by the user

    # Append file info/content to the user message if a file was processed
    if processed_file_data:
        filename = processed_file_data['filename']
        file_type = processed_file_data['type']

        if file_type == "image":
            # Store base64 data and filename in the message dict for state/API/display
            user_message_for_state["image_data"] = processed_file_data["data"]
            user_message_for_state["image_filename"] = filename
            # Add a default prompt if user only uploaded an image
            if not user_prompt_text:
                user_prompt_text = f"(Image attached: '{filename}') Please describe or analyze the image."

        elif file_type == "pdf":
            pdf_text = processed_file_data['data']
            # Store filename in the message dict for display purposes
            user_message_for_state["pdf_info"] = filename
            # Augment the user's text prompt with PDF content and instructions
            pdf_context_header = f"--- Start of content from '{filename}' ---\n{pdf_text}\n--- End of content from '{filename}' ---"
            # Limit context sent? Very long PDFs might exceed model limits or slow things down
            # Consider truncating pdf_text if necessary or summarizing it first.
            MAX_PDF_CHARS = 15000 # Example limit - adjust as needed
            if len(pdf_text) > MAX_PDF_CHARS:
                 print(f"Warning: PDF text truncated to {MAX_PDF_CHARS} chars for context.")
                 pdf_text = pdf_text[:MAX_PDF_CHARS] + "... (truncated)"
                 pdf_context_header = f"--- Start of truncated content from '{filename}' ---\n{pdf_text}\n--- End of truncated content from '{filename}' ---"

            if not user_prompt_text: # If only PDF is uploaded, ask for summary
                user_prompt_text = (
                    f"(PDF attached: '{filename}') Please summarize the main points or provide an overview of the document content provided below.\n\n"
                    f"{pdf_context_header}"
                )
            else: # If user asked a question, provide context and the question
                user_prompt_text = (
                    f"Please answer the following question based *only* on the provided text from the document '{filename}'. Do not use outside knowledge.\n\n"
                    f"{pdf_context_header}\n\n"
                    f"User Question: {user_prompt_text}" # Append original question
                )
            print(f"Augmented prompt with PDF content for {filename} (~{len(pdf_text)} chars).")

    # Add the final text content (potentially augmented) to the message object
    user_message_for_state["content"] = user_prompt_text

    # Add the complete user message dictionary to session state immediately
    st.session_state.messages.append(user_message_for_state)
    save_current_chat_to_file() # Auto-save after adding user message

    # --- Prepare API Payload using /api/chat ---
    messages_for_api = []

    # 1. Add System Prompt if defined and non-empty
    if st.session_state.system_instruction and st.session_state.system_instruction.strip():
        messages_for_api.append({"role": "system", "content": st.session_state.system_instruction.strip()})

    # 2. Add all messages from history (including the one just added)
    for msg in st.session_state.messages:
        # Basic structure for API message
        # Ensure content is a string, even if empty
        api_msg = {"role": msg["role"], "content": str(msg.get("content", ""))}

        # Add image data *only* to user messages that have it (API format)
        if msg["role"] == "user" and msg.get("image_data"):
            api_msg["images"] = [msg["image_data"]] # API expects a list of base64 strings
        messages_for_api.append(api_msg)
        # Note: PDF text is now embedded within the 'content' of the relevant user message

    # 3. Construct the final payload for /api/chat
    payload = {
        "model": st.session_state.model,
        "messages": messages_for_api,
        "stream": True,
        "options": {
            "temperature": st.session_state.temperature
            # Add other Ollama options here if needed, e.g.:
            # "top_p": 0.9,
            # "num_predict": 1024, # Max tokens
        }
    }

    # ---vvv DEBUG PAYLOAD (ENABLED BY DEFAULT) vvv---
    st.warning("Debugging: Payload to be sent to /api/chat", icon="🐞")
    try:
         st.json(payload, expanded=False) # Display in Streamlit app, collapsed by default
    except Exception as e_json_disp:
         st.text(f"Error displaying payload: {e_json_disp}")
         print("--- PAYLOAD (Console Fallback) ---")
         try: print(json.dumps(payload, indent=2))
         except Exception as e_json_print: print(f"Error printing payload: {e_json_print}")
         print("--- END PAYLOAD ---")
    # ---^^^ DEBUG PAYLOAD ^^^---

    # --- Call API and Handle Response ---
    try:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking... 💭") # Initial placeholder
            full_response = ""
            response = None # Initialize response to None

            # ** API CALL using /api/chat **
            start_time = time.time()
            print(f"Sending request to {OLLAMA_BASE_URL}/api/chat with model {st.session_state.model}")
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat", # Use the chat endpoint
                json=payload,
                stream=True,
                timeout=300 # Increased timeout for potentially long responses
            )
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

            # --- Stream Processing ---
            for line in response.iter_lines():
                if line:
                    try:
                        part = json.loads(line.decode('utf-8'))
                        # Extract content from the 'message' dictionary in the response part
                        content_part = part.get("message", {}).get("content", "") # <--- Parsing for /api/chat
                        if content_part: # Only append if there's content
                           full_response += content_part
                           # Update placeholder with streaming content + cursor simulation
                           message_placeholder.markdown(full_response + " ▌")
                        # Check the 'done' flag in the response part
                        # Note: Ollama's 'done' flag might differ slightly between versions
                        # Check if the 'message' structure exists and if 'done' is true at the top level
                        if part.get("done") and part.get("message", {}).get("content") is None:
                             if part.get("total_duration"): # Check for final metrics as another indicator
                                 print(f"Stream done flag received. Metrics: {part.get('total_duration')}")
                                 break # Exit loop when Ollama signals completion
                             else:
                                 # Sometimes 'done' is true but there's a final empty message part?
                                 # Let's break cautiously
                                 print("Stream 'done' flag received without metrics, potentially final part.")
                                 break

                    except json.JSONDecodeError:
                        print(f"Warning: JSON Decode Error on stream line: {line}")
                    except Exception as e_stream:
                        print(f"Error processing stream line: {e_stream}")
                        full_response += "\n*(Stream Error)*" # Append error to output
                        break # Stop processing stream on error

            end_time = time.time()
            print(f"Stream finished in {end_time - start_time:.2f} seconds. Full response length: {len(full_response)}")
            message_placeholder.markdown(full_response) # Display final complete response

    # --- ERROR HANDLING ---
    except requests.exceptions.ConnectionError as e:
        st.error(f"Connection Error: Could not connect to Ollama at {OLLAMA_BASE_URL}. Is it running?")
        print(f"Connection Error Detail: {e}")
        st.session_state.ollama_reachable = False # Update connection state
        full_response = "*Connection Error*" # Set error message for state
        if 'message_placeholder' in locals(): message_placeholder.error("Connection Error")
        if _remove_last_user_message_on_error(): st.rerun() # Remove user msg and refresh UI
        update_connection_status() # Update sidebar status display
    except requests.exceptions.Timeout:
        st.error("Request timed out. The Ollama server might be busy or the request took too long.")
        full_response = "*Request Timed Out.*"
        if 'message_placeholder' in locals(): message_placeholder.error("Request Timed Out")
        if _remove_last_user_message_on_error(): st.rerun()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API Error: {e}")
        full_response = f"*API Error: {e}*"
        error_detail = "(Could not retrieve details)"
        if e.response is not None:
             try:
                 status_code = e.response.status_code
                 error_content = e.response.text
                 try: # Try parsing Ollama's JSON error format
                     error_json = json.loads(error_content)
                     error_detail = f"(Status {status_code}) {error_json.get('error', error_content)}"
                 except json.JSONDecodeError:
                     error_detail = f"(Status {status_code}) {error_content}" # Use raw text if not JSON
             except Exception as e_resp:
                 error_detail = f"(Error retrieving details: {e_resp})"
        else: error_detail = "(No response object in exception)"

        st.error(f"Details: {error_detail}")
        full_response += f"\nDetails: {error_detail}"
        if 'message_placeholder' in locals(): message_placeholder.error(f"API Error: Check console/Ollama logs. {error_detail[:200]}...")
        if _remove_last_user_message_on_error(): st.rerun()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        full_response = f"*Unexpected Error: {e}*"
        if 'message_placeholder' in locals(): message_placeholder.error(f"Unexpected Error: {e}")
        if _remove_last_user_message_on_error(): st.rerun()

    # --- FINALLY Block / Cleanup ---
    finally:
        # Add assistant response to state ONLY if successful and not an error placeholder
        if full_response and not full_response.startswith("*"):
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # Auto-save after adding assistant message
            save_current_chat_to_file()

        # Clean up the temporary file if one was created during upload handling
        if temp_file_path_to_delete and os.path.exists(temp_file_path_to_delete):
            try:
                os.unlink(temp_file_path_to_delete)
                print(f"Deleted processed temp file: {temp_file_path_to_delete}")
            except Exception as e_del:
                print(f"Error deleting processed temp file {temp_file_path_to_delete}: {e_del}")

        # Increment response count to change file uploader key, effectively clearing it on rerun
        st.session_state.response_count += 1

    # Rerun to display the new assistant message and update UI (e.g., clear file uploader)
    st.rerun()


# --- Footer Buttons ---
st.divider()
if st.button("🗑️ Clear Current Chat Messages", key="clear_chat_main_area"):
    if st.session_state.messages:
        st.session_state.messages = []
        st.session_state.response_count = 0 # Reset uploader key too
        save_current_chat_to_file() # Save the cleared state
        st.toast("Current chat messages cleared.", icon="🧹")
    st.rerun()