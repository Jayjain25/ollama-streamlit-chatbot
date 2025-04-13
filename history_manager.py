import os
import json
import uuid
import streamlit as st
from datetime import datetime
from config import CHATS_DIR

def get_chats_dir():
    """Gets the chat directory path and creates it if it doesn't exist."""
    if not os.path.exists(CHATS_DIR):
        try:
            os.makedirs(CHATS_DIR)
        except OSError as e:
            st.error(f"Failed to create chat directory '{CHATS_DIR}': {e}", icon="🚨")
            return None
    return CHATS_DIR

def generate_chat_id():
    """Generates a unique chat ID."""
    return str(uuid.uuid4())

def save_chat(chat_state):
    """
    Saves the current chat state to a JSON file.
    Overwrites the file if it already exists based on chat_id.
    """
    chats_dir = get_chats_dir()
    if not chats_dir or not chat_state.get("chat_id"):
        st.warning("Cannot save chat: Invalid directory or missing chat ID.")
        return False

    chat_id = chat_state["chat_id"]
    file_path = os.path.join(chats_dir, f"{chat_id}.json")

    # Ensure essential keys exist
    data_to_save = {
        "chat_id": chat_id,
        "chat_name": chat_state.get("chat_name", f"Chat {chat_id[:8]}"),
        "messages": chat_state.get("messages", []),
        "model_settings": chat_state.get("model_settings", {}),
        "uploaded_files_info": chat_state.get("uploaded_files_info", {}), # Save metadata
        "rag_enabled": chat_state.get("rag_enabled", False),
        "web_search_enabled": chat_state.get("web_search_enabled", False),
        "created_at": chat_state.get("created_at", datetime.now().isoformat()),
        "last_updated": datetime.now().isoformat(),
    }

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)
        # print(f"Chat '{data_to_save['chat_name']}' ({chat_id}) saved successfully.") # Debug
        return True
    except IOError as e:
        st.error(f"Error saving chat '{data_to_save['chat_name']}': {e}", icon="💾")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while saving chat: {e}", icon="💥")
        return False

def load_chat(chat_id):
    """Loads a chat session from a JSON file."""
    chats_dir = get_chats_dir()
    if not chats_dir:
        return None

    file_path = os.path.join(chats_dir, f"{chat_id}.json")
    if not os.path.exists(file_path):
        st.error(f"Chat file not found: {chat_id}.json", icon="❓")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
            # Basic validation (can be expanded)
            if "chat_id" not in chat_data or "messages" not in chat_data:
                 st.error(f"Invalid chat file format: {chat_id}.json", icon=" L ")
                 return None
            # print(f"Chat '{chat_data.get('chat_name', chat_id)}' loaded.") # Debug
            return chat_data
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from chat file: {chat_id}.json", icon=" L ")
        return None
    except IOError as e:
        st.error(f"Error loading chat file '{chat_id}.json': {e}", icon=" L ")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading chat: {e}", icon="💥")
        return None


def list_chats():
    """Lists available chat sessions, sorted by modification time (newest first)."""
    chats_dir = get_chats_dir()
    if not chats_dir:
        return []

    chat_files = []
    try:
        for filename in os.listdir(chats_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(chats_dir, filename)
                try:
                    # Get modification time
                    mtime = os.path.getmtime(file_path)
                    # Try to read chat name from JSON for better display
                    chat_name = filename[:-5] # Fallback to ID
                    with open(file_path, 'r', encoding='utf-8') as f:
                         try:
                             data = json.load(f)
                             chat_name = data.get("chat_name", chat_name)
                         except json.JSONDecodeError:
                             pass # Keep fallback name if JSON is broken
                    chat_files.append({"id": filename[:-5], "name": chat_name, "mtime": mtime})
                except OSError:
                    continue # Skip files that can't be accessed

        # Sort by modification time, newest first
        chat_files.sort(key=lambda x: x["mtime"], reverse=True)
        return chat_files

    except OSError as e:
        st.error(f"Error listing chats in '{chats_dir}': {e}", icon=" L ")
        return []

def delete_chat(chat_id):
    """Deletes a chat session file."""
    chats_dir = get_chats_dir()
    if not chats_dir:
        return False

    file_path = os.path.join(chats_dir, f"{chat_id}.json")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            # print(f"Chat {chat_id} deleted.") # Debug
            return True
        except OSError as e:
            st.error(f"Error deleting chat file '{chat_id}.json': {e}", icon="🗑️")
            return False
    else:
        st.warning(f"Chat file not found for deletion: {chat_id}.json")
        return False

def rename_chat(chat_id, new_name):
    """Renames a chat by updating its name in the JSON file."""
    chats_dir = get_chats_dir()
    if not chats_dir or not new_name:
        return False

    file_path = os.path.join(chats_dir, f"{chat_id}.json")
    if not os.path.exists(file_path):
        st.error(f"Chat file not found for renaming: {chat_id}.json", icon="❓")
        return False

    try:
        # Read existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)

        # Update name and timestamp
        chat_data["chat_name"] = new_name.strip()
        chat_data["last_updated"] = datetime.now().isoformat()

        # Write updated data back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=4)
        # print(f"Chat {chat_id} renamed to '{new_name}'.") # Debug
        return True

    except (IOError, json.JSONDecodeError) as e:
        st.error(f"Error renaming chat '{chat_id}': {e}", icon="✏️")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during rename: {e}", icon="💥")
        return False