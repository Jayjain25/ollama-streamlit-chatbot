import streamlit as st
import time
import os
import json
from datetime import datetime
# Correct Copy Button Import
from st_copy_to_clipboard import st_copy_to_clipboard

# Import functions from other modules
from config import (
    DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P, DEFAULT_TOP_K, ALLOWED_FILE_TYPES, CHATS_DIR
)
from ollama_client import (
    check_ollama_connection, get_available_models, get_ollama_stream
)
from history_manager import (
    save_chat, load_chat, list_chats, delete_chat, rename_chat, generate_chat_id, get_chats_dir
)
from file_handler import (
    process_uploaded_files, initialize_rag_store, add_text_to_rag, rag_texts_store
)
from chat_logic import (
    generate_chat_title, prepare_context_and_messages
)
# Import template manager functions
from template_manager import (
    load_templates, add_or_update_template, delete_template
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Ollama Chat Advanced",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Function Definitions ---

def initialize_session_state():
    """Initializes Streamlit session state variables."""
    # Basic Chat State
    if "messages" not in st.session_state: st.session_state.messages = []
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = generate_chat_id()
        st.session_state.chat_name = "New Chat"
        st.session_state.created_at = datetime.now().isoformat()
        st.session_state.last_saved_message_count = 0
    # Model Config
    if "selected_model" not in st.session_state: st.session_state.selected_model = DEFAULT_MODEL
    if "system_prompt" not in st.session_state: st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    # --- ADD Persona State ---
    if "persona" not in st.session_state: st.session_state.persona = "" # Initialize persona
    # --- END Persona State ---
    if "temperature" not in st.session_state: st.session_state.temperature = DEFAULT_TEMPERATURE
    if "top_p" not in st.session_state: st.session_state.top_p = DEFAULT_TOP_P
    if "top_k" not in st.session_state: st.session_state.top_k = DEFAULT_TOP_K
    if "seed" not in st.session_state: st.session_state.seed = None
    # Advanced Features State
    if "uploaded_files_info" not in st.session_state: st.session_state.uploaded_files_info = {}
    if "pdf_texts" not in st.session_state: st.session_state.pdf_texts = {}
    if "rag_enabled" not in st.session_state: st.session_state.rag_enabled = False
    if "web_search_enabled" not in st.session_state: st.session_state.web_search_enabled = False
    if "pending_images" not in st.session_state: st.session_state.pending_images = {}
    # UI State
    if "ollama_status" not in st.session_state: st.session_state.ollama_status = ("unknown", "Checking...")
    if "available_models" not in st.session_state: st.session_state.available_models = []
    if "file_uploader_key" not in st.session_state: st.session_state.file_uploader_key = 0
    # History Management State
    if "saved_chats" not in st.session_state: st.session_state.saved_chats = []
    if "search_query" not in st.session_state: st.session_state.search_query = ""
    if "show_rename_input" not in st.session_state: st.session_state.show_rename_input = {}
    # Title Generation State
    if "title_generated" not in st.session_state: st.session_state.title_generated = False
    # --- ADD Template State ---
    if "prompt_templates" not in st.session_state: st.session_state.prompt_templates = load_templates()
    if "selected_template_name" not in st.session_state: st.session_state.selected_template_name = "Custom" # Default to custom
    if "new_template_name" not in st.session_state: st.session_state.new_template_name = ""
    # --- END Template State ---


def reset_chat_session(clear_files=True):
    """Resets the session state to start a new chat."""
    st.session_state.messages = []
    st.session_state.chat_id = generate_chat_id()
    st.session_state.chat_name = "New Chat"
    st.session_state.created_at = datetime.now().isoformat()
    st.session_state.last_saved_message_count = 0
    st.session_state.title_generated = False
    st.session_state.rag_enabled = False
    st.session_state.web_search_enabled = False
    # Keep persona/template selected? Or reset? Let's keep for now.
    # st.session_state.persona = ""
    # st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    # st.session_state.selected_template_name = "Custom"

    if clear_files:
        st.session_state.uploaded_files_info = {}
        st.session_state.pdf_texts = {}
        st.session_state.pending_images = {}
        if st.session_state.chat_id in rag_texts_store:
             initialize_rag_store(st.session_state.chat_id)
        st.session_state.file_uploader_key += 1
    st.toast("New chat started!", icon="✨")


def load_chat_session(chat_id):
    """Loads a chat session from a file into the session state."""
    chat_data = load_chat(chat_id)
    if chat_data:
        st.session_state.messages = chat_data.get("messages", [])
        st.session_state.chat_id = chat_data.get("chat_id", generate_chat_id())
        st.session_state.chat_name = chat_data.get("chat_name", f"Chat {st.session_state.chat_id[:8]}")
        st.session_state.created_at = chat_data.get("created_at", datetime.now().isoformat())
        st.session_state.last_saved_message_count = len(st.session_state.messages)
        model_settings = chat_data.get("model_settings", {})
        st.session_state.selected_model = model_settings.get("model", DEFAULT_MODEL)
        # Load persona and prompt from saved chat settings
        st.session_state.system_prompt = model_settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        st.session_state.persona = model_settings.get("persona", "") # Load persona
        st.session_state.temperature = model_settings.get("temperature", DEFAULT_TEMPERATURE)
        st.session_state.top_p = model_settings.get("top_p", DEFAULT_TOP_P)
        st.session_state.top_k = model_settings.get("top_k", DEFAULT_TOP_K)
        st.session_state.seed = model_settings.get("seed", None)
        st.session_state.uploaded_files_info = chat_data.get("uploaded_files_info", {})
        st.session_state.pdf_texts = chat_data.get("pdf_texts", {})
        st.session_state.pending_images = {}
        st.session_state.rag_enabled = chat_data.get("rag_enabled", False)
        st.session_state.web_search_enabled = chat_data.get("web_search_enabled", False)
        st.session_state.title_generated = True

        # Reset template selection as chat might not match a saved template
        st.session_state.selected_template_name = "Custom"

        initialize_rag_store(st.session_state.chat_id)
        if st.session_state.pdf_texts:
             for filename, text in st.session_state.pdf_texts.items():
                 add_text_to_rag(st.session_state.chat_id, text, filename)
        st.session_state.file_uploader_key += 1
        st.toast(f"Chat '{st.session_state.chat_name}' loaded.", icon="📂")
    else:
        st.error(f"Failed to load chat {chat_id}.", icon=" L ")

def delete_chat_session(chat_id):
    """Deletes a chat session file and updates the UI."""
    chat_name_to_delete = next((c['name'] for c in st.session_state.saved_chats if c['id'] == chat_id), "Unknown")
    if delete_chat(chat_id):
        st.toast(f"Chat '{chat_name_to_delete}' deleted.", icon="🗑️")
        if st.session_state.chat_id == chat_id:
            reset_chat_session()
        else:
             st.session_state.saved_chats = list_chats()
    else:
        st.error(f"Failed to delete chat {chat_id}.", icon=" L ")

def auto_save_chat():
    """Saves the current chat state if conditions are met."""
    if not st.session_state.messages or len(st.session_state.messages) <= st.session_state.last_saved_message_count:
        return
    current_state = {k: v for k, v in st.session_state.items() if k in [
        "chat_id", "chat_name", "messages", "uploaded_files_info", "pdf_texts",
        "rag_enabled", "web_search_enabled", "created_at"]}
    current_state["model_settings"] = {
        "model": st.session_state.selected_model,
        "system_prompt": st.session_state.system_prompt,
        "persona": st.session_state.persona, # <-- Save persona
        "temperature": st.session_state.temperature,
        "top_p": st.session_state.top_p,
        "top_k": st.session_state.top_k,
        "seed": st.session_state.seed}
    if save_chat(current_state):
        st.session_state.last_saved_message_count = len(st.session_state.messages)
    else:
        st.warning("Failed to auto-save chat.", icon="⚠️")

# --- Template Management Callbacks ---
def _apply_template():
    """Callback to apply selected template to persona and system prompt."""
    selected_name = st.session_state.selected_template_name # Name selected in selectbox
    if selected_name == "Custom":
        # User wants to use custom inputs, don't change anything
        pass
    else:
        # Find the template data
        template_data = next((t for t in st.session_state.prompt_templates if t.get("template_name") == selected_name), None)
        if template_data:
            st.session_state.persona = template_data.get("persona", "")
            st.session_state.system_prompt = template_data.get("system_prompt", "")
            st.toast(f"Template '{selected_name}' applied.", icon="📝")
        else:
            st.warning(f"Could not find template data for '{selected_name}'.", icon="❓")
            # Optionally reset to Custom if template data is missing
            st.session_state.selected_template_name = "Custom"

def _save_current_template():
    """Saves the current Persona and System Prompt as a new template."""
    template_name = st.session_state.new_template_name.strip()
    if not template_name:
        st.warning("Please enter a name for the template.", icon="⚠️")
        return

    # Check for duplicates
    if any(t.get("template_name") == template_name for t in st.session_state.prompt_templates):
         # Ask for confirmation to overwrite
         # Using a simple warning for now, overwrite is default
         st.warning(f"Template '{template_name}' already exists. Overwriting.", icon="⚠️")
         # Could add a modal confirmation here later if needed

    new_template = {
        "template_name": template_name,
        "persona": st.session_state.persona,
        "system_prompt": st.session_state.system_prompt
    }

    if add_or_update_template(new_template):
        st.toast(f"Template '{template_name}' saved.", icon="💾")
        # Reload templates into state and update selection
        st.session_state.prompt_templates = load_templates()
        st.session_state.selected_template_name = template_name # Select the newly saved one
        st.session_state.new_template_name = "" # Clear the input field
    else:
        st.error("Failed to save template.", icon=" L ")

def _delete_selected_template():
     """Deletes the currently selected template."""
     template_name_to_delete = st.session_state.selected_template_name
     if template_name_to_delete == "Custom":
          st.warning("Cannot delete the 'Custom' setting.", icon="🚫")
          return

     # Add confirmation maybe? For now, direct delete.
     if delete_template(template_name_to_delete):
          st.toast(f"Template '{template_name_to_delete}' deleted.", icon="🗑️")
          # Reload templates and reset selection to Custom
          st.session_state.prompt_templates = load_templates()
          st.session_state.selected_template_name = "Custom"
          # Explicitly set persona/prompt back to default/empty after deleting?
          st.session_state.persona = ""
          st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
     else:
          # Error message handled by delete_template()
          pass


# --- Initialization ---
initialize_session_state()
get_chats_dir()

# --- Sidebar ---
with st.sidebar:
    st.title("Ollama Chat")
    st.markdown("---")

    # --- Ollama Status Expander ---
    with st.expander("Ollama Connection Status", expanded=True):
        ollama_conn_placeholder = st.empty()
        def update_ollama_status():
            status, message = check_ollama_connection()
            st.session_state.ollama_status = (status, message)
            if status: st.session_state.available_models = get_available_models()
            else: st.session_state.available_models = []
        status_color = "green" if st.session_state.ollama_status[0] else "red"
        ollama_conn_placeholder.markdown(f"<p style='margin-bottom: 0px;'><span style='color:{status_color}; font-size: 1.5em;'>●</span> {st.session_state.ollama_status[1]}</p>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Connection & Models"):
            with st.spinner("Checking Ollama connection..."): update_ollama_status()
        if st.session_state.ollama_status[0] == "unknown" and not st.session_state.available_models:
            update_ollama_status(); st.rerun()

    # --- Model Configuration Expander ---
    with st.expander("Model Configuration", expanded=True):
        if st.session_state.available_models:
            current_model = st.session_state.selected_model
            if current_model not in st.session_state.available_models:
                st.warning(f"Model '{current_model}' not found. Resetting...")
                default_exists = DEFAULT_MODEL in st.session_state.available_models
                st.session_state.selected_model = DEFAULT_MODEL if default_exists else (st.session_state.available_models[0] if st.session_state.available_models else "")
            st.selectbox("Select Model:", st.session_state.available_models, key="selected_model", index=st.session_state.available_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0)
        else:
            st.warning("No models loaded."); st.text_input("Selected Model (Unavailable):", value=st.session_state.selected_model, disabled=True)

        # --- Template Management Section (Moved before System Prompt/Persona) ---
        st.subheader("Prompt Template")
        template_options = ["Custom"] + [t.get("template_name", "Unnamed") for t in st.session_state.prompt_templates]
        st.selectbox(
            "Load Template:",
            options=template_options,
            key="selected_template_name", # Use the state variable directly
            on_change=_apply_template # Callback to apply selection
        )

        # Button to delete the currently selected template
        if st.session_state.selected_template_name != "Custom":
             if st.button(f"🗑️ Delete Template '{st.session_state.selected_template_name}'", use_container_width=True, type="secondary"):
                 _delete_selected_template()
                 st.rerun() # Rerun needed to update selectbox after delete

        st.markdown("---") # Separator

        # --- Persona Input ---
        st.text_input("Persona:", key="persona", placeholder="e.g., A friendly pirate expert")

        # --- System Prompt Input ---
        st.text_area("System Prompt / Instructions:", key="system_prompt", height=150, placeholder="You are a helpful assistant...")

        # --- Save Template Section ---
        with st.container(border=True): # Add border for visual grouping
            st.caption("Save Current Persona & Prompt")
            st.text_input("New Template Name:", key="new_template_name")
            if st.button("💾 Save as New Template", use_container_width=True):
                _save_current_template()
                st.rerun() # Rerun needed to update selectbox after save

        # --- Advanced Model Parameters ---
        st.subheader("Advanced Parameters")
        st.slider("Temperature:", 0.0, 2.0, key="temperature", step=0.1)
        st.slider("Top P:", 0.0, 1.0, key="top_p", step=0.05)
        st.slider("Top K:", 0, 100, key="top_k", step=1)
        seed_value = st.number_input("Seed (Optional):", value=st.session_state.seed if st.session_state.seed is not None else 0, placeholder="Enter seed...", step=1)
        st.session_state.seed = seed_value if seed_value != 0 else None

    # --- File Upload Expander ---
    with st.expander("File Upload & Management", expanded=False):
        # ... (File upload logic remains the same) ...
        uploaded_files = st.file_uploader("Attach Files (Image/PDF)", type=ALLOWED_FILE_TYPES, accept_multiple_files=True, key=f"file_uploader_{st.session_state.file_uploader_key}")
        if uploaded_files:
            processed_data = process_uploaded_files(uploaded_files)
            newly_processed_info = processed_data["processed_info"]; new_images_base64 = processed_data["images_base64"]; new_pdf_texts = processed_data["pdf_texts"]
            current_files = set(st.session_state.uploaded_files_info.keys())
            for filename, info in newly_processed_info.items():
                if filename not in current_files:
                    st.session_state.uploaded_files_info[filename] = info
                    if info["type"] == "pdf" and filename in new_pdf_texts:
                        st.session_state.pdf_texts[filename] = new_pdf_texts[filename]; add_text_to_rag(st.session_state.chat_id, new_pdf_texts[filename], filename); st.success(f"PDF Ready: {filename}", icon="📄")
                    elif info["type"] == "image" and filename in new_images_base64:
                        st.session_state.pending_images[filename] = new_images_base64[filename]; st.success(f"Image Ready: {filename}", icon="🖼️")
                    else: st.warning(f"File '{filename}' processed but couldn't be stored.", icon="❓")
        if st.session_state.uploaded_files_info:
             st.caption("Active files for this chat:")
             for fn, info in st.session_state.uploaded_files_info.items():
                  st.caption(f"- {fn} ({info['type']})")
        if st.session_state.pending_images:
             st.caption(f"{len(st.session_state.pending_images)} image(s) ready for next message.")


    # --- Chat Actions Expander ---
    with st.expander("Current Chat Actions", expanded=False):
        # ... (New Chat, Export Chat logic remains the same) ...
        if st.button("➕ New Chat", use_container_width=True): reset_chat_session(); st.rerun()
        if st.session_state.messages:
            export_state = { k: v for k, v in st.session_state.items() if k in ["chat_id", "chat_name", "messages", "uploaded_files_info", "pdf_texts", "rag_enabled", "web_search_enabled", "created_at"]}
            export_state["model_settings"] = { "model": st.session_state.selected_model, "system_prompt": st.session_state.system_prompt, "persona": st.session_state.persona, "temperature": st.session_state.temperature, "top_p": st.session_state.top_p, "top_k": st.session_state.top_k, "seed": st.session_state.seed} # Add persona
            export_state["exported_at"] = datetime.now().isoformat()
            st.download_button(label="💾 Export Current Chat", data=json.dumps(export_state, indent=4), file_name=f"chat_{st.session_state.chat_id}_{st.session_state.chat_name.replace(' ', '_')[:20]}.json", mime="application/json", use_container_width=True)


    # --- Saved Chats Expander ---
    with st.expander("Saved Chats", expanded=True):
        # ... (Saved Chats listing/loading/deleting logic remains the same) ...
        st.text_input("Search Chats:", key="search_query")
        if not hasattr(st.session_state, 'saved_chats_last_updated') or time.time() - st.session_state.saved_chats_last_updated > 5:
             st.session_state.saved_chats = list_chats(); st.session_state.saved_chats_last_updated = time.time()
        filtered_chats = [c for c in st.session_state.saved_chats if st.session_state.search_query.lower() in c['name'].lower() or st.session_state.search_query.lower() in c['id'].lower()]
        if not filtered_chats:
             st.write("No saved chats found." if not st.session_state.search_query else "No matching chats found.")
        else:
            with st.container(height=300):
                for chat_info in filtered_chats:
                    cid, cname = chat_info['id'], chat_info['name']
                    col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2])
                    with col1: # Load / Rename Input
                        if st.session_state.show_rename_input.get(cid, False):
                            new_name = st.text_input(f"RN_{cid}", value=cname, label_visibility="collapsed", key=f"ri_{cid}")
                            if st.button("💾", key=f"sv_{cid}", help="Save Name"):
                                if new_name and new_name != cname and rename_chat(cid, new_name):
                                    st.session_state.saved_chats = list_chats(); st.session_state.show_rename_input[cid] = False; st.rerun()
                                else: st.error("Failed to rename.")
                            if st.button("❌", key=f"cn_{cid}", help="Cancel Rename"):
                                 st.session_state.show_rename_input[cid] = False; st.rerun()
                        else:
                            dname = cname[:25] + "..." if len(cname) > 25 else cname
                            st.button(f"{dname}", key=f"ld_{cid}", on_click=load_chat_session, args=(cid,), help=f"Load: {cname}\nID: {cid}", use_container_width=True)
                    with col2: # Rename Btn
                        if not st.session_state.show_rename_input.get(cid, False):
                            if st.button("✏️", key=f"rn_{cid}", help="Rename Chat"):
                                st.session_state.show_rename_input = {k: False for k in st.session_state.show_rename_input}; st.session_state.show_rename_input[cid] = True; st.rerun()
                    with col3: # Fork Btn
                         if st.button("🐑", key=f"fk_{cid}", help="Fork/Duplicate Chat"):
                            data = load_chat(cid)
                            if data:
                                data['chat_id'] = generate_chat_id(); data['chat_name'] = f"Copy of {data['chat_name']}"; data['created_at'] = datetime.now().isoformat()
                                if save_chat(data): st.toast(f"Chat duplicated as '{data['chat_name']}'"); st.session_state.saved_chats = list_chats(); st.rerun()
                                else: st.error("Failed to duplicate chat.")
                    with col4: # Delete Btn
                        if st.button("🗑️", key=f"dl_{cid}", help="Delete Chat", on_click=delete_chat_session, args=(cid,)):
                            pass # Action happens in callback

# --- Main Chat Area (No Changes Needed Here) ---
st.header(f"Chat: {st.session_state.chat_name}")
st.markdown("---") # Separator

# --- Display Chat Messages ---
message_container = st.container()
with message_container:
    num_messages = len(st.session_state.messages)
    for i, message in enumerate(st.session_state.messages):
        is_user = message["role"] == "user"
        with st.chat_message(message["role"]):
            content_col, button_col = st.columns([0.9, 0.1]) # Adjust ratios if needed
            with content_col:
                st.markdown(message["content"])
                if is_user and "images_base64" in message and message["images_base64"]:
                     img_cols = st.columns(min(len(message["images_base64"]), 4))
                     for idx, img_b64 in enumerate(message["images_base64"]):
                         with img_cols[idx % 4]:
                             try: st.image(f"data:image/png;base64,{img_b64}", width=150)
                             except Exception as e: st.error(f"Img err: {e}", icon=" L ")
                if not is_user and message.get("context_used"):
                    st.caption(f"ℹ️ Context: {message['context_used']}")

            with button_col:
                button_container = st.container()
                with button_container:
                    st_copy_to_clipboard(message["content"], key=f"copy_{st.session_state.chat_id}_{i}")


# --- Chat Input Area ---
input_container = st.container()
with input_container:
    toggle_cols = st.columns(2)
    with toggle_cols[0]:
        rag_possible = bool(st.session_state.pdf_texts) or (st.session_state.chat_id in rag_texts_store and rag_texts_store[st.session_state.chat_id]["chunks"])
        st.toggle("🧠 Use RAG Context", key="rag_enabled", help="Retrieve context from uploaded PDFs.", disabled=not rag_possible)
        if not rag_possible and st.session_state.rag_enabled: st.session_state.rag_enabled = False
    with toggle_cols[1]:
         st.toggle("🌐 Perform Web Search", key="web_search_enabled", help="Search the web and add results to context.")

    prompt = st.chat_input("What's on your mind?", key="chat_input")


# --- Processing Logic (Runs when prompt is submitted) ---
if prompt:
    user_message = {"role": "user", "content": prompt}
    if st.session_state.pending_images:
        user_message["images_base64"] = list(st.session_state.pending_images.values())
        st.session_state.pending_images = {}
    st.session_state.messages.append(user_message)

    # --- Prepare System Prompt (Combine Persona + Instructions) ---
    final_system_prompt = st.session_state.system_prompt
    if st.session_state.persona and st.session_state.persona.strip():
         # Prepend persona if it exists
         final_system_prompt = f"**Persona:** {st.session_state.persona.strip()}\n\n**Instructions:**\n{st.session_state.system_prompt}"
         # Alternative: Just add persona to the instructions field if preferred by model
         # final_system_prompt = f"{st.session_state.persona.strip()}\n\n{st.session_state.system_prompt}"

    # --- Prepare API Messages ---
    api_messages_prep, context_description = prepare_context_and_messages(
        chat_id=st.session_state.chat_id,
        messages=st.session_state.messages[:-1],
        user_input=prompt,
        include_rag=st.session_state.rag_enabled,
        include_web_search=st.session_state.web_search_enabled
    )
    if "images_base64" in user_message:
        if api_messages_prep and api_messages_prep[-1]["role"] == "user":
            api_messages_prep[-1]["images"] = user_message["images_base64"]

    # --- Call Ollama API ---
    with message_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌")
            full_response = ""
            try:
                stream = get_ollama_stream( # Pass combined system prompt
                    st.session_state.selected_model, api_messages_prep, final_system_prompt,
                    st.session_state.temperature, st.session_state.top_p, st.session_state.top_k, st.session_state.seed
                )
                for chunk, is_done in stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)

                message_placeholder.markdown(full_response)
                st_copy_to_clipboard(full_response, key=f"copybtn_{st.session_state.chat_id}_assistant_{int(time.time())}")
                if context_description:
                    st.caption(f"ℹ️ Context: {context_description}")

            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.error(full_response, icon="🔥")

    # --- Update History & State ---
    assistant_message = {"role": "assistant", "content": full_response}
    if context_description:
        assistant_message["context_used"] = context_description
    st.session_state.messages.append(assistant_message)

    if not st.session_state.title_generated and len(st.session_state.messages) >= 2:
        with st.spinner("Generating title..."):
             new_title = generate_chat_title(st.session_state.messages)
             if new_title != "New Chat" and new_title != st.session_state.chat_name:
                  st.session_state.chat_name = new_title
                  st.session_state.title_generated = True

    auto_save_chat()
    st.rerun()