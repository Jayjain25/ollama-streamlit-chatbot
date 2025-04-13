import os
import json
import streamlit as st
from datetime import datetime

TEMPLATES_FILE = "prompt_templates.json"

def load_templates():
    """Loads prompt templates from the JSON file."""
    if not os.path.exists(TEMPLATES_FILE):
        return [] # Return empty list if file doesn't exist
    try:
        with open(TEMPLATES_FILE, 'r', encoding='utf-8') as f:
            templates_data = json.load(f)
            # Basic validation: ensure it's a list
            if isinstance(templates_data, list):
                 # Sort templates alphabetically by name for consistent display
                 templates_data.sort(key=lambda x: x.get("template_name", "").lower())
                 return templates_data
            else:
                 st.error(f"Invalid format in '{TEMPLATES_FILE}'. Expected a list.", icon=" L ")
                 return [] # Return empty on invalid format
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from '{TEMPLATES_FILE}'. File might be corrupted.", icon=" L ")
        return []
    except IOError as e:
        st.error(f"Error reading templates file '{TEMPLATES_FILE}': {e}", icon=" L ")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred loading templates: {e}", icon="💥")
        return []

def save_templates(templates_list):
    """Saves the entire list of templates to the JSON file."""
    try:
        # Sort before saving to maintain order
        templates_list.sort(key=lambda x: x.get("template_name", "").lower())
        with open(TEMPLATES_FILE, 'w', encoding='utf-8') as f:
            json.dump(templates_list, f, indent=4)
        return True
    except IOError as e:
        st.error(f"Error saving templates file '{TEMPLATES_FILE}': {e}", icon="💾")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred saving templates: {e}", icon="💥")
        return False

def add_or_update_template(new_template_data):
    """Adds a new template or updates an existing one by name."""
    if not new_template_data.get("template_name"):
        st.error("Template name cannot be empty.", icon="❗")
        return False

    templates = load_templates()
    name_to_find = new_template_data["template_name"]
    found = False

    # Add creation/update timestamps
    now = datetime.now().isoformat()
    new_template_data["last_updated"] = now
    if "created_at" not in new_template_data: # Keep original creation if updating
         new_template_data["created_at"] = now

    for i, template in enumerate(templates):
        if template.get("template_name") == name_to_find:
            templates[i] = new_template_data # Update existing
            found = True
            break

    if not found:
        templates.append(new_template_data) # Add new

    return save_templates(templates)


def delete_template(template_name_to_delete):
    """Deletes a template by its name."""
    if not template_name_to_delete:
        return False

    templates = load_templates()
    initial_length = len(templates)

    # Filter out the template to delete
    templates = [t for t in templates if t.get("template_name") != template_name_to_delete]

    if len(templates) < initial_length:
        return save_templates(templates)
    else:
        st.warning(f"Template '{template_name_to_delete}' not found for deletion.")
        return False # Indicate template not found