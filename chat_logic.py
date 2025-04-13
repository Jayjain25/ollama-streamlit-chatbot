import streamlit as st
from ollama_client import generate_text_ollama # Use generate for title
from config import TITLE_GENERATION_MODEL, MAX_TITLE_GENERATION_MESSAGES
from file_handler import get_rag_context
# Import the updated web search functions
from web_search import perform_web_search, format_search_results
import tiktoken

# --- Chat Logic ---

def generate_chat_title(messages):
    """Generates a concise title for the chat using the first few messages."""
    if not messages or len(messages) < 2 :
        return "New Chat"

    prompt_context = ""
    for i, msg in enumerate(messages[:MAX_TITLE_GENERATION_MESSAGES]):
        indicator = "User" if msg["role"] == "user" else "Assistant"
        content_preview = msg["content"][:150]
        prompt_context += f"{indicator}: {content_preview}\n"

    full_prompt = f"""Based on the following conversation start, suggest a very short, concise title (max 5 words) for this chat session. Only output the title, nothing else.

Conversation:
{prompt_context.strip()}

Title:"""

    title = generate_text_ollama(
        model=TITLE_GENERATION_MODEL,
        prompt=full_prompt,
        temperature=0.1
    )

    if title:
        title = title.strip().replace('"', '').replace('*', '')
        if title.lower().startswith("title:"):
            title = title[len("title:") :].strip()
        if len(title) > 50 or '\n' in title:
             return f"Chat: {messages[0]['content'][:30]}..."
        return title
    else:
        return f"Chat: {messages[0]['content'][:30]}..."


def prepare_context_and_messages(chat_id, messages, user_input, include_rag=False, include_web_search=False):
    """
    Prepares the message list for the API call, potentially adding RAG/Web context.
    """
    context_description = ""
    augmented_input = user_input
    rag_context_str = ""
    web_context_str = "" # Initialize here

    # 1. Add RAG Context
    rag_sources = []
    if include_rag:
        # Pass the original user_input for retrieval, not the potentially augmented one
        rag_context_str, rag_sources = get_rag_context(chat_id, user_input)
        if rag_context_str:
            # We will prepend context later, just note it was used
            context_description += f"RAG context ({', '.join(rag_sources)})"
            print(f"Adding RAG context to prompt for chat {chat_id}")

    # 2. Add Web Search Context
    if include_web_search:
        # Pass the original user_input for search
        search_results = perform_web_search(user_input) # Calls the updated function
        if search_results is None: # Handle complete search failure
            web_context_str = "\n\n[Web search failed]\n\n"
            if context_description: context_description += " + "
            context_description += "Web search failed"
        elif search_results: # If results were returned (even if empty list)
            web_context_str = format_search_results(search_results)
            if web_context_str: # Only add description if context was actually generated
                if context_description: context_description += " + "
                context_description += "Web search results"
                print(f"Adding Web Search context to prompt for chat {chat_id}")
        # If search_results is an empty list, web_context_str remains empty

    # 3. Construct the augmented input *after* retrieving all context
    final_augmented_input = user_input
    if web_context_str:
        final_augmented_input = f"{web_context_str}\n\n{final_augmented_input}" # Prepend web
    if rag_context_str:
        final_augmented_input = f"{rag_context_str}\n\n{final_augmented_input}" # Prepend RAG (so it appears first)

    # Add a clear marker if context was added
    if rag_context_str or web_context_str:
         final_augmented_input += f"\n\nUser Query: {user_input}" # Reiterate original query if context was added


    # 4. Construct the final message list for the API
    api_messages = messages.copy()

    # Create the new user message dict with the potentially augmented content
    new_user_message = {"role": "user", "content": final_augmented_input}

    # Add images IF they belong to the *original* user message being processed
    # This assumes the caller (app.py) adds images to the LAST message in the `messages` list passed here
    if messages and "images_base64" in messages[-1] and messages[-1]["role"] == "user":
         # This logic might be flawed if the user input we are processing isn't the last one.
         # It's safer for app.py to handle adding images directly to the API message list after this function returns.
         # Let's modify this function to NOT handle images, and let app.py do it.
         pass # Image handling moved to app.py

    api_messages.append(new_user_message)

    return api_messages, context_description


# --- Token Counting (Remains the same) ---
def estimate_token_count(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except ImportError:
        print("Warning: tiktoken not installed. Using basic character count for token estimation.")
        return len(text) // 4
    except Exception as e:
        print(f"Warning: tiktoken error ({e}). Using basic character count for token estimation.")
        return len(text) // 4

def get_context_token_estimate(messages, system_prompt=None, rag_context="", web_context=""):
    full_text = ""
    if system_prompt:
        full_text += system_prompt + "\n"
    for msg in messages:
        full_text += msg.get("content", "") + "\n"
    if rag_context:
        full_text += rag_context + "\n"
    if web_context:
        full_text += web_context + "\n"
    return estimate_token_count(full_text)