# Advanced Streamlit Ollama Chat Interface

This Streamlit application provides a feature-rich frontend for interacting with local Ollama models.

## Features

*   **Conversational Chat:** Chat with Ollama models, retaining conversation history within a session.
*   **Model Selection:** Dynamically lists and allows selection from models available on your Ollama instance.
*   **Parameter Tuning:** Adjust Temperature, Top P/K, Seed, and System Prompt.
*   **File Uploads:** Attach Images (PNG, JPG, WEBP) and PDFs to your chat session via the sidebar.
    *   Images are sent directly to multimodal models.
    *   PDF text content is extracted.
*   **RAG (Retrieval-Augmented Generation):** Toggle option to retrieve relevant chunks from uploaded PDFs and inject them as context into the LLM prompt. (Basic keyword-based retrieval implemented).
*   **Web Search:** Toggle option to perform a DuckDuckGo web search based on your query. Uses `requests` and `BeautifulSoup` to fetch and parse basic HTML content from top results, injecting summarized context into the LLM prompt.
*   **Chat History Management:**
    *   Chats are automatically saved to individual JSON files.
    *   Load, rename, search, duplicate (fork), and delete saved chat sessions via the sidebar.
    *   Export the current chat session as a JSON file.
*   **Ollama Connection Status:** Indicates if the app can connect to the Ollama server.
*   **State Persistence:** Maintains the current chat state during browser session refreshes. Starts fresh on initial load.
*   **Automatic Title Generation:** Automatically suggests a title for the chat after the first couple of messages.
*   **Copy Button:** Easily copy the content of any user or assistant message.
*   **RAG/Web Toggles:** Conveniently placed below the chat input area.

## Setup

1.  **Prerequisites:**
    *   Python 3.8+
    *   Ollama installed and running ([https://ollama.com/](https://ollama.com/))
    *   At least one Ollama model pulled (e.g., `ollama run llama3`)

2.  **Clone the repository (or download the files):**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
    *(If downloading manually, ensure all `.py` files, `requirements.txt`, and `README.md` are in the same folder)*

3.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install dependencies:**
    *   Requires `requests`, `beautifulsoup4`, `duckduckgo-search` for web search.
    *   Requires `st-copy-to-clipboard` for copy buttons.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Ollama URL (Optional):**
    *   The app defaults to `http://localhost:11434`.
    *   To use a different URL, create a `.env` file in the project root directory with the following content:
        ```dotenv
        OLLAMA_BASE_URL=http://your_ollama_host:port
        ```
    *   You can also optionally configure `CHATS_DIR` in the `.env` file to change the save location.

6.  **Create the chats directory:**
    ```bash
    mkdir chats
    ```
    *(The application will also attempt to create this directory if it doesn't exist)*

## Running the Application

1.  **Ensure Ollama is running.**
2.  **Run the Streamlit app from your terminal (make sure your virtual environment is active):**
    ```bash
    streamlit run app.py
    ```
3.  Open the URL provided by Streamlit (usually `http://localhost:8501`) in your web browser.

## Code Structure

*   `app.py`: Main Streamlit application, UI layout, state management orchestration.
*   `config.py`: Configuration constants (Ollama URL, chat directory, RAG settings, defaults).
*   `ollama_client.py`: Functions for direct interaction with the Ollama API endpoints (renamed from `ollama_api.py`).
*   `history_manager.py`: Functions for managing chat history JSON files (save, load, list, delete, rename).
*   `chat_logic.py`: Higher-level chat functions like title generation, context preparation (RAG/Web).
*   `file_handler.py`: Functions for processing uploaded files (PDF text extraction, image encoding, basic RAG implementation).
*   `web_search.py`: Web search integration logic using DuckDuckGo, requests, and BeautifulSoup.
*   `requirements.txt`: Python package dependencies.
*   `README.md`: This file.
*   `chats/`: Default directory where chat session JSON files are stored.

## Notes

*   **RAG Implementation:** The current RAG is very basic (keyword matching). For better results, consider integrating vector embeddings (e.g., using `sentence-transformers`) and a vector store (like FAISS or ChromaDB).
*   **Web Search:** Uses DuckDuckGo for search results and attempts to fetch/parse basic HTML content using `requests` and `BeautifulSoup`. Content extraction from JavaScript-heavy or complex websites might be limited or fail. Results are summarized/truncated before being added to the LLM context.
*   **Error Handling:** Basic error handling is included, but can be expanded for more robustness.
*   **Token Counting:** Uses `tiktoken` if available for better estimation, otherwise falls back to a character count heuristic. This may not perfectly match the specific tokenizer used by your Ollama model.
*   **UI Elements:** Feel free to customize the Streamlit UI further.