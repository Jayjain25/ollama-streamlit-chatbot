 # Streamlit Ollama Chat Interface

A Streamlit application providing a web interface to chat with local language models running via Ollama, featuring chat management and basic file context support (PDF text extraction, Image sending - requires multimodal Ollama model).

**Requires Ollama to be installed and running locally.** ([Download Ollama](https://ollama.com/))

## Features

*   **Chat with Local Models:** Interact with models served by your local Ollama instance.
*   **Model Management:**
    *   Select from available models pulled/running in Ollama.
    *   Switches between models (attempts to stop the previous one).
*   **Parameter Tuning:** Adjust `temperature`.
*   **System Instructions:** Set a system prompt to guide the model's behavior.
*   **File Context:**
    *   **PDF Text Extraction:** Upload a PDF; its text content is extracted and prepended to the user prompt.
    *   **Image Upload:** Upload an image; its base64 representation is sent with the prompt (**Requires a multimodal model like LLaVA running in Ollama**).
*   **Chat History:** View the conversation history within the app session.
*   **Chat Management:**
    *   **New Chat:** Start a fresh conversation.
    *   **Rename Chat:** Give the current chat session a name.
    *   **Save Chat:** Save the current chat history (including name and model) as a JSON file locally in a `chat_history` folder.
    *   **Load Chat:** Browse and load previously saved chat JSON files.
    *   **Delete Chat:** Remove saved chat files.

## Prerequisites

1.  **Ollama Installed and Running:** You *must* have Ollama installed and the `ollama serve` command running on your machine (or accessible via the network if you change the API URL). Download from [ollama.com](https://ollama.com/).
2.  **Ollama Models Pulled:** You need to have pulled the models you want to use within Ollama itself (e.g., `ollama run llama2`, `ollama pull mistral`, `ollama pull llava`). This app lists models available *in your Ollama instance*.

## Setup

1.  **Clone the repository (or save the files).**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Optional) Create `.gitignore`:** Add `venv/`, `__pycache__/`, `*.pyc`, `chat_history/` (if you don't want to commit saved chats).

## Running the App

1.  **Ensure Ollama is running:** Open a separate terminal and run `ollama serve`. Wait for it to indicate it's listening.
2.  **Run the Streamlit app:** In your project terminal (with the virtual environment activated), run:
    ```bash
    streamlit run app.py
    ```

Navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  **Select Model:** Choose an available model from the dropdown in the sidebar. These are models present in *your* local Ollama installation. Switching models will clear the current chat.
2.  **System Prompt:** Modify the system instructions if desired.
3.  **Temperature:** Adjust the model's creativity.
4.  **Upload Files (Optional):** Upload a PDF (text is used) or an image (requires a compatible multimodal model like LLaVA running in Ollama). The file content is associated with the *next* message you send.
5.  **Chat:** Type your prompt in the input box and press Enter.
6.  **Manage Chat:** Use the sidebar options to start a new chat, rename, save the current one, or load/delete previous chats saved locally in the `chat_history` folder.

## Important Notes

*   **Ollama Dependency:** This app is just a *frontend*. Ollama must be running locally (or wherever `http://localhost:11434` points) to handle the actual model inference.
*   **Multimodal Models:** Image support *only* works if the selected Ollama model is multimodal (like LLaVA) and capable of processing images sent in the specified payload format (`images: [base64_string]`). Standard text models will ignore image data.
*   **Model Switching:** The app attempts to stop the previous model before switching. This relies on the Ollama API and might not always be instantaneous or perfectly reliable. Chat history is cleared on model switch.
*   **Resource Usage:** Running large language models locally via Ollama can consume significant RAM, VRAM (if using GPU), and CPU resources.
*   **Error Handling:** Basic error handling for API connection issues is included.

## Disclaimer

This interface communicates with a locally running Ollama service. Ensure you understand the resource requirements and capabilities of the models you run via Ollama.