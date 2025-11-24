"""
Gradio Chat Interface for the RAG Chatbot.

This module provides a web-based chat UI that communicates with the FastAPI
backend to answer user questions using retrieval-augmented generation.
"""

import gradio as gr
import requests

# ------------------ CONFIG ------------------
# FastAPI endpoint URL - uses localhost since both services run in the same container
FASTAPI_URL = "http://127.0.0.1:8000/ask"


# ------------------ CHAT FUNCTION ------------------
def ask_llm(message, history):
    """
    Send a user message to the FastAPI RAG endpoint and return the response.

    Handles conversation history by converting Gradio's format to the API's
    expected format. Supports Gradio 4.x, 5.x, and 6.x history formats.

    Args:
        message: The current user message.
        history: Conversation history from Gradio (format varies by version).

    Returns:
        The assistant's response string, or an error message if the request fails.
    """
    try:
        # Convert Gradio history to API format: [[user_msg, assistant_msg], ...]
        formatted_history = []

        # Debug: log the history format (visible in container logs)
        print(f"[DEBUG] History type: {type(history)}, length: {len(history) if history else 0}")
        if history:
            print(f"[DEBUG] First item type: {type(history[0])}, value: {history[0]}")

        if history:
            first_item = history[0]

            # Gradio 5.x/6.x format: list of dicts with "role" and "content" keys
            if isinstance(first_item, dict):
                user_msg = None
                for item in history:
                    if item.get("role") == "user":
                        user_msg = item.get("content", "")
                    elif item.get("role") == "assistant" and user_msg is not None:
                        formatted_history.append([user_msg, item.get("content", "")])
                        user_msg = None
            # Gradio 4.x format: list of [user_msg, assistant_msg] pairs
            elif isinstance(first_item, (list, tuple)):
                formatted_history = [[str(h[0]), str(h[1])] for h in history if len(h) >= 2]
            # Handle any other format by trying to convert to string
            else:
                print(f"[DEBUG] Unknown history format: {first_item}")

        print(f"[DEBUG] Formatted history: {formatted_history}")

        # Send request to FastAPI backend
        resp = requests.post(FASTAPI_URL, json={"question": message, "history": formatted_history})
        data = resp.json()

        answer = data.get("answer", "No answer returned.")
        return answer

    except Exception as e:
        print(f"[ERROR] ask_llm failed: {e}")
        return f"❌ Error: {e}"


# ------------------ GRADIO UI ------------------
# Create the chat interface with custom title and description
chat = gr.ChatInterface(
    fn=ask_llm,
    title="Your Internal Assistant",
    description="Ask me anything",
)


if __name__ == "__main__":
    # Launch the Gradio server
    # server_name="0.0.0.0" makes it accessible from outside the container
    chat.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
