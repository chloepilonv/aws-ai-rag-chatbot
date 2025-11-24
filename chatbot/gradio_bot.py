"""
Gradio Chat Interface for the RAG Chatbot.

This module provides a web-based chat UI that communicates with the FastAPI
backend to answer user questions using retrieval-augmented generation.
"""

import os
import logging

import gradio as gr
import requests

# ------------------ CONFIG ------------------
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/ask")
FASTAPI_STREAM_URL = os.getenv("FASTAPI_STREAM_URL", "http://127.0.0.1:8000/ask/stream")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


# ------------------ HISTORY CONVERSION ------------------
def _convert_history(history):
    """
    Convert Gradio history to API format: [[user_msg, assistant_msg], ...]
    Supports Gradio 4.x, 5.x, and 6.x history formats.
    """
    if not history:
        return []

    formatted_history = []
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
    else:
        logger.warning(f"Unknown history format: {type(first_item)}")

    return formatted_history


# ------------------ STREAMING CHAT FUNCTION ------------------
def ask_llm_stream(message, history):
    """
    Send a user message to the FastAPI RAG endpoint with streaming response.

    Args:
        message: The current user message.
        history: Conversation history from Gradio.

    Yields:
        Partial response strings as they arrive.
    """
    try:
        formatted_history = _convert_history(history)

        if DEBUG:
            logger.debug(f"Question: {message}")
            logger.debug(f"History length: {len(formatted_history)}")

        # Try streaming endpoint first
        try:
            with requests.post(
                FASTAPI_STREAM_URL,
                json={"question": message, "history": formatted_history},
                timeout=REQUEST_TIMEOUT,
                stream=True,
            ) as resp:
                if resp.status_code == 200:
                    partial_response = ""
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            partial_response += chunk
                            yield partial_response
                    return
                elif resp.status_code == 404:
                    # Streaming endpoint not available, fall back to non-streaming
                    pass
                else:
                    yield f"❌ Error: Server returned status {resp.status_code}"
                    return
        except requests.exceptions.ConnectionError:
            # Streaming endpoint not available, fall back to non-streaming
            pass

        # Fall back to non-streaming endpoint
        resp = requests.post(
            FASTAPI_URL,
            json={"question": message, "history": formatted_history},
            timeout=REQUEST_TIMEOUT,
        )
        data = resp.json()
        answer = data.get("answer", "No answer returned.")
        yield answer

    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        yield "❌ Request timed out. Please try again."
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        yield "❌ Could not connect to the server. Please try again later."
    except Exception as e:
        logger.error(f"ask_llm failed: {e}")
        yield f"❌ Error: {e}"


# ------------------ GRADIO UI ------------------
chat = gr.ChatInterface(
    fn=ask_llm_stream,
    title="Your Internal Assistant",
    description="Ask me anything. We suggest to precise if you want to use qarnot with the HPC/Tasq Platform or with the Python SDK.",
)


if __name__ == "__main__":
    chat.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
