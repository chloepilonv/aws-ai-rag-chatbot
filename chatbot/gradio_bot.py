"""
Gradio Chat Interface for the RAG Chatbot with 3-star feedback.

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
FASTAPI_FEEDBACK_URL = os.getenv("FASTAPI_FEEDBACK_URL", "http://127.0.0.1:8000/feedback")
FASTAPI_CONVERSATION_URL = os.getenv("FASTAPI_CONVERSATION_URL", "http://127.0.0.1:8000/conversation/latest")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))  # seconds
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO)
logger = logging.getLogger(__name__)

# Store conversation IDs for feedback
conversation_ids = {}


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
        streaming_succeeded = False
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
                    streaming_succeeded = True

                    # After streaming completes, fetch the conversation ID
                    try:
                        conv_resp = requests.get(
                            FASTAPI_CONVERSATION_URL,
                            params={"question": message},
                            timeout=5
                        )
                        if conv_resp.status_code == 200:
                            conv_data = conv_resp.json()
                            conversation_id = conv_data.get("conversation_id")
                            if conversation_id:
                                conversation_ids[message] = conversation_id
                                if DEBUG:
                                    logger.debug(f"Stored conversation ID: {conversation_id}")
                    except Exception as conv_error:
                        logger.warning(f"Failed to fetch conversation ID: {conv_error}")

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
        if not streaming_succeeded:
            resp = requests.post(
                FASTAPI_URL,
                json={"question": message, "history": formatted_history},
                timeout=REQUEST_TIMEOUT,
            )
            data = resp.json()
            answer = data.get("answer", "No answer returned.")
            conversation_id = data.get("conversation_id")

            # Store conversation ID for feedback
            if conversation_id:
                conversation_ids[message] = conversation_id

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


# ------------------ FEEDBACK FUNCTION ------------------
def submit_feedback_click(rating, history):
    """
    Submit feedback for the last message in the conversation.

    Args:
        rating: -1 (bad), 0 (neutral), or 1 (good)
        history: Conversation history

    Returns:
        Status message
    """
    if not history or len(history) == 0:
        return "⚠️ No conversation to rate yet."

    # Get the last user message
    last_item = history[-1]
    if isinstance(last_item, dict):
        last_question = last_item.get("content", "") if last_item.get("role") == "user" else None
    elif isinstance(last_item, (list, tuple)) and len(last_item) > 0:
        last_question = last_item[0]
    else:
        return "⚠️ Could not identify last question."

    # Find conversation ID
    conv_id = conversation_ids.get(last_question)
    if not conv_id:
        return "⚠️ No conversation ID found. Please ask a question first."

    # Submit feedback
    try:
        resp = requests.post(
            FASTAPI_FEEDBACK_URL,
            json={"conversation_id": conv_id, "feedback": rating},
            timeout=10,
        )
        if resp.status_code == 200:
            stars = ["⭐", "⭐⭐", "⭐⭐⭐"][rating + 1]
            return f"✅ Thanks! Rated: {stars}"
        else:
            return f"❌ Failed to submit feedback: {resp.text}"
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return f"❌ Error: {e}"


# ------------------ GRADIO UI ------------------
with gr.Blocks(title="Your Internal Assistant", css="""
    .star-btn button {
        border: none !important;
        background: transparent !important;
        padding: 4px 8px !important;
        min-width: auto !important;
        font-size: 1.5em !important;
        cursor: pointer !important;
        opacity: 0.85 !important;
        transition: all 0.2s ease !important;
    }
    .star-btn button:hover {
        opacity: 1 !important;
        transform: scale(1.15);
    }
    .feedback-container {
        text-align: center;
        margin: 16px 0;
    }
""") as demo:
    gr.Markdown("# Votre assistant Qarnot")
    gr.Markdown("Posez-moi des questions sur la plateforme Qarnot / Ask me questions about Qarnot's platofrm. ")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question", placeholder="Type your question here...")

    # Wire up the chat
    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        user_message = history[-1][0]
        history[-1][1] = ""
        for partial_response in ask_llm_stream(user_message, history[:-1]):
            history[-1][1] = partial_response
            yield history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
