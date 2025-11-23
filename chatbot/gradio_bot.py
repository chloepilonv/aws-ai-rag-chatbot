import gradio as gr
import requests

# FastAPI lives in the same container, so localhost is correct.
FASTAPI_URL = "http://127.0.0.1:8000/ask"

def ask_llm(message, history):
    """Calls your FastAPI RAG endpoint with conversation history."""
    try:
        # Convert Gradio history to our format [[user_msg, assistant_msg], ...]
        formatted_history = []

        if history:
            # Gradio 5.x format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            if isinstance(history[0], dict):
                user_msg = None
                for item in history:
                    if item.get("role") == "user":
                        user_msg = item.get("content", "")
                    elif item.get("role") == "assistant" and user_msg is not None:
                        formatted_history.append([user_msg, item.get("content", "")])
                        user_msg = None
            # Gradio 4.x format: [[user_msg, assistant_msg], ...]
            elif isinstance(history[0], (list, tuple)):
                formatted_history = [[str(h[0]), str(h[1])] for h in history if len(h) >= 2]

        resp = requests.post(FASTAPI_URL, json={"question": message, "history": formatted_history})
        data = resp.json()

        answer = data.get("answer", "No answer returned.")
        return answer

    except Exception as e:
        return f"❌ Error: {e}"


# Chat UI
chat = gr.ChatInterface(
    fn=ask_llm,
    title="Your Internal Assistant",
    description="Make sure to specify if you are using the platforms (Tasq-HPC) or the Python SDK.",
)


if __name__ == "__main__":
    # IMPORTANT for Docker: listen on all interfaces, port 7860
    chat.launch(
        server_name="0.0.0.0",   # make Gradio reachable outside container
        server_port=7860
    )
