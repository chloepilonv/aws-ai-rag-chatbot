import gradio as gr
import requests

# FastAPI lives in the same container, so localhost is correct.
FASTAPI_URL = "http://127.0.0.1:8000/ask"

def ask_llm(message, history):
    """Calls your FastAPI RAG endpoint."""
    try:
        resp = requests.post(FASTAPI_URL, json={"question": message})
        data = resp.json()

        answer = data.get("answer", "No answer returned.")
        return answer

    except Exception as e:
        return f"❌ Error: {e}"


# Chat UI
chat = gr.ChatInterface(
    fn=ask_llm,
    title="Your Internal Assistant",
    description="Ask anything...",
)


if __name__ == "__main__":
    # IMPORTANT for Docker: listen on all interfaces, port 7860
    chat.launch(
        server_name="0.0.0.0",   # make Gradio reachable outside container
        server_port=7860
    )
