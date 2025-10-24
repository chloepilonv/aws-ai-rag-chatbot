import gradio as gr
import requests

# Point this to your running FastAPI service
FASTAPI_URL = "http://localhost:8000/ask"

def ask_llm(message, history):
    """Calls your FastAPI RAG endpoint."""
    try:
        resp = requests.post("http://localhost:8000/ask", json={"question": message})
        data = resp.json()
        answer = data.get("answer", "No answer returned.")
        cites = data.get("citations", [])
        if cites:
            answer += "\n\n**Sources:**\n" + "\n".join(
                f"- [{c['label']}]({c['url']})" for c in cites
            )
        return answer
    except Exception as e:
        return f"❌ Error: {e}"

# Simple chat interface
chat = gr.ChatInterface(
    fn=ask_llm,
    title="Your Internal Assistant",
    description="Ask anything.."
)

if __name__ == "__main__":
    chat.launch()
