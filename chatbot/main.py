import os
from typing import List, Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_community.vectorstores import FAISS

# Import shared constants from index_builder
from index.index_builder import EMBED_MODEL, OPENAI_MODEL


INDEX_PATH = "index/index-faiss"
COMPANY_SUPPORT_EMAIL = "support-compute@qarnot.com"
COMPANY_NAME = "Qarnot"

load_dotenv()


# ------------------ LOAD INDEX ------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
)

# ------------------ RAG CHAIN ------------------
def _format_docs(docs: List[Any]) -> str:
    import re

    def clean_footnotes(text: str) -> str:
        # Remove trailing [number] or sequences like [3][4][5]
        return re.sub(r"\s*\[\d+\]", "", text)

    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        src_type = d.metadata.get("source_type", "web")

        cleaned = clean_footnotes(d.page_content)

        # Mark git sources as code examples
        if src_type == "git":
            blocks.append(f"[{i}] (CODE EXAMPLE from {src})\n```python\n{cleaned}\n```")
        else:
            blocks.append(f"[{i}] ({src})\n{cleaned}")

    return "\n\n---\n\n".join(blocks)

SYSTEM_PROMPT = f"""
You are an internal company assistant.

1. If the user is just greeting you or making small talk
   (e.g., "hi", "hello", "thanks", "how are you", emojis, etc.):
   - Respond briefly and friendly.
   - DO NOT use any external context.
   - DO NOT include a "Sources" section.

2. For product / documentation / technical questions:
   - Answer ONLY from the provided context.
   - The context contains two types of sources:
     * Web documentation: explanations and tutorials
     * CODE EXAMPLE sources: complete, working Python scripts from our GitHub repo
   - When the user asks "how to" do something, ALWAYS include the full code from CODE EXAMPLE sources. These are real, tested scripts that users can copy and run.
   - Don't just reference filenames - show the actual code content.
   - If the answer is not in the context, say you don't know and suggest contacting support at {COMPANY_SUPPORT_EMAIL}.
   - Cite sources using [n] and list them under "Sources" with their URL.
   - Do not fabricate sources or code.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "User question:\n{question}\n\nContext:\n{context}\n\n"
            "Format: a helpful answer followed by a 'Sources' section.",
        ),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

rag_inputs = RunnableParallel(
    context=lambda x: _format_docs(retriever.invoke(x["question"])),
    question=lambda x: x["question"],
)

rag_chain = rag_inputs | PROMPT | llm


# ------------------ FASTAPI ------------------
app = FastAPI(title=f"{COMPANY_NAME} RAG Chatbot")


class Ask(BaseModel):
    question: str
    history: List[List[str]] = []  # [[user_msg, assistant_msg], ...]


@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>It works 🎉</h1><p>Try <a href='/docs'>/docs</a></p>"


def _format_history(history: List[List[str]]) -> str:
    """Format conversation history for the prompt."""
    if not history:
        return ""
    lines = []
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


@app.post("/ask")
def ask(payload: Ask) -> Dict[str, Any]:
    try:
        cb = ConsoleCallbackHandler()

        # Build prompt with history
        history_text = _format_history(payload.history)
        question_with_history = payload.question
        if history_text:
            question_with_history = f"Previous conversation:\n{history_text}\n\nCurrent question: {payload.question}"

        result = rag_chain.invoke(
            {"question": question_with_history}, config={"callbacks": [cb]}
        )

        answer = (
            result if isinstance(result, str) else getattr(result, "content", str(result))
        )
        return {"answer": answer}
    except Exception as e:
        print(f"[ERROR] /ask failed: {e}")
        return {"answer": f"Error processing request: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "ok"}
