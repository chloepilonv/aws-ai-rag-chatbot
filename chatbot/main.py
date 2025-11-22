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
COMPANY_SUPPORT_EMAIL = "support@yourcompany.com"
COMPANY_NAME = "my_company"

load_dotenv()


# ------------------ LOAD INDEX ------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7},
)

# ------------------ RAG CHAIN ------------------
def _format_docs(docs: List[Any]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] ({src})\n{d.page_content}")
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
   - If the context contains a relevant code example, always include it in the answer as a fenced code block (```language). Don’t invent code that isn’t in the context.
   - If the answer is not in the context, say you don't know and suggest contacting support at {COMPANY_SUPPORT_EMAIL}.
   - Cite sources using [n] and list them under "Sources" with their URL.
   - Do not fabricate sources.
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
    question=RunnablePassthrough(),
)

rag_chain = rag_inputs | PROMPT | llm


# ------------------ FASTAPI ------------------
app = FastAPI(title=f"{COMPANY_NAME} RAG Chatbot")


class Ask(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>It works 🎉</h1><p>Try <a href='/docs'>/docs</a></p>"


@app.post("/ask")
def ask(payload: Ask) -> Dict[str, Any]:
    cb = ConsoleCallbackHandler()
    result = rag_chain.invoke(
        {"question": payload.question}, config={"callbacks": [cb]}
    )

    answer = (
        result if isinstance(result, str) else getattr(result, "content", str(result))
    )
    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok"}
