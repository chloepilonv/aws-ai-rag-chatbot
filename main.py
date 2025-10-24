import os
import re
from typing import List, Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Providers
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Core (prompts, parsers, runnables)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # was langchain.schema.runnable

# Recursive character text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders / transformers (community)
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer

# Tracing to console
from langchain_core.tracers.stdout import ConsoleCallbackHandler

# Vector store (FAISS)
from langchain_community.vectorstores import FAISS

# Load variables from .env into environment
load_dotenv()

# ------------------ CONFIG ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
INDEX_PATH = os.getenv("INDEX_PATH", "./index-faiss")

START_URLS = [
    "URL_1_OF_YOUR_DOCUMENTATION",
    "URL_2_OF_YOUR_DOCUMENTATION_AND_SO_ON"

]

MAX_DEPTH = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
TIMEOUT_SEC = int(os.getenv("CRAWL_TIMEOUT_SEC", "30"))

# NOTE: even if env says true, we force sync loader below to avoid asyncio.run() under uvicorn
USE_ASYNC = os.getenv("CRAWL_USE_ASYNC", "true").lower() == "true"
PREVENT_OUTSIDE = os.getenv("CRAWL_PREVENT_OUTSIDE", "true").lower() == "true"

INCLUDE_PATTERNS = [p for p in os.getenv("CRAWL_INCLUDE_PATTERNS", "").split(",") if p.strip()]
EXCLUDE_PATTERNS = [p for p in os.getenv("CRAWL_EXCLUDE_PATTERNS", "").split(",") if p.strip()]

REBUILD_INDEX = os.getenv("REBUILD_INDEX", "0") == "1"


# ------------------ HELPERS ------------------
def _passes_filters(url: str) -> bool:
    if INCLUDE_PATTERNS:
        if not any(re.search(p, url) for p in INCLUDE_PATTERNS):
            return False
    if EXCLUDE_PATTERNS:
        if any(re.search(p, url) for p in EXCLUDE_PATTERNS):
            return False
    return True


def _crawl_one(url_root: str) -> List[Any]:
    """Crawl a single root URL recursively and return Documents (HTML kept)."""
    # We log the env's USE_ASYNC, but we will pass use_async=False to avoid asyncio.run()
    print(f"[crawler] root={url_root} depth={MAX_DEPTH} prevent_outside={PREVENT_OUTSIDE} use_async_env={USE_ASYNC}")

    loader = RecursiveUrlLoader(
        url=url_root,
        max_depth=MAX_DEPTH,
        use_async=False,          # 🔴 force SYNC path so .load() won't call asyncio.run()
        timeout=TIMEOUT_SEC,
        prevent_outside=PREVENT_OUTSIDE,
    )

    docs = loader.load()          # ✅ sync call, safe under uvicorn's running loop
    print(f"[crawler] fetched {len(docs)} pages from {url_root}")
    return docs


def _crawl_sites(start_urls: List[str]) -> List[Any]:
    """Crawl each root separately; merge results; HTML → text; de-dup."""
    transformer = Html2TextTransformer()
    all_docs = []

    for root in start_urls:
        raw_docs = _crawl_one(root)  # ✅ list of Documents

        # Filter by URL patterns if provided
        filtered = [d for d in raw_docs if _passes_filters(d.metadata.get("source", ""))]
        print(f"[crawler] {root}: after filters -> {len(filtered)}")

        # HTML → text
        text_docs = transformer.transform_documents(filtered)

        # Ensure source metadata present
        for d in text_docs:
            d.metadata["source"] = d.metadata.get("source") or root

        all_docs.extend(text_docs)

    # De-duplicate by (url, text hash)
    seen = set()
    deduped = []
    for d in all_docs:
        key = (d.metadata.get("source", ""), hash(d.page_content))
        if key not in seen and d.page_content.strip():
            seen.add(key)
            deduped.append(d)

    print(f"[crawler] total after merge+dedup: {len(deduped)}")
    return deduped


def _chunk_docs(docs: List[Any]) -> List[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for d in chunks:
        d.metadata["source"] = d.metadata.get("source", "unknown")

    print(f"[index] chunks: {len(chunks)}")
    return chunks


def build_or_load_index() -> FAISS:
    if os.path.isdir(INDEX_PATH) and not REBUILD_INDEX:
        print(f"[index] loading index at {INDEX_PATH}")
        return FAISS.load_local(
            INDEX_PATH,
            OpenAIEmbeddings(model=EMBED_MODEL),
            allow_dangerous_deserialization=True,
        )

    print("[index] building new index...")
    docs = _crawl_sites(START_URLS)
    chunks = _chunk_docs(docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    print(f"[index] saved to {INDEX_PATH}")
    return vs


# ------------------ RAG CHAIN ------------------
vectorstore = build_or_load_index()
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7},
)

SYSTEM_PROMPT = """You are an internal company assistant. Answer ONLY from the provided context.
If the answer is not in the context, say you don't know and suggest where it might be found.
Cite sources using [n], and list them under "Sources" with their URL.
Keep answers concise and accurate. No fabrication.
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


def _format_docs(docs: List[Any]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# Build inputs for the chain using retriever.invoke(...)
rag_inputs = RunnableParallel(
    context=lambda x: _format_docs(retriever.invoke(x["question"])),
    question=RunnablePassthrough(),
)

# If you prefer the raw LLM message, keep as-is; or add StrOutputParser() to return a string
rag_chain = rag_inputs | PROMPT | llm
# rag_chain = rag_inputs | PROMPT | llm | StrOutputParser()  # <- uncomment to get a plain string


# ------------------ FASTAPI ------------------
app = FastAPI(title="Company RAG Chatbot (Multi-Root Crawler)")


class Ask(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def root():
    return "<h1>It works 🎉</h1><p>Try <a href='/docs'>/docs</a></p>"


@app.post("/ask")
def ask(payload: Ask) -> Dict[str, Any]:
    cb = ConsoleCallbackHandler()
    result = rag_chain.invoke({"question": payload.question}, config={"callbacks": [cb]})

    # Use retriever.invoke(...) instead of get_relevant_documents(...)
    docs = retriever.invoke(payload.question)
    citations = [{"label": f"[{i}]", "url": d.metadata.get("source", "unknown")} for i, d in enumerate(docs, 1)]

    # If using StrOutputParser above, result is a string; otherwise it's an AIMessage with .content
    answer = result if isinstance(result, str) else getattr(result, "content", str(result))
    return {"answer": answer, "citations": citations}


@app.get("/health")
def health():
    return {"status": "ok"}

# To run:
# uvicorn main:app --reload
