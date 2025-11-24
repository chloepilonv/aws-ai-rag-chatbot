"""
FastAPI Backend for the RAG Chatbot.

This module provides the REST API that handles user questions using
retrieval-augmented generation (RAG). It loads a FAISS vector index,
retrieves relevant documents, and uses an LLM to generate answers.
"""

import os
import time
from typing import List, Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain models
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangChain RAG components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_community.vectorstores import FAISS

# Import shared constants from index_builder
from index.index_builder import EMBED_MODEL, OPENAI_MODEL

# Import logging module
from chatbot.logger import log_conversation, add_feedback, get_conversations, get_stats


# ------------------ CONFIG ------------------
INDEX_PATH = "index/index-faiss"  # Path to the FAISS vector index
COMPANY_SUPPORT_EMAIL = "support-compute@qarnot.com"
COMPANY_NAME = "Qarnot"

# Load environment variables from .env file
load_dotenv()

# Load system prompt from external file
SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.md")
with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read().replace("{COMPANY_SUPPORT_EMAIL}", COMPANY_SUPPORT_EMAIL)


# ------------------ LOAD INDEX ------------------
# Initialize embeddings model (must match the one used during indexing)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

# Load the FAISS vector store from disk
vectorstore = FAISS.load_local(
    INDEX_PATH,
    embeddings,
    allow_dangerous_deserialization=True,  # Required for loading pickled data
)

# Create a retriever using MMR (Maximum Marginal Relevance) for diverse results
# This ensures we get both web docs AND git code examples, not just similar web pages
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,  # Return 8 documents
        "fetch_k": 20,  # Fetch 20 candidates before MMR re-ranking
        "lambda_mult": 0.5,  # Balance between similarity (1.0) and diversity (0.0)
    },
)


# ------------------ RAG CHAIN ------------------
def _extract_sources(docs: List[Any]) -> List[str]:
    """
    Extract unique source URLs from retrieved documents.

    Args:
        docs: List of retrieved Document objects with metadata.

    Returns:
        List of unique source URLs.
    """
    sources = []
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in seen:
            sources.append(src)
            seen.add(src)
    return sources


def _format_docs(docs: List[Any]) -> str:
    """
    Format retrieved documents into a context string for the LLM.

    Processes each document by:
    - Cleaning footnote references (e.g., [1], [2])
    - Marking git sources as CODE EXAMPLE with Python syntax highlighting
    - Adding source attribution for each document

    Args:
        docs: List of retrieved Document objects with page_content and metadata.

    Returns:
        Formatted context string with numbered sources and content.
    """
    import re

    def clean_footnotes(text: str) -> str:
        """Remove footnote references like [1], [2], [3][4][5] from text."""
        return re.sub(r"\s*\[\d+\]", "", text)

    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        src_type = d.metadata.get("source_type", "web")

        cleaned = clean_footnotes(d.page_content)

        # Mark git sources as code examples with Python syntax highlighting
        if src_type == "git":
            blocks.append(f"[{i}] (CODE EXAMPLE from {src})\n```python\n{cleaned}\n```")
        else:
            blocks.append(f"[{i}] ({src})\n{cleaned}")

    return "\n\n---\n\n".join(blocks)


# Chat prompt template combining system instructions and user question
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

# Initialize the LLM with zero temperature for deterministic responses
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# Streaming LLM for the streaming endpoint
llm_streaming = ChatOpenAI(model=OPENAI_MODEL, temperature=0, streaming=True)

# RAG chain: retrieve context and pass question, then format and send to LLM
rag_inputs = RunnableParallel(
    context=lambda x: _format_docs(retriever.invoke(x["question"])),
    question=lambda x: x["question"],
)

# Complete RAG chain: inputs -> prompt -> LLM
rag_chain = rag_inputs | PROMPT | llm


# ------------------ FASTAPI APP ------------------
app = FastAPI(title=f"{COMPANY_NAME} RAG Chatbot")


class Ask(BaseModel):
    """Request model for the /ask endpoint."""
    question: str  # The user's question
    history: List[List[str]] = []  # Conversation history: [[user_msg, assistant_msg], ...]


@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint - displays a simple welcome page."""
    return "<h1>It works 🎉</h1><p>Try <a href='/docs'>/docs</a></p>"


def _format_history(history: List[List[str]]) -> str:
    """
    Format conversation history into a string for the LLM prompt.

    Args:
        history: List of [user_message, assistant_message] pairs.

    Returns:
        Formatted string with labeled user/assistant messages,
        or empty string if no history.
    """
    if not history:
        return ""
    lines = []
    for user_msg, assistant_msg in history:
        lines.append(f"User: {user_msg}")
        lines.append(f"Assistant: {assistant_msg}")
    return "\n".join(lines)


@app.post("/ask")
def ask(payload: Ask) -> Dict[str, Any]:
    """
    Main endpoint for asking questions to the chatbot.

    Processes the user's question through the RAG pipeline:
    1. Optionally prepends conversation history for context
    2. Retrieves relevant documents from the vector store
    3. Formats context and sends to the LLM
    4. Returns the generated answer
    5. Logs the conversation to SQLite database

    Args:
        payload: Request containing question and optional history.

    Returns:
        Dictionary with "answer" and "conversation_id" keys.
    """
    start_time = time.time()
    conversation_id = None

    try:
        # Callback handler for logging LLM interactions to console
        cb = ConsoleCallbackHandler()

        # Prepend conversation history to the question if available
        history_text = _format_history(payload.history)
        question_with_history = payload.question
        if history_text:
            question_with_history = f"Previous conversation:\n{history_text}\n\nCurrent question: {payload.question}"

        # Retrieve documents for context
        docs = retriever.invoke(question_with_history)
        sources = _extract_sources(docs)

        # Run the RAG chain
        result = rag_chain.invoke(
            {"question": question_with_history}, config={"callbacks": [cb]}
        )

        # Extract the answer content from the LLM response
        answer = (
            result if isinstance(result, str) else getattr(result, "content", str(result))
        )

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log the conversation
        try:
            conversation_id = log_conversation(
                question=payload.question,
                answer=answer,
                sources=sources,
                response_time_ms=response_time_ms
            )
        except Exception as log_error:
            print(f"[WARNING] Failed to log conversation: {log_error}")

        return {"answer": answer, "conversation_id": conversation_id}

    except Exception as e:
        print(f"[ERROR] /ask failed: {e}")
        return {"answer": f"Error processing request: {str(e)}", "conversation_id": None}


@app.post("/ask/stream")
async def ask_stream(payload: Ask):
    """
    Streaming endpoint for asking questions to the chatbot.

    Same as /ask but streams the response token by token for better UX.
    Also logs the conversation to SQLite database after streaming completes.

    Args:
        payload: Request containing question and optional history.

    Returns:
        StreamingResponse with the LLM's response streamed as text.
    """
    start_time = time.time()

    async def generate():
        answer_chunks = []
        try:
            # Prepend conversation history to the question if available
            history_text = _format_history(payload.history)
            question_with_history = payload.question
            if history_text:
                question_with_history = f"Previous conversation:\n{history_text}\n\nCurrent question: {payload.question}"

            # Get context and extract sources
            docs = retriever.invoke(question_with_history)
            sources = _extract_sources(docs)
            context = _format_docs(docs)

            # Build the streaming chain
            rag_chain_stream = PROMPT | llm_streaming

            # Stream the response and collect chunks
            async for chunk in rag_chain_stream.astream({
                "question": question_with_history,
                "context": context,
            }):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    answer_chunks.append(content)
                    yield content

            # After streaming completes, log the conversation
            full_answer = "".join(answer_chunks)
            response_time_ms = int((time.time() - start_time) * 1000)

            try:
                log_conversation(
                    question=payload.question,
                    answer=full_answer,
                    sources=sources,
                    response_time_ms=response_time_ms
                )
            except Exception as log_error:
                print(f"[WARNING] Failed to log conversation: {log_error}")

        except Exception as e:
            error_msg = f"\n\n❌ Error: {str(e)}"
            yield error_msg
            # Log error cases too
            try:
                log_conversation(
                    question=payload.question,
                    answer=error_msg,
                    sources=[],
                    response_time_ms=int((time.time() - start_time) * 1000)
                )
            except:
                pass

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
def health():
    """Health check endpoint for container orchestration."""
    return {"status": "ok"}


# ------------------ FEEDBACK ENDPOINTS ------------------

class Feedback(BaseModel):
    """Request model for the /feedback endpoint."""
    conversation_id: int
    feedback: int  # 1 for thumbs up, -1 for thumbs down
    comment: str = None


@app.post("/feedback")
def submit_feedback(payload: Feedback):
    """
    Submit user feedback for a conversation.

    Args:
        payload: Feedback data including conversation_id, feedback score, and optional comment.

    Returns:
        Success status.
    """
    try:
        add_feedback(
            conversation_id=payload.conversation_id,
            feedback=payload.feedback,
            comment=payload.comment
        )
        return {"status": "success", "message": "Feedback recorded"}
    except Exception as e:
        print(f"[ERROR] /feedback failed: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/feedback/conversations")
def export_conversations(limit: int = 100, offset: int = 0, feedback_only: int = None):
    """
    Export logged conversations for analysis.

    Args:
        limit: Maximum number of conversations to return (default: 100)
        offset: Number of conversations to skip (default: 0)
        feedback_only: Filter by feedback score (1 for positive, -1 for negative, None for all)

    Returns:
        List of conversation records.
    """
    try:
        conversations = get_conversations(limit=limit, offset=offset, feedback_only=feedback_only)
        return {"conversations": conversations, "count": len(conversations)}
    except Exception as e:
        print(f"[ERROR] /feedback/conversations failed: {e}")
        return {"conversations": [], "count": 0, "error": str(e)}


@app.get("/feedback/stats")
def feedback_stats():
    """
    Get statistics about logged conversations and feedback.

    Returns:
        Dictionary with stats (total conversations, feedback counts, avg response time).
    """
    try:
        stats = get_stats()
        return stats
    except Exception as e:
        print(f"[ERROR] /feedback/stats failed: {e}")
        return {"error": str(e)}
