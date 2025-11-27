"""
Index Builder for the RAG Chatbot.

This module crawls web documentation and Git repositories, processes the content,
and builds a FAISS vector index for semantic search. The index is used by the
chatbot to retrieve relevant documents when answering user questions.

Usage:
    python index/index_builder.py
"""

import os
import tempfile
from typing import List, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    RecursiveUrlLoader,
    GitLoader,
)
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


# ------------------ CONFIG ------------------
# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"  # LLM model for chat responses (cost-effective)
EMBED_MODEL = "text-embedding-3-large"  # Embedding model for vectorization

# Index storage path (relative to this file's directory)
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index-faiss")

# Crawler settings
MAX_DEPTH = 3  # Maximum depth for recursive URL crawling
TIMEOUT_SEC = 30  # Timeout for HTTP requests
USE_ASYNC = True  # Whether to use async crawling (currently disabled)
PREVENT_OUTSIDE = True  # Prevent crawling outside the starting domain

# URLs to crawl for documentation
# Note: qarnot.com/ with MAX_DEPTH=3 covers /documentation, /blog, and all other pages

START_URLS = [
    "https://qarnot.com/",  # Main site (covers docs, blog, product pages)
    "https://doc.tasq.qarnot.com/documentation/sdk-python/",  # Separate domain - SDK docs
    "https://qarnot.com/documentation/overview",
    "https://doc.tasq.qarnot.com/documentation/en/home",
    "https://qarnot.com/blog/cluster-roce",
    "https://qarnot.com/blog/paraview-web-qarnot",
    "https://qarnot.com/blog/code-saturne-qarnot",
    "https://qarnot.com/blog/openfoam-foundation-qarnot",
    "https://qarnot.com/blog/ansys-fluent-qarnot",
    "https://qarnot.com/blog/matlab-simulink-qarnot",
    "https://qarnot.com/blog/converge-qarnot",
    "https://qarnot.com/blog/star-ccm-qarnot",
    "https://qarnot.com/blog/openfoam-qarnot",
    "https://qarnot.com/blog/code-aster-qarnot",
    "https://qarnot.com/blog/fire-dynamics-simulator-qarnot",
    "https://qarnot.com/blog/ls-dyna-qarnot",
]



# Git repositories containing code examples to index
GIT_REPOS = [
    {
        "clone_url": "https://github.com/qarnot/blog-samples.git",
        "branch": "main",
    }
]


# ------------------ CRAWLING HELPERS ------------------
def _crawl_one(url_root: str) -> List[Any]:
    """
    Crawl a single URL recursively up to MAX_DEPTH levels.

    Args:
        url_root: The starting URL to crawl.

    Returns:
        List of Document objects containing the crawled HTML content.
    """
    print(
        f"[crawler] root={url_root} depth={MAX_DEPTH} "
        f"prevent_outside={PREVENT_OUTSIDE} use_async_env={USE_ASYNC}"
    )

    loader = RecursiveUrlLoader(
        url=url_root,
        max_depth=MAX_DEPTH,
        use_async=False,  # Async disabled for stability
        timeout=TIMEOUT_SEC,
        prevent_outside=PREVENT_OUTSIDE,
    )

    docs = loader.load()
    print(f"[crawler] fetched {len(docs)} pages from {url_root}")
    return docs


def _crawl_sites(start_urls: List[str]) -> List[Any]:
    """
    Crawl multiple websites and convert HTML to plain text.

    Processes each URL by:
    1. Recursively crawling all pages
    2. Converting HTML to markdown/text
    3. Adding source metadata
    4. Deduplicating based on content hash

    Args:
        start_urls: List of root URLs to crawl.

    Returns:
        Deduplicated list of Document objects with text content.
    """
    all_docs = []

    for root in start_urls:
        # Crawl the site
        raw_docs = _crawl_one(root)

        # Convert HTML to plain text
        text_docs = Html2TextTransformer().transform_documents(raw_docs)

        # Add metadata to each document
        for d in text_docs:
            d.metadata["source"] = d.metadata.get("source") or root
            d.metadata.setdefault("source_type", "web")

        print(f"[crawler] {root}: got {len(text_docs)} docs")

        # Log sample blog sources for debugging
        if "blog" in root:
            print("[crawler] sample blog sources:")
            for d in text_docs[:20]:
                print("   ", d.metadata.get("source"))

        all_docs.extend(text_docs)

    # Deduplicate documents by source URL and content hash
    seen = set()
    deduped = []
    for d in all_docs:
        key = (d.metadata.get("source", ""), hash(d.page_content))
        if key not in seen and d.page_content.strip():
            seen.add(key)
            deduped.append(d)

    print(f"[crawler] total after merge+dedup: {len(deduped)}")
    return deduped


# ------------------ GIT LOADING HELPERS ------------------
def _build_github_url(clone_url: str, branch: str, file_path: str) -> str:
    """
    Build a full GitHub URL for a file in a repository.

    Args:
        clone_url: The .git clone URL (e.g., https://github.com/user/repo.git)
        branch: The branch name
        file_path: The file path within the repo

    Returns:
        Full GitHub URL to view the file (e.g., https://github.com/user/repo/blob/main/path/file.py)
    """
    # Convert clone URL to browser URL
    # https://github.com/qarnot/blog-samples.git -> https://github.com/qarnot/blog-samples
    base_url = clone_url.replace(".git", "")
    return f"{base_url}/blob/{branch}/{file_path}"


def _load_git_repos() -> List[Any]:
    """
    Clone and load documents from configured Git repositories.

    Clones each repository to a temporary directory and loads
    markdown (.md) and Python (.py) files as documents.
    These are marked with source_type="git" to identify them
    as code examples in the RAG pipeline.

    Returns:
        List of Document objects from all Git repositories.
    """
    repo_docs: List[Any] = []

    for repo in GIT_REPOS:
        clone_url = repo["clone_url"]
        branch = repo.get("branch", "main")

        # Clone to a temporary directory
        local_path = tempfile.mkdtemp(prefix="qarnot_repo_")
        print(f"[git] cloning {clone_url} (branch={branch}) into {local_path}")

        # Load only .md and .py files from the repository
        loader = GitLoader(
            clone_url=clone_url,
            repo_path=local_path,
            branch=branch,
            file_filter=lambda p: (
                p.startswith("openfoam/") or
                p.startswith("ansys-fluent/") or
                p.startswith("ls-dyna/")
            ) and (p.endswith(".md") or p.endswith(".py"))
        )

        docs = loader.load()
        print(f"[git] loaded {len(docs)} docs from {clone_url}")

        # Add metadata to identify these as git/code sources
        for d in docs:
            # Build full GitHub URL from the file path
            file_path = d.metadata.get("source", "")
            if file_path:
                d.metadata["source"] = _build_github_url(clone_url, branch, file_path)
            else:
                d.metadata["source"] = clone_url
            d.metadata.setdefault("source_type", "git")

        repo_docs.extend(docs)

    print(f"[git] total repo docs: {len(repo_docs)}")
    return repo_docs


# ------------------ CHUNKING HELPERS ------------------
def _chunk_docs(docs: List[Any]) -> List[Any]:
    """
    Split documents into smaller chunks for embedding.

    Uses recursive character splitting to break documents into
    chunks of ~1000 characters with 150 character overlap.
    Overlap ensures context is preserved across chunk boundaries.

    Args:
        docs: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Target size for each chunk
        chunk_overlap=150,  # Overlap between chunks for context continuity
        separators=["\n\n", "\n", " ", ""],  # Split priority: paragraph > line > word > char
    )

    chunks = splitter.split_documents(docs)

    # Ensure each chunk has source metadata
    for d in chunks:
        d.metadata["source"] = d.metadata.get("source", "unknown")

    print(f"[index] chunks: {len(chunks)}")
    return chunks


# ------------------ BUILD INDEX ------------------
def build_index() -> None:
    """
    Build the complete FAISS vector index.

    Pipeline:
    1. Crawl web documentation from START_URLS
    2. Load code examples from GIT_REPOS
    3. Merge all documents
    4. Split into chunks for embedding
    5. Generate embeddings using OpenAI
    6. Save FAISS index to disk

    The resulting index can be loaded by the chatbot for semantic search.
    """
    print("[index] building new index...")

    # Step 1: Crawl web documentation
    site_docs = _crawl_sites(START_URLS)

    # Step 2: Load code examples from Git repositories
    git_docs = _load_git_repos()

    # Step 3: Merge all documents
    all_docs = site_docs + git_docs
    print(f"[index] total docs (web + git): {len(all_docs)}")

    # Step 4: Chunk documents for embedding
    chunks = _chunk_docs(all_docs)

    # Step 5: Generate embeddings and create FAISS index
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)

    # Step 6: Save index to disk
    vs.save_local(INDEX_PATH)
    print(f"[index] saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
