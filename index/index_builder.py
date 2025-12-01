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
    DirectoryLoader,
    TextLoader,
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
START_URLS = [
    "https://qarnot.com/",
    "https://qarnot.com/documentation/overview",
    "https://doc.tasq.qarnot.com/documentation/sdk-python/",
    # Tasq documentation routes are auto-discovered (see _get_tasq_docs_urls below)
]

# Enable automatic route discovery for doc.tasq.qarnot.com
# Set to False to use manual URL list above
AUTO_DISCOVER_TASQ_ROUTES = True

# Blog posts to KEEP (all others from /blog/ will be filtered out)
ALLOWED_BLOG_POSTS = [
    "blog/ansys-fluent-qarnot",
    "blog/openfoam-foundation-qarnot",
    "blog/ls-dyna-qarnot",
]



# Git repositories containing code examples to index
GIT_REPOS = [
    {
        "clone_url": "https://github.com/qarnot/blog-samples.git",
        "branch": "main",
    }
]

# Specific directories to include from Git repositories
# Only .py and .md files from these directories will be indexed
GIT_REPO_DIRECTORIES = [
    "ansys-fluent/",
    "openfoam-foundation/",
    "openfoam/",
    "ls-dyna/",
]


# ------------------ CRAWLING HELPERS ------------------
def _crawl_js_rendered_recursive(start_url: str, max_depth: int = MAX_DEPTH) -> List[Any]:
    """
    Recursively crawl JavaScript-rendered pages using Playwright headless browser.

    SPECIAL CASE FOR doc.tasq.qarnot.com:
    This site is built with Vue.js and renders content client-side via JavaScript.
    Regular HTTP crawlers (like RecursiveUrlLoader) only see the empty HTML shell,
    missing all the actual documentation content.

    This function:
    1. Launches a headless Chromium browser (kept alive across multiple pages)
    2. Navigates to the starting URL and waits for JavaScript to execute
    3. Extracts links from the rendered page
    4. Recursively follows links within the same domain up to max_depth
    5. Converts all rendered HTML to markdown text
    6. Returns all documents

    Args:
        start_url: The root URL to start crawling from
        max_depth: Maximum depth to crawl (default: MAX_DEPTH from config)

    Returns:
        List of Document objects from all crawled pages
    """
    from playwright.sync_api import sync_playwright
    from langchain_core.documents import Document
    from urllib.parse import urljoin, urlparse

    print(f"[js-crawler] starting recursive crawl from {start_url} (max_depth={max_depth})")

    all_docs = []
    visited = set()  # Track visited URLs to avoid duplicates
    to_visit = [(start_url, 0)]  # Queue of (url, depth) tuples

    # Extract base domain to prevent crawling outside
    base_domain = urlparse(start_url).netloc
    # Allow crawling anywhere under /documentation/en/ for English docs
    if "/documentation/en/" in start_url:
        base_path = f"https://{base_domain}/documentation/en/"
    elif "/documentation/sdk-python/" in start_url:
        base_path = f"https://{base_domain}/documentation/sdk-python/"
    else:
        base_path = "/".join(start_url.split("/")[:4])  # Default: first 4 segments

    try:
        with sync_playwright() as p:
            # Launch headless browser (reused across all pages)
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            try:
                while to_visit:
                    current_url, depth = to_visit.pop(0)

                    # Skip if already visited or too deep
                    if current_url in visited or depth > max_depth:
                        continue

                    # Skip if outside base path
                    if not current_url.startswith(base_path):
                        continue

                    visited.add(current_url)
                    print(f"[js-crawler] [{len(visited)}] depth={depth} {current_url}")

                    try:
                        # Navigate and wait for content to load
                        page.goto(current_url, wait_until="networkidle", timeout=TIMEOUT_SEC * 1000)
                        page.wait_for_timeout(2000)  # Extra wait for Vue.js rendering

                        # Get rendered HTML
                        content = page.content()

                        # Create document
                        raw_doc = Document(page_content=content, metadata={"source": current_url})
                        text_docs = Html2TextTransformer().transform_documents([raw_doc])

                        for d in text_docs:
                            d.metadata["source"] = current_url
                            d.metadata.setdefault("source_type", "web")

                        all_docs.extend(text_docs)

                        # Extract links if not at max depth
                        if depth < max_depth:
                            links = page.eval_on_selector_all(
                                'a[href]',
                                '(elements) => elements.map(e => e.href)'
                            )

                            for link in links:
                                # Normalize URL
                                absolute_url = urljoin(current_url, link)
                                # Remove fragments
                                absolute_url = absolute_url.split('#')[0]

                                # Only follow links within the same base path
                                if (absolute_url.startswith(base_path) and
                                    absolute_url not in visited and
                                    absolute_url not in [u for u, _ in to_visit]):
                                    to_visit.append((absolute_url, depth + 1))

                    except Exception as e:
                        print(f"[js-crawler] ERROR on {current_url}: {e}")
                        continue  # Skip this page but continue crawling

            finally:
                browser.close()

    except Exception as e:
        print(f"[js-crawler] FATAL ERROR: {e}")

    print(f"[js-crawler] crawled {len(visited)} pages, got {len(all_docs)} documents")
    return all_docs


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

    IMPORTANT: This function handles TWO types of sites:
    1. Regular static HTML sites -> uses RecursiveUrlLoader
    2. JavaScript-rendered sites (doc.tasq.qarnot.com) -> uses Playwright

    Processes each URL by:
    1. Recursively crawling all pages (or rendering with JS if needed)
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
        # Check if this is a JavaScript-rendered site that needs Playwright
        if "doc.tasq.qarnot.com" in root:
            # Use Playwright for Vue.js rendered pages
            text_docs = _crawl_js_rendered_recursive(root)
        else:
            # Crawl the site (normal HTTP crawler)
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

    # Filter blog posts: keep only allowed ones, keep all non-blog content
    filtered_docs = []
    excluded_count = 0
    for d in all_docs:
        source = d.metadata.get("source", "")

        # Check if this is a blog post
        if "/blog/" in source:
            # Only keep if it's in the allowlist
            is_allowed = any(allowed in source for allowed in ALLOWED_BLOG_POSTS)
            if is_allowed:
                filtered_docs.append(d)
            else:
                excluded_count += 1
        else:
            # Not a blog post, keep it
            filtered_docs.append(d)

    if excluded_count > 0:
        print(f"[crawler] excluded {excluded_count} non-allowed blog posts")

    # Deduplicate documents by source URL and content hash
    seen = set()
    deduped = []
    for d in filtered_docs:
        key = (d.metadata.get("source", ""), hash(d.page_content))
        if key not in seen and d.page_content.strip():
            seen.add(key)
            deduped.append(d)

    print(f"[crawler] total after filter+dedup: {len(deduped)}")
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
    import subprocess
    import glob as glob_module

    repo_docs: List[Any] = []

    for repo in GIT_REPOS:
        clone_url = repo["clone_url"]
        branch = repo.get("branch", "main")

        # Clone to a temporary directory
        local_path = tempfile.mkdtemp(prefix="qarnot_repo_")
        print(f"[git] cloning {clone_url} (branch={branch}) into {local_path}")

        # Clone the repository using subprocess
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, clone_url, local_path],
            check=True,
            capture_output=True
        )

        # Load files from specific directories
        for directory in GIT_REPO_DIRECTORIES:
            dir_path = os.path.join(local_path, directory.rstrip("/"))
            if not os.path.exists(dir_path):
                print(f"[git] directory not found: {directory}")
                continue

            # Find all .py and .md files in this directory
            py_files = glob_module.glob(os.path.join(dir_path, "*.py"))
            md_files = glob_module.glob(os.path.join(dir_path, "*.md"))
            all_files = py_files + md_files

            print(f"[git] loading {len(all_files)} files from {directory}")

            for file_path in all_files:
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()

                    # Update metadata for each document
                    for d in docs:
                        # Get relative path from repo root
                        rel_path = os.path.relpath(file_path, local_path)
                        # Build GitHub URL
                        d.metadata["source"] = _build_github_url(clone_url, branch, rel_path)
                        d.metadata["source_type"] = "git"
                        d.metadata["file_name"] = os.path.basename(file_path)

                    repo_docs.extend(docs)
                except Exception as e:
                    print(f"[git] error loading {file_path}: {e}")

    print(f"[git] total repo docs: {len(repo_docs)}")
    return repo_docs


# ------------------ CHUNKING HELPERS ------------------
def _chunk_docs(docs: List[Any]) -> List[Any]:
    """
    Split documents into smaller chunks for embedding with improved strategy.

    IMPROVEMENTS FROM BASIC CHUNKING:
    1. Different chunk sizes for code vs documentation
       - Code examples: 1500 chars (need more context for complete functions)
       - Documentation: 1000 chars (standard for text)
    2. Source quality scoring metadata
       - Official docs: priority 3 (highest)
       - Git code examples: priority 2 (high for "how to" questions)
       - Blog posts: priority 1 (lowest)
    3. Source type metadata for better filtering

    Args:
        docs: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with enhanced metadata.
    """
    # Separate docs by type for different chunking strategies
    code_docs = [d for d in docs if d.metadata.get("source_type") == "git"]
    web_docs = [d for d in docs if d.metadata.get("source_type") != "git"]

    # Code splitter: larger chunks to preserve function context
    code_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\nclass ", "\ndef ", "\n", " ", ""],
    )

    # Web docs splitter: standard size
    web_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    # Chunk each type
    code_chunks = code_splitter.split_documents(code_docs) if code_docs else []
    web_chunks = web_splitter.split_documents(web_docs) if web_docs else []

    # Add source quality scoring
    for d in code_chunks:
        d.metadata["source"] = d.metadata.get("source", "unknown")
        d.metadata["source_type"] = "git"
        d.metadata["quality_score"] = 3  # CODE EXAMPLES: Highest priority (changed from 2)

    for d in web_chunks:
        source = d.metadata.get("source", "")
        d.metadata["source"] = source

        # Determine quality score based on source
        if "doc.tasq.qarnot.com" in source or "qarnot.com/documentation" in source:
            d.metadata["quality_score"] = 3  # Official docs (same as code)
        elif "/blog/" in source:
            d.metadata["quality_score"] = 1  # Blog posts (lowest)
        else:
            d.metadata["quality_score"] = 2  # Other web content (medium)

    all_chunks = code_chunks + web_chunks
    print(f"[index] chunks: {len(all_chunks)} (code: {len(code_chunks)}, web: {len(web_chunks)})")
    return all_chunks


# ------------------ ROUTE DISCOVERY ------------------
def _get_tasq_docs_urls() -> List[str]:
    """
    Get all doc.tasq.qarnot.com documentation URLs.

    If AUTO_DISCOVER_TASQ_ROUTES is True, automatically discovers all routes.
    Otherwise, uses the cached discovered_routes.json file.

    Returns:
        List of documentation URLs to crawl
    """
    # Add index directory to path for imports
    import sys
    index_dir = os.path.dirname(os.path.abspath(__file__))
    if index_dir not in sys.path:
        sys.path.insert(0, index_dir)

    if not AUTO_DISCOVER_TASQ_ROUTES:
        print("[index] auto-discovery disabled, using cached routes")
        # Try to load from cache
        import route_discovery
        cached_routes = route_discovery.load_discovered_routes()
        if cached_routes:
            print(f"[index] loaded {len(cached_routes)} routes from cache")
            return cached_routes
        else:
            print("[index] no cached routes found, falling back to manual list")
            return []

    # Auto-discover routes
    print("[index] auto-discovering Tasq documentation routes...")
    import route_discovery

    routes = route_discovery.discover_routes_from_nuxt("https://doc.tasq.qarnot.com")

    if routes:
        # Save for next time (caching)
        route_discovery.save_routes(routes)
        print(f"[index] discovered {len(routes)} routes from doc.tasq.qarnot.com")
        return routes
    else:
        print("[index] route discovery failed, using empty list")
        return []


# ------------------ BUILD INDEX ------------------
def build_index() -> None:
    """
    Build the complete FAISS vector index with BM25 support.

    Pipeline:
    1. Auto-discover Tasq documentation routes (if enabled)
    2. Crawl web documentation from START_URLS + discovered routes
    3. Load code examples from GIT_REPOS
    4. Merge all documents
    5. Split into chunks for embedding
    6. Generate embeddings using OpenAI
    7. Save FAISS index to disk
    8. Save chunked documents for BM25 (hybrid search)

    The resulting index can be loaded by the chatbot for semantic + keyword search.
    """
    import pickle

    print("[index] building new index...")

    # Step 1: Get all URLs to crawl
    urls_to_crawl = list(START_URLS)  # Copy the base list

    # Add auto-discovered Tasq routes
    tasq_routes = _get_tasq_docs_urls()
    if tasq_routes:
        urls_to_crawl.extend(tasq_routes)
        print(f"[index] total URLs to crawl: {len(urls_to_crawl)}")

    # Step 2: Crawl web documentation
    site_docs = _crawl_sites(urls_to_crawl)

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

    # Step 6: Save FAISS index to disk
    vs.save_local(INDEX_PATH)
    print(f"[index] saved FAISS index to {INDEX_PATH}")

    # Step 7: Save chunks for BM25 hybrid search
    chunks_path = os.path.join(INDEX_PATH, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[index] saved {len(chunks)} chunks for BM25 to {chunks_path}")


if __name__ == "__main__":
    build_index()
