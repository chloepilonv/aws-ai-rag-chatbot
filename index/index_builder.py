import os
import re
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

from bs4 import BeautifulSoup


load_dotenv()

# ------------------ CONFIG ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1"
EMBED_MODEL = "text-embedding-3-large"
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index-faiss")

MAX_DEPTH = 3
TIMEOUT_SEC = 30
USE_ASYNC = True
PREVENT_OUTSIDE = True

START_URLS = [
    "https://qarnot.com/documentation/overview",
    "https://doc.tasq.qarnot.com/documentation/sdk-python/",
    "https://qarnot.com/blog",
]

# Git repos to include in the index
GIT_REPOS = [
    {
        "clone_url": "https://github.com/qarnot/blog-samples.git",
        "branch": "main",
    }
]




# ------------------ HELPERS ------------------

def _crawl_one(url_root: str) -> List[Any]:
    print(
        f"[crawler] root={url_root} depth={MAX_DEPTH} "
        f"prevent_outside={PREVENT_OUTSIDE} use_async_env={USE_ASYNC}"
    )

    loader = RecursiveUrlLoader(
        url=url_root,
        max_depth=MAX_DEPTH,
        use_async=False,  # you can wire USE_ASYNC here if you want
        timeout=TIMEOUT_SEC,
        prevent_outside=PREVENT_OUTSIDE,
    )

    docs = loader.load()
    print(f"[crawler] fetched {len(docs)} pages from {url_root}")
    return docs


def _crawl_sites(start_urls: List[str]) -> List[Any]:
    all_docs = []

    for root in start_urls:
        raw_docs = _crawl_one(root)

        text_docs = Html2TextTransformer().transform_documents(raw_docs)
        for d in text_docs:
            d.metadata["source"] = d.metadata.get("source") or root
            d.metadata.setdefault("source_type", "web")

        print(f"[crawler] {root}: got {len(text_docs)} docs")

        if "blog" in root:
            print("[crawler] sample blog sources:")
            for d in text_docs[:20]:
                print("   ", d.metadata.get("source"))

        all_docs.extend(text_docs)


    # Deduplicate
    seen = set()
    deduped = []
    for d in all_docs:
        key = (d.metadata.get("source", ""), hash(d.page_content))
        if key not in seen and d.page_content.strip():
            seen.add(key)
            deduped.append(d)

    print(f"[crawler] total after merge+dedup: {len(deduped)}")
    return deduped


def _load_git_repos() -> List[Any]:
    repo_docs: List[Any] = []

    for repo in GIT_REPOS:
        clone_url = repo["clone_url"]
        branch = repo.get("branch", "main")

        local_path = tempfile.mkdtemp(prefix="qarnot_repo_")
        print(f"[git] cloning {clone_url} (branch={branch}) into {local_path}")

        loader = GitLoader(
            clone_url=clone_url,
            repo_path=local_path,
            branch=branch,
            file_filter=lambda p: p.endswith(".md") or p.endswith(".py"),
        )

        docs = loader.load()
        print(f"[git] loaded {len(docs)} docs from {clone_url}")

        for d in docs:
            # Ensure we have a meaningful source and type
            d.metadata["source"] = d.metadata.get("source") or clone_url
            d.metadata.setdefault("source_type", "git")

        repo_docs.extend(docs)

    print(f"[git] total repo docs: {len(repo_docs)}")
    return repo_docs


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


# ------------------ BUILD INDEX ------------------
def build_index() -> None:
    print("[index] building new index...")

    # 1. Crawl your web docs
    site_docs = _crawl_sites(START_URLS)

    # 2. Load GitHub repo docs (README, .py, etc.)
    git_docs = _load_git_repos()

    # 3. Merge all docs
    all_docs = site_docs + git_docs
    print(f"[index] total docs (web + git): {len(all_docs)}")

    # 4. Chunk, embed, store
    chunks = _chunk_docs(all_docs)
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_PATH)
    print(f"[index] saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
