# RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot that can answer questions based on your own documents. Can be use internally or to helps your clients to use your application.

Built with **FastAPI**, **FAISS**, and **Gradio** for a lightweight local assistant setup.

---

## 🚀 How to Run

### 1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Run the app
```bash
./run.sh
```

---

## 💬 Description

This chatbot:
- Uses a **RAG pipeline** to combine LLM reasoning with document retrieval.
- Stores embeddings locally in a **FAISS** index.
- Exposes a simple **FastAPI** backend and a **Gradio** chat interface.

---

🧩 **Stack**
- Python 3.11+
- FastAPI
- FAISS
- Sentence Transformers
- OpenAI / compatible LLM
- Gradio

