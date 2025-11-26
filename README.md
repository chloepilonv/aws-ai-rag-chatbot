# RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot that can answer questions based on your own documents. Can be use internally or to helps your clients to use your application.

Built with **FastAPI**, **FAISS**, and **Gradio** for a lightweight local assistant setup.

---

## How to Run

### Local Development

1. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip3 install -r requirements.txt
```

3. **Run the app**
```bash
./run.sh
```

### Docker Production Deployment

1. **Build the image**
```bash
docker build -t chatbot-mvp .
```

2. **Run with volume mount for SQLite logging**
```bash
# Create a directory for persistent data
mkdir -p /srv/chatbot-data

# Run the container with volume mount
docker run -d \
  --env-file .env \
  -p 8000:8000 \
  -p 7860:7860 \
  -v $(pwd)/chatbot-data:/data \
  chatbot-mvp
```

3. **Access the database**
```bash
# The SQLite database is stored on your host at /srv/chatbot-data/feedback.db

# View all conversations (compact)
sqlite3 /srv/chatbot-data/feedback.db "SELECT * FROM conversations LIMIT 5;"

# View specific columns with headers
sqlite3 -column -header /srv/chatbot-data/feedback.db \
  "SELECT id, question, substr(answer, 1, 100) as answer_preview, response_time_ms FROM conversations;"

# Export via API
curl http://localhost:8000/feedback/conversations > conversations.json

# View statistics
curl http://localhost:8000/feedback/stats
```

---

## Description

This chatbot:
- Uses a **RAG pipeline** to combine LLM reasoning with document retrieval.
- Stores embeddings locally in a **FAISS** index.
- Exposes a simple **FastAPI** backend and a **Gradio** chat interface.
- **Logs all conversations** to SQLite for monitoring and improvement.

---

## Features

### Conversation Logging
Every question and answer is automatically logged to a SQLite database with:
- Question and answer text
- Source documents used
- Response time
- Timestamp
- Optional user feedback (thumbs up/down)

### API Endpoints

**Main Endpoints:**
- `POST /ask` - Ask a question (with logging, returns conversation_id)
- `POST /ask/stream` - Ask a question with streaming response
- `GET /conversation/latest?question=<text>` - Get the latest conversation ID for a question
- `GET /health` - Health check

**Feedback & Analytics:**
- `POST /feedback` - Submit 3-star rating (-1/0/1) for a conversation
- `GET /feedback/conversations` - Export all logged conversations
- `GET /feedback/stats` - View statistics (total Q&A, feedback counts, avg response time)

### Example: Submit Feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": 1, "feedback": 1, "comment": "Very helpful!"}'
```

### Example: View Statistics
```bash
curl http://localhost:8000/feedback/stats
# Returns: {"total_conversations": 42, "positive_feedback": 30, "negative_feedback": 5, ...}
```

---

**Stack**
- Python 3.11+
- FastAPI
- FAISS
- Sentence Transformers
- OpenAI / compatible LLM
- Gradio

**License**
Under MIT license, see LICENSE.

## Next steps

1. Insert external docs (Ansys, StarCCM, etc.)
2. Test run with and without -p 8000:8000
3. Improve referencing an dsourcing
4. Try with Qdrant
5. Use an agent instead :-)
