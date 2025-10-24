#!/bin/bash
# ───────────────────────────────────────────────
# 🚀 Launch FastAPI RAG backend and Gradio UI
# ───────────────────────────────────────────────

# Exit on error
set -e

echo "🔧 Starting FastAPI backend (Uvicorn)..."
uvicorn main:app --reload --port 8000 &
FASTAPI_PID=$!

# Wait a few seconds for backend to start
sleep 3

echo "💬 Starting Gradio frontend..."
python gradio_bot.py &

GRADIO_PID=$!

# Wait for user to press Ctrl+C to stop both
trap "echo '🛑 Shutting down...'; kill $FASTAPI_PID $GRADIO_PID" SIGINT

# Keep the script running
wait
