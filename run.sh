#!/bin/bash
set -e

echo "🔧 Starting FastAPI backend (Uvicorn)..."
uvicorn chatbot.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

sleep 3

echo "💬 Starting Gradio frontend..."
python chatbot/gradio_bot.py &
GRADIO_PID=$!

echo ""
echo "🌐 FastAPI docs:  http://localhost:8000/docs"
echo "💻 Your Chatbot ! :     http://localhost:7860"
echo ""

trap "echo '🛑 Shutting down...'; kill $FASTAPI_PID $GRADIO_PID" SIGINT SIGTERM

wait
