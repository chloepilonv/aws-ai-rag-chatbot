FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for faiss, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project (including run.sh)
COPY . .

# Ensure script is executable
RUN chmod +x run.sh

# Expose FastAPI and Gradio ports
EXPOSE 8000
EXPOSE 7860

# Use run script
CMD ["./run.sh"]
