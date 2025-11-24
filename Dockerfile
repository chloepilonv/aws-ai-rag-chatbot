# Stage 1: Build stage - install dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies to a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage - minimal image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary application files
COPY chatbot/ ./chatbot/
COPY index/ ./index/
COPY run.sh .

# Ensure script is executable
RUN chmod +x run.sh

# Expose FastAPI and Gradio ports
EXPOSE 8000
EXPOSE 7860

# Use run script
CMD ["./run.sh"]
