FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Google Cloud Run expects port 8080 by default
ENV PORT 8080

# Run Gunicorn binding to the $PORT environment variable
# 1 Worker, 8 Threads (Better for I/O waiting like audio processing)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
