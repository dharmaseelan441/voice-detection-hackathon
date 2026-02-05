# Use a lightweight Python image
FROM python:3.9-slim

# Install system dependencies (FFmpeg is required for MP3 processing)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 10000

# Start the application using Gunicorn (Production Server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:10000", "app:app"]