# Use the official Python image
FROM python:3.12.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /work

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY video_processor/app/requirements.txt /work/video_processor/app/requirements.txt
COPY video_processor/worker/requirements.txt /work/video_processor/worker/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r video_processor/app/requirements.txt
RUN pip install --no-cache-dir -r video_processor/worker/requirements.txt

# Copy project files
COPY . /work/

# Expose FastAPI and Redis ports
EXPOSE 8000
EXPOSE 6379

# Set working directory to video_processor
WORKDIR /work/video_processor

# Command to run the application
CMD ["bash", "-c", "service redis-server start && celery -A worker.celery_worker.app worker --beat --loglevel=info --pool=solo & uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
