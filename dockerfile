# Use an official Python runtime as a base image
FROM python:3.10
# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY video_processor/app/requirements.txt /app/app/requirements.txt
COPY video_processor/worker/requirements.txt /app/worker/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/app/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

# Install Redis
RUN apt-get update && apt-get install -y redis-server

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Copy the entire project directory into the container at /app
COPY . /app

# Start Redis server and Celery worker
CMD redis-server & \
    celery -A video_processor.worker.celery_worker.app worker --beat --loglevel=info --pool=solo & \
    uvicorn video_processor.app.main:app --host 0.0.0.0 --reload
