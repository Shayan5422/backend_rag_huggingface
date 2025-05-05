# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONIOENCODING=utf-8
# Set cache directory for Hugging Face libraries to a writable location
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
# Create the cache directory (optional, libraries might create it, but good practice)
# RUN mkdir -p /app/.cache && chown -R <user>:<group> /app/.cache # Might need user/group if not running as root

# Set the working directory in the container
WORKDIR /app

# Create the cache directory and set permissions *before* copying files
# This ensures the directory exists and is writable by the user running the process
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
# This includes app.py, index files, model_data_json directory, etc.
COPY . .

# Make port 8080 available to the world outside this container
# Hugging Face Spaces will map $PORT to this internally if needed, but Gunicorn binds to $PORT directly.
# EXPOSE 8080 is more informational here.
EXPOSE 8080

# Define the command to run the application using Gunicorn
# Bind to the standard Hugging Face Spaces port 7860 explicitly
# Use shell form for CMD
CMD gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --timeout 120 