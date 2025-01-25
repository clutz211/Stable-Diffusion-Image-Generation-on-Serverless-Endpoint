# Start with a lightweight Python image to keep the container small
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set longer timeouts because of my slow internet connection
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

# Install packages one at a time
RUN pip3 install --no-cache-dir runpod --timeout 1000
RUN pip3 install --no-cache-dir torch --timeout 1000
RUN pip3 install --no-cache-dir transformers --timeout 1000
RUN pip3 install --no-cache-dir diffusers --timeout 1000

# Copy over handler code
COPY handler.py .

# Command to run when the container starts
CMD [ "python3", "-u", "handler.py" ]
