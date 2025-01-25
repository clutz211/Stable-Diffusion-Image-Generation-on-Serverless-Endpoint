# Stable Diffusion Image Generation on Serverless Endpoint 

# 1. Project Summary 

This project implements a serverless endpoint on a GPU platform that generates images from text prompts. The goal was to create a working API endpoint that could take text descriptions and return AI generated images. 

## 1.1 What Was Built 
- A Docker container configured to run a Stable Diffusion model
- A serverless endpoint deployed on GPU cloud infrastructure
- A working API that can be tested using Postman

## 1.2 Key Features 
- Text to image generation
- Simple API endpoint that accepts text prompts
- Returns generated images in base64 format

## 1.3 Technical Stack Used
- Docker for creating the container
- GPU cloud for processing
- Postman for testing the API

# 2. Configuration & Setup

## 2.1 GPU Cloud Configuration 
- GPU Type: NVIDIA L4/A5000/3090 with 24GB VRAM
- Memory: 24GB RAM (6 vCPU)
- Disk Space: 25 GB
- Container Image linked to the image I created

## 2.2 Handler Implementation 

The handler.py file processes the incoming text prompts and generates images: 

```python
import runpod
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
import gc

def handler(event):
    try:
        # Get the prompt from the API request
        # This is what the user wants to generate an image of
        prompt = event["input"]["prompt"]

        # Setting up the image generation model
        # Using float16 to save GPU memory and safetensors for faster loading
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to("cuda")

        # Enable memory saving settings to solve memory issues
        pipe.enable_attention_slicing()

        # Generate the image from the prompt
        image = pipe(prompt).images[0]

        # Clean up GPU memory after generation
        pipe = pipe.to("cpu")
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

        # Convert to base64 string to make it easy to send to API
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_str}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Always clean up memory even if there's an error
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
```
Key Handler Points: 
- Takes text prompts as input
- Uses Stable Diffusion model for image generation
- Includes memory management for GPU efficiency
- Returns images in base64 format

## 2.3 Docker Setup

The project uses a Docker container with the following configuration: 

```dockerfile
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
```
Key Configuration Points: 
- Built on Python 3.9 slim for image efficiency
- Includes necessary packages for image generation
- Configured with extended timeouts for image generation

# 3. Testing with Postman

## 3.1 Generate Image Request (POST)
URL: Grab URL from GPU cloud

Headers: 
- Authorization: Bearer [API Key]
- Content-Type: application/json

Request Body: 

```json
{
    "input": {
        "prompt": "A double-black diamond mountain to ski on near Seattle."
    }
}
```
# 3.2 Check Status Request (GET) 
Url: Grab the ID generated from the POST request. Can also check the status of the request here too. 

Header: 
- Authorization: Bearer [API Key]

## 3.3 Example Results 

Test prompt: "a double-black mountain to ski on near Seattle"




