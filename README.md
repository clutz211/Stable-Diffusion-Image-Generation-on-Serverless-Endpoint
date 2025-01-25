# Stable Diffusion Image Generation on Serverless Endpoint 

# 1) Project Summary 

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

# 2) Configuration & Setup

## 2.1 GPU Cloud Configuration 
- GPU Type: NVIDIA L4/A5000/3090 with 24GB VRAM
- Memory: 24GB RAM (6 vCPU)
- Disk Space: 25 GB
- Container Image linked to the image I created

## 2.2 Handler Implementation 

The handler.py file processes the incoming text prompts and generates images: 

''' 
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
'''






