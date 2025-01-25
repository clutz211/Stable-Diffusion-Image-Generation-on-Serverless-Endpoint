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
        
        # Convert to base64 string to make it easy to send to APi
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
