import io
import random
import time
from pathlib import Path
import base64
from fastapi.responses import JSONResponse

import modal

MINUTES = 60

# start Modal app
app = modal.App("pentagram-text-to-image")

# create container image with required configuration
# include ther OSS package huggingface_hub to access models and datasets, especially (libraries):
# transformer: https://huggingface.co/docs/transformers/index
# diffusers: https://huggingface.co/docs/diffusers/index
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface-hub[hf_transfer]==0.25.2",
        "accelerate==0.33.0",
        "diffusers==0.31.0",
        "fastapi[standard]==0.115.4",
        "sentencepiece==0.2.0",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers~=4.44.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster downloads
)

# prepare container to run application: make packages available to use in the code.
with image.imports():
    import diffusers
    import torch
    from fastapi import Response
    from datetime import datetime, timezone

# To load a model for inferences, make the inference function into a class and define a lifecycle
# Cls adds method pooling and lifecycle hook behavior to modal.Function.
# lifecycle hooks: enter, exit, build (https://modal.com/docs/guide/lifecycle-functions)

# ungated! can be used without accepting the gate. read more: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#usage-example
model_id = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
model_revision_id = "9ad870ac0b0e5e48ced156bb02f85d324b7275d2"

@app.cls(
    image=image,
    # gpu="A10G", # cheap model, with less memory
    gpu="H100",
    timeout=6 * MINUTES,
)
class Inference:
    @modal.build()
    @modal.enter()
    def initialize(self):
        # creates a local reference to the generative pipeline
        # similar to using a "seed" to allow reproducibility and debugging, keep track of the model_revision_id.
        self.pipe = diffusers.StableDiffusion3Pipeline.from_pretrained(
            model_id,
            revision=model_revision_id,
            torch_dtype=torch.bfloat16,
            # The SD3 pipeline uses three text encoders to generate an image. Model offloading is necessary in order for it to run on most commodity hardware.
        )

    @modal.enter()
    def move_to_gpu(self):
        # The to() function in PyTorch is used to move tensors or models to a specific device.
        # Use "cpu" or "cuda(:0-9+)?" to access the COU or GPU devices available.
        self.pipe.to("cuda")

        print("🏁 Starting up! --@move_to_gpu")
        self.start_time = datetime.now(timezone.utc)

    # Decorator for methods that will be transformed into a Modal Function registered against this class’s App.
    @modal.method()
    def run(
        self, prompt: str, batch_size: int = 4, seed: int = None
    ) -> list[bytes]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        print("seeding RNG with", seed)
        torch.manual_seed(seed)
        
        images = self.pipe( # images is an array of PIL images
            prompt,
            num_images_per_prompt=batch_size,  # outputting multiple images per prompt is much cheaper than separate calls
            num_inference_steps=4,  # turbo is tuned to run in four steps
            guidance_scale=0.0,  # turbo doesn't use CFG
            max_sequence_length=512,  # T5-XXL text encoder supports longer sequences, more complex prompts
        ).images

        # Convert PIL Image to bytes
        image_output = []
        for image in images:
            with io.BytesIO() as buf:
                image.save(buf, format="PNG")
                image_output.append(buf.getvalue())
        torch.cuda.empty_cache()  # reduce fragmentation
        return image_output # returning image bytes

    @modal.web_endpoint(method="POST", docs=True)
    def web(self, data: dict, seed: int = None):
        # pass prompt as { "prompt":"WHATEVER"} and "seed" as a URL param
        # returns only the first image of the requested array

        # Return base64 encoded image
        content = self.run.local(  # run in the same container
                data.get("prompt"), batch_size=1, seed=seed)[0]
        
        base64_image = base64.b64encode(content).decode('utf-8')
        return JSONResponse({
            "success": True,
            "image": {
                "base64": base64_image,
                "content_type": "image/png"
            }
        })