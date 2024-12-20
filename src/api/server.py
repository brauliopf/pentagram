import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import os

def download_model():
    from diffusers import DiffusionPipeline
    import torch

    DiffusionPipeline.from_pretrained( # https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline
        "stabilityai/sdxl-turbo", # https://huggingface.co/stabilityai/sdxl-turbo
        torch_dtype=torch.float16,
        variant="fp16"
    )

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",
        "transformers",
        "accelerate",
        "diffusers",
        "requests"
        )
    .run_function(download_model)
)

app = modal.App("text-2-image-demo", image=image)

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("API_KEY")],
    gpu="A10G", # cheap model, with less memory
    # gpu="H100",
    timeout=5 * 60, # 5 MINUTES
)
class Model:
    @modal.build()
    @modal.enter()
    def initialize(self):
        '''
        Sets up a local reference to the generative pipeline and moves processing to CUDA.
        '''
        from diffusers import DiffusionPipeline
        import torch

        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]
        print("🏁 Starting up! --@move_to_gpu")
        self.start_time = datetime.now(timezone.utc)

    @modal.web_endpoint() # defaut HTTP method: GET
    def generate(self, request:Request, prompt: str = Query(..., description = "Prompt for generative task")):
        
        # check API_KEY in request header
        api_key = request.headers.get("X-API-Key")
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401,
                detail="unauthorized"
            )
        
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        return Response(content=buf.getvalue(), media_type="image/png")
