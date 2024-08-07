from diffusers.utils import load_image, check_min_version
import torch

# Local File
from controlnet_sd3 import SD3ControlNetModel
from pipeline_stable_diffusion_3_controlnet_inpainting import StableDiffusion3ControlNetInpaintingPipeline

check_min_version("0.29.2")

# Build model
controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting",
    use_safetensors=True,
    extra_conditioning_channels=1,
)
pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to("cuda")

# Load image
image = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog_mask.png"
)

# Set args
width = 1024
height = 1024
prompt="A cat is sitting next to a puppy."
generator = torch.Generator(device="cuda").manual_seed(24)

# Inference
res_image = pipe(
    negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
    prompt=prompt,
    height=height,
    width=width,
    control_image = image,
    control_mask = mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]

res_image.save(f'sd3.png')
