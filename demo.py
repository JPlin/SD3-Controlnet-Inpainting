from diffusers.utils import load_image, check_min_version
import torch

# Local File
from pipeline_sd3_controlnet_inpainting import StableDiffusion3ControlNetInpaintingPipeline, one_image_and_mask
from controlnet_sd3 import SD3ControlNetModel

check_min_version("0.29.2")

# Build model
controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting",
    use_safetensors=True,
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
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/blob/main/images/prod.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/blob/main/images/mask.jpeg"
)

# Set args
width = 1024
height = 1024
prompt="a woman wearing a white jacket, black hat and black pants is standing in a field, the hat writes SD3"
generator = torch.Generator(device="cuda").manual_seed(24)
input_dict = one_image_and_mask(image, mask, size=(width, height), latent_scale=pipe.vae_scale_factor, invert_mask = True)

# Inference
res_image = pipe(
    negative_prompt='deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW',
    prompt=prompt,
    height=height,
    width=width,
    control_image= input_dict['pil_masked_image'],  # H, W, C,
    control_mask=input_dict["mask"] > 0.5,  # B,1,H,W
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]

res_image.save(f'res.png')
