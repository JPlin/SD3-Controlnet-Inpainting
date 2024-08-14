# Updates

✨🎉 This model has been merged into [Diffusers](https://moon-ci-docs.huggingface.co/docs/diffusers/pr_9099/en/api/pipelines/controlnet_sd3) and can now be used conveniently. 💡 🎉✨

# Examples

![SD3](images/sd3_compressed.png)

<center><i>a woman wearing a white jacket, black hat and black pants is standing in a field, the hat writes SD3</i></center>

![bucket_alibaba](images/bucket_ali_compressed.png )

<center><i>a person wearing a white shoe, carrying a white bucket with text "alibaba" on it</i></center>

## SD3 Controlnet Inpainting

Finetuned controlnet inpainting model based on sd3-medium, the inpainting model offers several advantages:

* Leveraging the SD3 16-channel VAE and high-resolution generation capability at 1024, the model effectively preserves the integrity of non-inpainting regions, including text.

* It is capable of generating text through inpainting.

* It demonstrates superior aesthetic performance in portrait generation.

Compared with [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1)

From left to right: Input image, Masked image, SDXL inpainting, Ours.

![0](images/0_compressed.png)
<center><i>a tiger sitting on a park bench</i></center>

![1](images/0r_compressed.png)
<center><i>a dog sitting on a park bench</i></center>

![2](images/1_compressed.png)
<center><i>a young woman wearing a blue and pink floral dress</i></center>

![3](images/3_compressed.png)
<center><i>a woman wearing a white jacket, black hat and black pants is standing in a field, the hat writes SD3</i></center>

![4](images/5_compressed.png)
<center><i>an air conditioner hanging on the bedroom wall</i></center>

# Using with Diffusers

Install from source and Run

``` Shell
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers
```

``` python
import torch
from diffusers.utils import load_image, check_min_version
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from diffusers.models.controlnet_sd3 import SD3ControlNetModel

controlnet = SD3ControlNetModel.from_pretrained(
    "alimama-creative/SD3-Controlnet-Inpainting", use_safetensors=True, extra_conditioning_channels=1
)
pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet=controlnet,
    torch_dtype=torch.float16,
)
pipe.text_encoder.to(torch.float16)
pipe.controlnet.to(torch.float16)
pipe.to("cuda")

image = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog.png"
)
mask = load_image(
    "https://huggingface.co/alimama-creative/SD3-Controlnet-Inpainting/resolve/main/images/dog_mask.png"
)
width = 1024
height = 1024
prompt = "A cat is sitting next to a puppy."
generator = torch.Generator(device="cuda").manual_seed(24)
res_image = pipe(
    negative_prompt="deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
    prompt=prompt,
    height=height,
    width=width,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    generator=generator,
    controlnet_conditioning_scale=0.95,
    guidance_scale=7,
).images[0]
res_image.save(f"sd3.png")
```


## Training Detail

The model was trained on 12M laion2B and internal source images for 20k steps at resolution 1024x1024. 

* Mixed precision : FP16
* Learning rate : 1e-4
* Batch size : 192
* Timestep sampling mode : 'logit_normal'
* Loss : Flow Matching

## Limitation

Due to the fact that only 1024*1024 pixel resolution was used during the training phase, the inference performs best at this size, with other sizes yielding suboptimal results. We will initiate multi-resolution training in the future, and at that time, we will open-source the new weights.

## LICENSE
The model is based on SD3 finetuning; therefore, the license follows the original [SD3 license](https://huggingface.co/stabilityai/stable-diffusion-3-medium#license).
