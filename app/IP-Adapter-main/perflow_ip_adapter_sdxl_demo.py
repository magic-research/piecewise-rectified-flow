# %%
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterXL

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

device = "cuda"

# %%
# load SDXL pipeline
base_model_path = "hansyan/perflow-sdxl-base"
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    variant="v0-fix",
    torch_dtype=torch.float16,
    add_watermarker=False,
)

import sys
sys.path.append('../../')
from src.utils_perflow import load_delta_weights_into_unet
from src.scheduler_perflow import PeRFlowScheduler
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4,)


# %%
# load ip-adapter
image_encoder_path = "./sdxl_models/image_encoder"
ip_ckpt = "./sdxl_models/ip-adapter_sdxl.bin"
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

# %%
# read image prompt
# image = Image.open("assets/images/woman.png")
image = Image.open("assets/images/girl.png")
# image = Image.open("assets/images/ai_face2.png")
# image = Image.open("assets/images/statue.png")
image.resize((512, 512))

# %%
# multimodal prompts
num_samples = 2
images = ip_model.generate(pil_image=image, num_samples=num_samples, num_inference_steps=6, seed=420, scale=0.6,
        negative_prompt="distorted, blur, low-quality, haze, out of focus", guidance_scale=1.5,
        prompt="best quality, high quality, on the beach", 
        )
grid = image_grid(images, 1, num_samples)
grid.save('tmp.png')


