import random, argparse, os
from pathlib import Path
import numpy as np
import torch, torchvision

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

Path("demo").mkdir(parents=True, exist_ok=True) 


from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained("hansyan/perflow-sdxl-dreamshaper", torch_dtype=torch.float16, use_safetensors=True, variant="v0-fix")
from src.scheduler_perflow import PeRFlowScheduler
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4)
pipe.to("cuda", torch.float16)


num_inference_steps = 6  # suggest steps >= num_win=4
cfg_scale_list = [2.0]  # suggest values [1.5, 2.0, 2.5]
num_img = 2
seed = 42
prompts_list = [
    ["photorealistic, uhd, high resolution, high quality, highly detailed; RAW photo, a handsome man, wearing a black coat, outside, closeup face",
        "distorted, blur, low-quality, haze, out of focus",],
    ["photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, A closeup face photo of girl, wearing a rain coat, in the street, heavy rain, bokeh,",
        "distorted, blur, low-quality, haze, out of focus",],
    ["photorealistic, uhd, high resolution, high quality, highly detailed; RAW photo, a red luxury car, studio light",
        "distorted, blur, low-quality, haze, out of focus",],
    ["photorealistic, uhd, high resolution, high quality, highly detailed; masterpiece, A beautiful cat bask in the sun",
        "distorted, blur, low-quality, haze, out of focus",],
]


for cfg_scale in cfg_scale_list:
    for i, prompts in enumerate(prompts_list):
        setup_seed(seed)
        prompt, neg_prompt = prompts[0], prompts[1]
        samples = pipe(
            prompt              = [prompt] * num_img, 
            negative_prompt     = [neg_prompt] * num_img,
            height              = 1024,
            width               = 1024,
            num_inference_steps = num_inference_steps, 
            guidance_scale      = cfg_scale,
            output_type         = 'pt',
        ).images
        
        cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
        save_name = f'step_{num_inference_steps}_txt{i+1}_cfg{cfg_int}-{cfg_float}.png'
        torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow = num_img), os.path.join("demo", save_name))