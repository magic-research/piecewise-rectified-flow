import random, argparse, os
from pathlib import Path
import numpy as np
import torch, torchvision
from diffusers import DiffusionPipeline, StableDiffusionPipeline

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

Path("demo").mkdir(parents=True, exist_ok=True) 


prompts_list = [
    ["RAW photo, 8k uhd, dslr, high quality, film grain; A man with brown skin and a beard, looking at the viewer with dark eyes, in front of lake",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["RAW photo, 8k uhd, dslr, high quality, film grain; A closeup face photo of girl, wearing a rain coat, in the street, heavy rain, bokeh",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["RAW photo, 8k uhd, dslr, high quality, film grain; an elegant table top with a vase of vibrant, mixed flowers, soft daylight illuminating the scene",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["RAW photo, 8k uhd, dslr, high quality, film grain; a plate of fruit on a rustic wooden table, low-contrast",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["RAW photo, 8k uhd, dslr, high quality, film grain; A beautiful cat bask in the sun",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["RAW photo, 8k uhd, dslr, high quality, film grain; A colorful bird standing on the tree stick, open beak",  "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, dark",],
    ["(masterpiece, best quality:1.3), 8k, hd, no humans, mountain, snowcap, sunset, close-up,",  "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, humans, animal,"],
    ["high quality, high resolution, extreme detail, masterpiece, forest, lake, moonlit trees, moon, <lora:Moon_LoRA:1>, volumetric shading",  "lowres, bad-hands-5, easynegative"],
]


pipe = StableDiffusionPipeline.from_pretrained("hansyan/perflow-sd21-artius", torch_dtype=torch.float16,)
from src.scheduler_perflow import PeRFlowScheduler
pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="velocity", num_time_windows=4,)
pipe.to("cuda", torch.float16)


num_inference_steps = 8
cfg_scale_list = [7.5]

for cfg_scale in cfg_scale_list:
    for i, prompts in enumerate(prompts_list):
        setup_seed(42)
        prompt, neg_prompt = prompts[0], prompts[1]
        samples = pipe(
            prompt              = [prompt] * 8, 
            negative_prompt     = [neg_prompt] * 8,
            height              = 768,
            width               = 768,
            num_inference_steps = num_inference_steps, 
            guidance_scale      = cfg_scale,
            output_type         = 'pt',
        ).images
        cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
        save_name = f'txt{i+1}_step{num_inference_steps}_cfg{cfg_int}-{cfg_float}.png'
        torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow = 4), os.path.join("demo", save_name))