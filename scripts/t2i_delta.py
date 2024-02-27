import random, argparse, os
from pathlib import Path
import numpy as np
import torch, torchvision
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


## Load model and sampling scheduler
from src.utils_perflow import merge_delta_weights_into_unet
from src.scheduler_perflow import PeRFlowScheduler
delta_weights = UNet2DConditionModel.from_pretrained("hansyan/piecewise-rectified-flow-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()
pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16,)
pipe = merge_delta_weights_into_unet(pipe, delta_weights)
pipe.scheduler = PeRFlowScheduler(
    num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
    beta_start = pipe.scheduler.config.beta_start,
    beta_end = pipe.scheduler.config.beta_end,
    beta_schedule = pipe.scheduler.config.beta_schedule,
    prediction_type="epsilon",
    num_time_windows=4,
)
# pipe.scheduler = PeRFlowScheduler.from_config(pipe.scheduler.config, prediction_type="epsilon", num_time_windows=4,)
pipe.to("cuda", torch.float16)


## Sampling
num_inference_steps = 8
cfg_scale_list = [7.5]
# num_inference_steps = 4
# cfg_scale_list = [4.5, 5.0]

prompt_prefix = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; "
neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
prompts_list = [
    [prompt_prefix+"A young woman with a crown and a masterpiece necklace, at a royal event.", neg_prompt],
    [prompt_prefix+"A man with brown skin and a beard, looking at the viewer with dark eyes.", neg_prompt],
    [prompt_prefix+"A colorful bird standing on the tree stick, open beak", neg_prompt],
    [prompt_prefix+"A majestic tiger with blazing white fur, baring its sharp teeth in a ferocious roar.", neg_prompt],
    ["masterpiece, best quality, red ice, red glacier, mountain, blue water",
     "(worst quality, low quality, lowres), blurry, bokeh, depth of field, error, censored, bar censor, text, speech bubble, artist name, signature, border, sketch, too dark",],
    ["a small and beautiful modern house on a slope of a green hill, the hill has millions of tiny colorful wild flowers, blue sky as background, high details, masterpiece, highres, best quality, photo realistic, hyper detailed photo, ArchModern",
        "low quality, normal quality, lowres, monochrome, drawing, painting, sketch, (text, signature, watermark:1.2)"],
]

Path("demo").mkdir(parents=True, exist_ok=True) 

for i, prompts in enumerate(prompts_list):
    for cfg_scale in cfg_scale_list:
        setup_seed(42)
        prompt, neg_prompt = prompts[0], prompts[1]
        samples = pipe(
            prompt              = [prompt] * 8, 
            negative_prompt     = [neg_prompt] * 8,
            height              = 512,
            width               = 512,
            num_inference_steps = num_inference_steps, 
            guidance_scale      = cfg_scale,
            output_type         = 'pt',
        ).images
        cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
        save_name = f'txt{i+1}_step{num_inference_steps}_cfg{cfg_int}-{cfg_float}.png'
        torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow = 4), os.path.join("demo", save_name))


