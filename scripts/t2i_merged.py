import random, argparse, os
from pathlib import Path
import numpy as np
import torch
import torchvision
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


Path("demo").mkdir(parents=True, exist_ok=True) 

## >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###
## PeRFlow-DreamShaper
## >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###

## Load model and sampling scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "hansyan/piecewise-rectified-flow-dreamshaper",
    torch_dtype=torch.float16,
)
from src.scheduler_perflow import PeRFlowScheduler
pipe.scheduler = PeRFlowScheduler(
    num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
    beta_start = pipe.scheduler.config.beta_start,
    beta_end = pipe.scheduler.config.beta_end,
    beta_schedule = pipe.scheduler.config.beta_schedule,
    prediction_type="epsilon",
    num_time_windows=4,
)
pipe.to("cuda", torch.float16)


## Sampling
prompt_prefix = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; "
neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
prompts_list = [
    [prompt_prefix+"A young woman with a crown and a masterpiece necklace, at a royal event.", neg_prompt],
    [prompt_prefix+"A man with brown skin and a beard, looking at the viewer with dark eyes.", neg_prompt],
    # [prompt_prefix+"A colorful bird standing on the tree stick, open beak", neg_prompt],
    # [prompt_prefix+"A majestic tiger with blazing white fur, baring its sharp teeth in a ferocious roar.", neg_prompt],
    # ["masterpiece, best quality, red ice, red glacier, mountain, blue water",
    #  "(worst quality, low quality, lowres), blurry, bokeh, depth of field, error, censored, bar censor, text, speech bubble, artist name, signature, border, sketch, too dark",],
    # ["hyper detailed masterpiece, dynamic realistic digital art, awesome quality,Impenetrable brook,commitment wilderness,unadorned pond anti-aliasing,synchronicity,nautical,blood,wildfire,vaporizing,solar",
    #  "deformed, distorted, (disfigured:1.3), too dark, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands, mutated fingers, deformed hands, deformed fingers, extra fingers, missing fingers, extra digits, missing digits:1.6), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation,(((text))), (((watermark))),  bad-hands-5, bad-picture-chill-75v, bad_pictures, BadDream, UnrealisticDream, FastNegativeV2",],
]


num_inference_steps = 8
cfg_scale_list = [7.5]
# num_inference_steps = 4
# cfg_scale_list = [4.5, 5.0]

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









### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###
### PeRFlow-ArchitectureExterior
### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###

# ## Load model and sampling scheduler
# pipe = StableDiffusionPipeline.from_pretrained(
#     "hansyan/piecewise-rectified-flow-architectureexterior",
#     torch_dtype=torch.float16,
# )
# from src.scheduler_perflow import PeRFlowScheduler
# pipe.scheduler = PeRFlowScheduler(
#     num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
#     beta_start = pipe.scheduler.config.beta_start,
#     beta_end = pipe.scheduler.config.beta_end,
#     beta_schedule = pipe.scheduler.config.beta_schedule,
#     prediction_type="epsilon",
#     num_time_windows=4,
# )
# pipe.to("cuda", torch.float16)


# ## Sampling
# prompts_list = [
#     ["a small and beautiful modern house on a slope of a green hill, the hill has millions of tiny colorful wild flowers, blue sky as background, high details, masterpiece, highres, best quality, photo realistic, hyper detailed photo, ArchModern",
#      "low quality, normal quality, lowres, monochrome, drawing, painting, sketch, (text, signature, watermark:1.2)"],
#     ["A futuristic and stunningly beautiful high-rise shopping center architectural structure with bold, futuristic design elements, blending seamlessly into the art form of digital illustration. Inspired by the works of Syd Mead. The scene showcases the center amidst a bustling city, its sleek lines contrasting with the urban environment. A warm color temperature adds vibrancy, highlighting the architectural details. ",
#      "(normal quality), (low quality), (worst quality), paintings, dark, sketches,fog,signature,soft, blurry,drawing,sketch, poor quality, uply text,type, word, logo, pixelated, low resolution.,saturated,high contrast, oversharpened,dirt,"],
# ]

# num_inference_steps = 8
# cfg_scale_list = [7.5]

# for i, prompts in enumerate(prompts_list):
#     for cfg_scale in cfg_scale_list:
#         setup_seed(42)
#         prompt, neg_prompt = prompts[0], prompts[1]
#         samples = pipe(
#             prompt              = [prompt] * 8, 
#             negative_prompt     = [neg_prompt] * 8,
#             height              = 512,
#             width               = 512,
#             num_inference_steps = num_inference_steps, 
#             guidance_scale      = cfg_scale,
#             output_type         = 'pt',
#         ).images

#         cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
#         save_name = f'txt{i+1}_step{num_inference_steps}_cfg{cfg_int}-{cfg_float}.png'
#         torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow = 4), os.path.join("demo", save_name))









### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###
### PeRFlow-realisticVisionV51
### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###

# ## Load model and sampling scheduler
# pipe = StableDiffusionPipeline.from_pretrained(
#     "hansyan/piecewise-rectified-flow-realisticVisionV51",
#     torch_dtype=torch.float16,
# )
# from src.scheduler_perflow import PeRFlowScheduler
# pipe.scheduler = PeRFlowScheduler(
#     num_train_timesteps=pipe.scheduler.config.num_train_timesteps,
#     beta_start = pipe.scheduler.config.beta_start,
#     beta_end = pipe.scheduler.config.beta_end,
#     beta_schedule = pipe.scheduler.config.beta_schedule,
#     prediction_type="epsilon",
#     num_time_windows=4,
# )
# pipe.to("cuda", torch.float16)


# ## Sampling
# # prompt_prefix = "RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; "
# # neg_prompt = "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark"
# prompts_list = [
#     ["instagram photo, closeup face photo of 18 y.o swedish woman in dress, beautiful face, makeup, bokeh, motion blur",
#         "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), too dark, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",],
#     ["A face portrait photo of beautiful 26 year woman, cute face, wearing gray dress, happy face, cinematic shot, dramatic lighting",
#         "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), too dark, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation",],
#     ["RAW photo, 8k uhd, dslr, high quality, film grain, highly detailed, masterpiece; a plate of fruit on a rustic wooden table, low-contrast",
#         "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast, out of focus, dark",],
#     ["masterpiece, best quality, highres, a shiny car with a white number plate, white paint, realistic, sunset, gradient sky, quarry area",
#         "worst quality, low quality, normal quality, ugly woman, error, unfinished, sketch, illustration, too dark",],
# ]

# num_inference_steps = 8
# cfg_scale_list = [7.5]

# for i, prompts in enumerate(prompts_list):
#     for cfg_scale in cfg_scale_list:
#         setup_seed(42)
#         prompt, neg_prompt = prompts[0], prompts[1]
#         samples = pipe(
#             prompt              = [prompt] * 8, 
#             negative_prompt     = [neg_prompt] * 8,
#             height              = 512,
#             width               = 512,
#             num_inference_steps = num_inference_steps, 
#             guidance_scale      = cfg_scale,
#             output_type         = 'pt',
#         ).images

#         cfg_int = int(cfg_scale); cfg_float = int(cfg_scale*10 - cfg_int*10)
#         save_name = f'txt{i+1}_step{num_inference_steps}_cfg{cfg_int}-{cfg_float}.png'
#         torchvision.utils.save_image(torchvision.utils.make_grid(samples, nrow = 4), os.path.join("demo", save_name))