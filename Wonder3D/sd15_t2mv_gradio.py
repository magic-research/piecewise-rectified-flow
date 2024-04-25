import torch
import requests
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from diffusers import DiffusionPipeline  # only tested on diffusers[torch]==0.19.3, may have conflicts with newer versions of diffusers

import sys
sys.path.insert(0, sys.path[0]+"/../")
import src
from src.scheduler_perflow import PeRFlowScheduler
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

import torchvision
import rembg
import imageio
import gradio as gr

def merge_delta_weights_into_unet(pipe, delta_weights, org_alpha = 1.0):
    unet_weights = pipe.unet.state_dict()
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        try:
            unet_weights[key] = org_alpha * unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        except:
            unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe


def load_wonder3d_pipeline():

    pipeline = DiffusionPipeline.from_pretrained(
    'flamehaze1115/wonder3d-v1.0', # or use local checkpoint './ckpts'
    custom_pipeline='flamehaze1115/wonder3d-pipeline',
    torch_dtype=torch.float16
    )

    # enable xformers
    pipeline.unet.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

pipe_3d = load_wonder3d_pipeline()
pipe_t2i = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8", torch_dtype=torch.float16, safety_checker=None)

### PeRFlow
delta_weights = UNet2DConditionModel.from_pretrained("hansyan/perflow-sd15-delta-weights", torch_dtype=torch.float16, variant="v0-1",).state_dict()

pipe_t2i = merge_delta_weights_into_unet(pipe_t2i, delta_weights)
pipe_t2i.scheduler = PeRFlowScheduler.from_config(pipe_t2i.scheduler.config, prediction_type="diff_eps", num_time_windows=4)
pipe_t2i.to('cuda:0', torch.float16)

pipe_3d = merge_delta_weights_into_unet(pipe_3d, delta_weights)
pipe_3d.scheduler = PeRFlowScheduler.from_config(pipe_3d.scheduler.config, prediction_type="epsilon", num_time_windows=4)
pipe_3d.to('cuda:0', torch.float16)

def generate_gif(prompt):
    samples = pipe_t2i(
            prompt              = [prompt],
            negative_prompt     = [""],
            height              = 512,
            width               = 512,
            num_inference_steps = 4,
            guidance_scale      = 4.5,
            output_type         = 'pt',
        ).images
    samples = torch.nn.functional.interpolate(samples, size=256, mode='bilinear')
    samples = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()*255.
    samples = samples.astype(np.uint8)

    samples = rembg.remove(samples)
    # import pdb; pdb.set_trace()
    samples = samples.copy()
    samples[:, :, 0][samples[:, :, -1]<=5] = 255
    samples[:, :, 1][samples[:, :, -1]<=5] = 255
    samples[:, :, 2][samples[:, :, -1]<=5] = 255

    # The object should be located in the center and resized to 80% of image height.
    cond = Image.fromarray(samples[:, :, :3])

    # Run the pipeline!
    images = pipe_3d(cond, num_inference_steps=1, output_type='pt', guidance_scale=1.0).images
    images = (images * 255).int()

    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = images.transpose(0, 2, 1, 3)
    images_norm=images[:6].reshape(-1, 256, 3).transpose(1, 0, 2)
    images_out=images[6:].reshape(-1, 256, 3).transpose(1, 0, 2)
    return images_norm, images_out

with gr.Blocks() as demo:
    with gr.Column():
        out1 = gr.Image(width=1024)
        out2 = gr.Image(width=1024)
        text = gr.Textbox(label="Input")

    text.submit(
        generate_gif,
        [text], # input
        [out1, out2], # output
    )
demo.dependencies[0]["show_progress"] = False  # the hack
demo.queue().launch()


if __name__ == "__main__":
    demo.launch()
