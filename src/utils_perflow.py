import os
from collections import OrderedDict
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint
    

def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()
    assert unet_weights.keys() == delta_weights.keys()
    for key in delta_weights.keys():
        dtype = unet_weights[key].dtype
        unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
        unet_weights[key] = unet_weights[key].to(dtype)
    pipe.unet.load_state_dict(unet_weights, strict=True)
    return pipe
