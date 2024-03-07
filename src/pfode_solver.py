import os, math, random, argparse, logging
from pathlib import Path
from typing import Optional, Union, List, Callable
from collections import OrderedDict
from packaging import version
from tqdm.auto import tqdm
from omegaconf import OmegaConf
        
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision


class PFODESolver():
    def __init__(self, scheduler, t_initial=1, t_terminal=0,) -> None:
        self.t_initial = t_initial
        self.t_terminal = t_terminal
        self.scheduler = scheduler

        train_step_terminal = 0 
        train_step_initial = train_step_terminal + self.scheduler.config.num_train_timesteps # 0+1000
        self.stepsize  = (t_terminal-t_initial) / (train_step_terminal - train_step_initial) #1/1000
    
    def get_timesteps(self, t_start, t_end, num_steps):
        # (b,) -> (b,1)
        t_start = t_start[:, None]
        t_end = t_end[:, None]
        assert t_start.dim() == 2
        
        timepoints = torch.arange(0, num_steps, 1).expand(t_start.shape[0], num_steps).to(device=t_start.device)
        interval = (t_end - t_start) / (torch.ones([1], device=t_start.device) * num_steps)
        timepoints = t_start + interval * timepoints
        
        timesteps = (self.scheduler.num_train_timesteps - 1) + (timepoints - self.t_initial) / self.stepsize # correspondint to StableDiffusion indexing system, from 999 (t_init) -> 0 (dt)
        return timesteps.round().long()
    
    def solve(self, 
              latents, 
              unet, 
              t_start, 
              t_end, 
              prompt_embeds, 
              negative_prompt_embeds, 
              guidance_scale=1.0,
              num_steps = 2,
              num_windows = 1,
    ):
        assert t_start.dim() == 1
        assert guidance_scale >= 1 and torch.all(torch.gt(t_start, t_end))
        
        do_classifier_free_guidance = True if guidance_scale > 1 else False
        bsz = latents.shape[0]
            
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        timestep_cond = None
        if unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(bsz)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=unet.config.time_cond_proj_dim
            ).to(device=latents.device, dtype=latents.dtype)
            
        
        timesteps = self.get_timesteps(t_start, t_end, num_steps).to(device=latents.device)
        timestep_interval = self.scheduler.config.num_train_timesteps // (num_windows * num_steps)

        # Denoising loop
        with torch.no_grad():
            for i in range(num_steps):
                t = torch.cat([timesteps[:, i]]*2) if do_classifier_free_guidance else timesteps[:, i]
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                ##### STEP: compute the previous noisy sample x_t -> x_t-1
                batch_timesteps = timesteps[:, i].cpu()
                prev_timestep = batch_timesteps - timestep_interval

                alpha_prod_t = self.scheduler.alphas_cumprod[batch_timesteps]
                alpha_prod_t_prev = torch.zeros_like(alpha_prod_t)
                for ib in range(prev_timestep.shape[0]): 
                    alpha_prod_t_prev[ib] = self.scheduler.alphas_cumprod[prev_timestep[ib]] if prev_timestep[ib] >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                
                alpha_prod_t = alpha_prod_t.to(device=latents.device, dtype=latents.dtype)
                alpha_prod_t_prev = alpha_prod_t_prev.to(device=latents.device, dtype=latents.dtype)
                beta_prod_t = beta_prod_t.to(device=latents.device, dtype=latents.dtype)

                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (latents - beta_prod_t[:,None,None,None] ** (0.5) * noise_pred) / alpha_prod_t[:, None,None,None] ** (0.5)
                    pred_epsilon = noise_pred
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t[:,None,None,None]**0.5) * latents - (beta_prod_t[:,None,None,None]**0.5) * noise_pred
                    pred_epsilon = (alpha_prod_t[:,None,None,None]**0.5) * noise_pred + (beta_prod_t[:,None,None,None]**0.5) * latents
                else:
                    raise NotImplementedError
                    
                pred_sample_direction = (1 - alpha_prod_t_prev[:,None,None,None]) ** (0.5) * pred_epsilon
                latents = alpha_prod_t_prev[:,None,None,None] ** (0.5) * pred_original_sample + pred_sample_direction

            
        return latents
    
    
    
    