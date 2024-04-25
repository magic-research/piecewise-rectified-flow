import os, math, random, argparse, logging, copy
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

from huggingface_hub import HfFolder, Repository, create_repo, whoami
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import datasets
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, AutoTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from safetensors import safe_open

def convert_peft_keys_to_kohya(module, prefix: str, dtype: torch.dtype, adapter_name: str = "default"):
    # convert peft lora to kohya lora (diffsuers support kohya lora naming)
    kohya_ss_state_dict = {}
    for peft_key, weight in get_peft_model_state_dict(module, adapter_name=adapter_name).items():
        kohya_key = peft_key.replace("base_model.model", prefix)
        kohya_key = kohya_key.replace("lora_A", "lora_down")
        kohya_key = kohya_key.replace("lora_B", "lora_up")
        kohya_key = kohya_key.replace(".", "_", kohya_key.count(".") - 2)
        kohya_ss_state_dict[kohya_key] = weight.to(dtype)
        # Set alpha parameter
        if "lora_down" in kohya_key:
            alpha_key = f'{kohya_key.split(".")[0]}.alpha'
            kohya_ss_state_dict[alpha_key] = torch.tensor(module.peft_config[adapter_name].lora_alpha).to(dtype)
    return kohya_ss_state_dict

def _compute_time_ids(original_size, crops_coords_top_left, target_size, dtype):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    return add_time_ids


def main(args):
    ## 0. misc
    logger = get_logger(__name__, log_level="INFO") # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.report_to),)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=args.report_to, project_config=accelerator_project_config,)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    ## 1. Prepare models, noise scheduler, tokenizer.
    logger.info("***** preparing models *****")
    
    tokenizer_1 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False,)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False,)
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_1.requires_grad_(False)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_2.requires_grad_(False)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]
    
    if args.pretrained_vae_name_or_path is not None:
        vae = AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path, torch_dtype=weight_dtype)
        print("***************** using fixed fp16 vae for sdxl")
    else:
        # if vae.config.force_upcast: The VAE is in float32 to avoid NaN losses. 
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32)
    vae.requires_grad_(False)
    vae.to(accelerator.device)

    ## 1.1 Prepare teacher
    teacher_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",)
    teacher_unet.requires_grad_(False)
    teacher_unet.to(accelerator.device, dtype=weight_dtype)
    
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    teacher_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    teacher_num_train_timesteps = teacher_scheduler.config.num_train_timesteps
    from src.pfode_solver import PFODESolverSDXL
    solver = PFODESolverSDXL(scheduler=teacher_scheduler, t_initial=1, t_terminal=0,)
    
    ## 1.2 Prepare student
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",)
    if args.unet_model_path != "":
        _tmp_ = OrderedDict()
        assert args.unet_model_path.endswith(".safetensors")
        with safe_open(args.unet_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                _tmp_[key] = f.get_tensor(key)
        missing, unexpected = unet.load_state_dict(_tmp_, strict=False)
        assert len(unexpected) == 0
        del _tmp_
    
    if weight_dtype == torch.bfloat16:
        unet.to(accelerator.device, dtype=weight_dtype)
    else:
        unet.to(accelerator.device)
        
    use_lora = args.lora_rank > 0
    if use_lora: # default -1, not using lora
        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "proj_in",
                "proj_out",
                "ff.net.0.proj",
                "ff.net.2",
                "conv1",
                "conv2",
                "conv_shortcut",
                "downsamplers.0.conv",
                "upsamplers.0.conv",
                "time_emb_proj",
            ],
        )
        unet = get_peft_model(unet, lora_config)
        
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    if args.use_ema: # Create EMA for the unet.
        assert not use_lora
        ema_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet",)
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)
        ema_unet.to(accelerator.device)
        
    from src.scheduler_perflow import PeRFlowScheduler
    perflow_scheduler = PeRFlowScheduler(
        num_train_timesteps=teacher_scheduler.config.num_train_timesteps,
        beta_start = teacher_scheduler.config.beta_start,
        beta_end = teacher_scheduler.config.beta_end,
        beta_schedule = teacher_scheduler.config.beta_schedule,
        prediction_type=args.pred_type,
        t_noise = 1,
        t_clean = 0,
        num_time_windows=args.windows,
    )
    
    
    ## 2. Prepare dataset    
    logger.info("***** PREPARE YOUR OWN DATASETS *****")
    train_dataset = None
    train_dataloader = None
    
    if args.support_cfg:
        cfg_drop_ratio = 0.1
    else:
        cfg_drop_ratio = 0.


    ## 3. Optimization
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
        
    optimizer = optimizer_class(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    

    ## 4. build validation pipeline
    logger.info("***** building validation pipeline *****")
    Path(f"{args.output_dir}/samples").mkdir(parents=True, exist_ok=True)
    
    def log_validation(
        accelerator, vae, unet, weight_dtype,
        height=512, width=512, num_inference_steps=5, guidance_scale=3.0,
        log_dir="", global_step=0, rank=0, sanity_check=False,
        use_lora = False,
        ):
        num_inference_steps = num_inference_steps.split("-")
        num_inference_steps = [int(x) for x in num_inference_steps]
        guidance_scale = guidance_scale.split("-")
        guidance_scale = [float(x) for x in guidance_scale]
        
        vae_dtype = vae.dtype
        
        with torch.no_grad():
            prompts = [
                # ["RAW photo, 8k uhd, dslr, high quality, film grain; a man",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a man",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a woman",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a dog",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a cat",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; green grassland and blue sky",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; mountains and trees",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a desk and a chair",   ""],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a car on the road",   ""],
            ]
                
            val_pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae = vae.to(weight_dtype),
                text_encoder = text_encoder_1,
                text_encoder_2 = text_encoder_2,
                tokenizer = tokenizer_1,
                tokenizer_2 = tokenizer_2,
                scheduler = perflow_scheduler,
                torch_dtype = weight_dtype,
            ).to(device=accelerator.device)
            
            if use_lora:
                lora_state_dict = convert_peft_keys_to_kohya(accelerator.unwrap_model(unet), "lora_unet", weight_dtype,)
                val_pipeline.load_lora_weights(lora_state_dict)
                val_pipeline.fuse_lora()
                val_pipeline.to(device=accelerator.device, dtype=weight_dtype)
            else:
                for src_param, val_param in zip(unet.parameters(), val_pipeline.unet.parameters()):
                    val_param.data.copy_(src_param.to(val_param.device, val_param.dtype).data)
                val_pipeline.to(device=accelerator.device, dtype=weight_dtype)
            
            # sampling
            for inf_step, cfg_scale in zip(num_inference_steps, guidance_scale):
                generator = torch.Generator(device=accelerator.device)
                generator.manual_seed(123456789 + rank)
                
                samples = []
                for _, prompt in enumerate(prompts):
                    sample = val_pipeline(
                        prompt              = prompt[0],
                        negative_prompt     = prompt[1],
                        height              = height,
                        width               = width,
                        num_inference_steps = inf_step,
                        guidance_scale      = cfg_scale,
                        generator           = generator,
                        output_type         = 'pt',
                    ).images
                    sample = torchvision.transforms.functional.resize(
                        sample, (512,512), antialias=True
                    ) # save low-res images for preview, 1024/768->512 #!!!:
                    samples.append(sample)

                samples = torchvision.utils.make_grid(torch.concat(samples), nrow=4)       
                
                if not sanity_check:
                    Path(f"{log_dir}/step_{global_step}").mkdir(parents=True, exist_ok=True)
                    save_path = f"{log_dir}/step_{global_step}/sample-{inf_step}_r{rank}.png"
                else:
                    Path(f"{log_dir}/sanity_check").mkdir(parents=True, exist_ok=True)
                    save_path = f"{log_dir}/sanity_check/sample-{inf_step}_r{rank}.png"
                torchvision.utils.save_image(samples, save_path)
                logging.info(f"Saved samples to {save_path}")
        
        del val_pipeline
        torch.cuda.empty_cache()
        vae.to(vae_dtype)
    
    
    ## 5. Prepare for training
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info( "*****  Running training *****")
    logger.info(f"*****  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"*****  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"*****  Total optimization steps = {args.max_train_steps/1000:.2f} K")
    logger.info(f"*****  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size/1000:.2f} K")
    logger.info(f"*****  Num examples = {len(train_dataset)/1_000_000:.2f} M")

    ## 5.1. Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    ## 5.2. Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        resume_global_step = int(path.split("-")[-1])
        if args.use_ema:
            _tmp_ = EMAModel.from_pretrained(os.path.join(args.resume_from_checkpoint, "unet_ema"), UNet2DConditionModel)
            ema_unet.load_state_dict(_tmp_.state_dict())
            ema_unet.to(accelerator.device)
            del _tmp_

    ## 5.3 initializing logging
    if accelerator.is_main_process:
        accelerator.init_trackers("piecewise_linear_flow")
        exp_config = OmegaConf.create(vars(args))
        exp_config['total_batch_size'] = total_batch_size
        OmegaConf.save(exp_config, os.path.join(args.output_dir, 'config.yaml'))
        
    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    global_step = 0
    if args.resume_from_checkpoint:
        while global_step < resume_global_step:
            global_step += 1
            progress_bar.update(1)
    
    ## 6. Start training
    unet.train()
    train_loss = 0.0
    sanity_check_flag = True
    
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            ## 6.0 sanity check
            if accelerator.sync_gradients:
                if sanity_check_flag:
                    if accelerator.state.process_index % 8 == 0:
                        logger.info("***** sanity checking *****")
                        log_validation(
                            accelerator             = accelerator,
                            vae                     = vae,
                            unet                    = unet,
                            weight_dtype            = weight_dtype,
                            height                  = args.resolution,
                            width                   = args.resolution,
                            num_inference_steps     = args.inference_steps,
                            guidance_scale          = args.inference_cfg,
                            log_dir                 = os.path.join(args.output_dir, 'samples'),
                            global_step             = global_step,
                            rank                    = accelerator.state.process_index,
                            sanity_check            = sanity_check_flag,
                            use_lora                = use_lora,
                        )
                    sanity_check_flag = False
                    
            with accelerator.accumulate(unet):
                bsz = batch["pixel_values"].shape[0]
                ## 6.1 prompt embeddings
                with torch.no_grad():
                    prompt_embeds_list = []
                    text_input_ids_1 = batch['input_ids'].to(accelerator.device)
                    text_input_ids_2 = batch['input_ids_2'].to(accelerator.device)
                    for text_input_ids, tokenizer, text_encoder in zip([text_input_ids_1, text_input_ids_2], tokenizers, text_encoders):
                        text_input_ids.to(accelerator.device)
                        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
                        pooled_prompt_embeds = prompt_embeds[0] # (b, 1280) We are only ALWAYS interested in the pooled output of the final text encoder
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        prompt_embeds_list.append(prompt_embeds)
                    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1) # (b, 77, 2048)
                    
                    null_prompt_embeds_list = []     
                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                        null_text_ids = tokenizer([""]*bsz, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt",)
                        null_text_ids = null_text_ids.input_ids
                        null_text_ids = null_text_ids.to(accelerator.device)
                        null_prompt_embeds = text_encoder(null_text_ids, output_hidden_states=True,)
                        pooled_null_prompt_embeds = null_prompt_embeds[0] # We are only ALWAYS interested in the pooled output of the final text encoder
                        null_prompt_embeds = null_prompt_embeds.hidden_states[-2]
                        null_prompt_embeds_list.append(null_prompt_embeds)
                    null_prompt_embeds = torch.concat(null_prompt_embeds_list, dim=-1) # (b, 77, 2048)

                    if args.support_cfg:
                        mask_text = torch.zeros((bsz,1,1), dtype=prompt_embeds.dtype, device=prompt_embeds.device)
                        for i in range(bsz):
                            mask_text[i] = 1 if random.random() > cfg_drop_ratio else 0                        
                        prompt_embeds_dropout =  mask_text * prompt_embeds + (1-mask_text) * null_prompt_embeds
                        pooled_prompt_embeds_dropout = mask_text[:,:,0] * pooled_prompt_embeds + (1-mask_text[:,:,0]) * pooled_null_prompt_embeds
                    else:
                        prompt_embeds_dropout = prompt_embeds
                        pooled_prompt_embeds_dropout = pooled_prompt_embeds
                    
                    ## 6.3 Prepare additional conditions
                    add_time_ids = torch.cat(
                        [_compute_time_ids((args.resolution, args.resolution), (0, 0), (args.resolution, args.resolution), weight_dtype) for _ in range(bsz)]
                    ).to(accelerator.device)
                
                    unet_added_conditions = {"time_ids": add_time_ids}
                    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds_dropout})


                if args.train_mode == 'perflow':
                    ## 6.2 Prepare input latents, sample timesteps. NOTE: t \in [1, 0]
                    ## Prepare the endpoints of windows, model inputs
                    with torch.no_grad():
                        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype) # (b c h w)
                        if args.pretrained_vae_name_or_path is None:
                            pixel_values = pixel_values.float() # if vae.config.force_upcast: The VAE is in float32 to avoid NaN losses.
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        latents = latents.to(dtype=weight_dtype)
                        
                        timepoints = torch.rand((bsz,), device=latents.device) # [0,1)
                        if args.discrete_timesteps == -1:
                            timepoints = (timepoints * teacher_num_train_timesteps).floor() / teacher_num_train_timesteps # assert [0, 999/1000]
                        else:
                            assert isinstance(args.discrete_timesteps, int)
                            timepoints = (timepoints * args.discrete_timesteps).floor() / args.discrete_timesteps # in [0, 39/40)
                        timepoints = 1 - timepoints # [1, 1/1000], [1, 1/40]
                        
                        t_start, t_end = perflow_scheduler.time_windows.lookup_window(timepoints)
                        
                        noises = torch.randn_like(latents)
                        latents_start = teacher_scheduler.add_noise(latents, noises, torch.clamp((t_start*teacher_num_train_timesteps).long()-1, min=0))
                        ### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###
                        if args.zero_snr:
                            latents_start_mask = ((t_start==1.0)*1.0)[:,None,None,None].to(accelerator.device, dtype=weight_dtype)
                            latents_start = latents_start * (1 - latents_start_mask) + (latents_start_mask) * noises
                        ### >>>>>>>> >>>>>>>> ### >>>>>>>> >>>>>>>> ###
                        
                        if args.cfg_sync:
                            solver_prompt_embedes = prompt_embeds_dropout
                            solver_pooled_prompt_embeds = pooled_prompt_embeds_dropout
                            solver_cfg_scale = 1.0
                        else:
                            solver_prompt_embedes = prompt_embeds
                            solver_pooled_prompt_embeds = pooled_prompt_embeds
                            solver_cfg_scale = 7.5
                        
                        latents_end = solver.solve(
                            latents                         = latents_start, 
                            t_start                         = t_start,
                            t_end                           = t_end,
                            unet                            = teacher_unet, 
                            prompt_embeds                   = solver_prompt_embedes,
                            pooled_prompt_embeds            = solver_pooled_prompt_embeds,
                            negative_prompt_embeds          = null_prompt_embeds, 
                            negative_pooled_prompt_embeds   = pooled_null_prompt_embeds,
                            guidance_scale                  = solver_cfg_scale, 
                            num_steps                       = args.solving_steps,
                            num_windows                     = args.windows,
                            resolution                      = args.resolution,
                        )

                        latents_t = latents_start + (latents_end - latents_start) / (t_end[:,None,None,None] - t_start[:,None,None,None]) * (timepoints[:, None, None, None] - t_start[:, None, None, None])
                        latents_t = latents_t.to(weight_dtype)

        
                    ## 6.4 prepare targets -> perform inference -> convert -> compute loss
                    with torch.no_grad():
                        if args.loss_type == "noise_matching" and args.pred_type == "ddim_eps":
                            _, _, _, _, _, alphas_cumprod_start, alphas_cumprod_end = perflow_scheduler.get_window_alpha(timepoints.float().cpu())
                            alphas_cumprod_start = alphas_cumprod_start[:,None,None,None].to(device=latents.device)
                            alphas_cumprod_end = alphas_cumprod_end[:,None,None,None].to(device=latents.device)
                            lambda_s = (alphas_cumprod_end / alphas_cumprod_start)**0.5
                            eta_s = (1-alphas_cumprod_end)**0.5 - ( alphas_cumprod_end / alphas_cumprod_start * (1-alphas_cumprod_start) )**0.5
                            targets = (latents_end - lambda_s * latents_start ) / eta_s
                        else:
                            raise NotImplementedError
                        
                    model_pred = unet(
                        latents_t, 
                        timepoints.float() * teacher_num_train_timesteps, 
                        encoder_hidden_states=prompt_embeds_dropout,
                        added_cond_kwargs=unet_added_conditions,
                    ).sample
  
                    if args.reweighting_scheme is None:
                        if args.loss_distance == "l2":
                            loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")
                        elif args.loss_distance == "huber":
                            huber_c = args.huber_c
                            loss = torch.mean( torch.sqrt( (model_pred.float() - targets.float())**2 + huber_c**2 ) - huber_c )
                        else:
                            raise NotImplementedError
                    elif args.reweighting_scheme == 'reciprocal':
                        if args.loss_distance == "l2":
                            loss_weights = 1.0 / torch.clamp(1.0 - timepoints, min=0.1) / 2.3    # \int_0^{0.9} 1/(1-t)dt = 2.3
                            loss = (((model_pred.float() -  targets.float())**2).mean(dim=[1,2,3]) * loss_weights).mean()
                        else:
                            raise NotImplementedError
                    elif args.reweighting_scheme == 'rcp_win':
                        if args.loss_distance == "l2":
                            loss_weights = t_start / ( (1+args.windows) / (2*args.windows) )
                            loss = (((model_pred.float() -  targets.float())**2).mean(dim=[1,2,3]) * loss_weights).mean()
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        logger.info("***** Saving checkpoints *****")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if args.use_ema:
                            ema_unet.save_pretrained(os.path.join(save_path, "unet_ema"))
                        else:
                            if use_lora:
                                accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet_lora"))
                            else:
                                accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet"))
                                
                        if args.save_ckpt_state:
                            accelerator.save_state(save_path, safe_serialization=False)
                        
                if global_step % args.validation_steps == 0 or global_step in (1,):
                    if accelerator.state.process_index % 4 == 0:
                        logger.info("***** Running validation *****")
                        log_validation(
                            accelerator             = accelerator,
                            vae                     = vae,
                            unet                    = unet,
                            weight_dtype            = weight_dtype,
                            height                  = args.resolution,
                            width                   = args.resolution,
                            num_inference_steps     = args.inference_steps,
                            guidance_scale          = args.inference_cfg,
                            log_dir                 = os.path.join(args.output_dir, 'samples'),
                            global_step             = global_step,
                            rank                    = accelerator.state.process_index,
                            use_lora                = use_lora,
                        )
                        

            logs = {"global_step": global_step, "step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

    accelerator.end_training()


if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument("--debug", action="store_true",)
        ## dataset
        parser.add_argument("--data_root", type=str, default=None,)
        parser.add_argument("--resolution", type=int, default=512,)
        parser.add_argument("--dataloader_num_workers", type=int, default=0,)
        parser.add_argument("--train_batch_size", type=int, default=16, )
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1,)
        ## model
        parser.add_argument("--pretrained_vae_name_or_path", type=str, default=None, required=False,)
        parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,)
        parser.add_argument("--unet_model_path", type=str, default=None, required=True,)
        parser.add_argument("--revision", type=str, default=None, required=False,)
        parser.add_argument("--lora_rank", type=int, default=-1,)
        ## loss
        parser.add_argument("--pred_type", type=str, default='velocity',)
        parser.add_argument("--loss_type", type=str, default="velocity_matching",)
        parser.add_argument("--loss_distance", type=str, default='l2', choices=['l2', 'huber'],)
        parser.add_argument("--reweighting_scheme", type=str, default=None,)
        parser.add_argument("--huber_c", type=float, default=0.001, help="The huber loss parameter. Only used if `--loss_type=huber`.",)
        parser.add_argument("--windows", type=int, default=16,)
        parser.add_argument("--solving_steps", type=int, default=2,)
        parser.add_argument("--support_cfg", action="store_true", default=False,)
        parser.add_argument("--cfg_sync", action="store_true", default=False,)
        parser.add_argument("--discrete_timesteps", type=int, default=-1,)
        parser.add_argument("--train_mode", type=str, default="perflow",)
        parser.add_argument("--zero_snr", action="store_true")
        ## lr
        parser.add_argument("--learning_rate", type=float, default=5e-5,)
        parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
        parser.add_argument("--lr_scheduler", type=str, default="constant",)
        parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
        ## optimization
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        ## misc config
        parser.add_argument("--max_train_steps", type=int, default=1_000_000,)
        parser.add_argument("--gradient_checkpointing", action="store_true",)
        parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
        parser.add_argument("--use_8bit_adam", action='store_true')
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        ## checkpointing
        parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
        parser.add_argument("--output_dir", type=str, default="sd-model-finetuned",)
        parser.add_argument("--report_to", type=str, default="tensorboard",)
        parser.add_argument("--validation_steps", type=int, default=250,)
        parser.add_argument("--inference_steps", type=str, default="8", help="validation inference steps")
        parser.add_argument("--inference_cfg", type=str, default="7.5",)
        parser.add_argument("--save_ckpt_state", action="store_true",)
        parser.add_argument("--checkpointing_steps", type=int, default=2500,)
        parser.add_argument("--checkpoints_total_limit", type=int, default=None)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None,)
        
        args = parser.parse_args()
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != args.local_rank:
            args.local_rank = env_local_rank
        
        return args
    
    args = parse_args()
    
    main(args)
