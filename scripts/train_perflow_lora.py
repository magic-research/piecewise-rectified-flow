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

from huggingface_hub import HfFolder, Repository, create_repo, whoami
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
import datasets
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
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

def main(args):
    ## 0. misc
    logger = get_logger(__name__, log_level="INFO") # Make one log on every process with the configuration for debugging.
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.report_to),)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision=args.mixed_precision, log_with=args.report_to, project_config=accelerator_project_config,)

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32
        
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
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", )
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    ## 1.1 Prepare teacher unet
    teacher_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
    )
    teacher_unet.to(accelerator.device, dtype=weight_dtype)
    teacher_unet.requires_grad_(False)
    
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    teacher_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    teacher_num_train_timesteps = teacher_scheduler.config.num_train_timesteps

    from src.pfode_solver import PFODESolver
    solver = PFODESolver(scheduler=teacher_scheduler, t_initial=1, t_terminal=0,) #!!!:
    
    ## 1.2 Prepare student unet
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
        
    # Add LoRA to the student U-Net, only the LoRA projection matrix will be updated by the optimizer.    
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
    unet.train()
    
    if weight_dtype == torch.bfloat16:
        unet.to(accelerator.device, dtype=weight_dtype)
    else:
        unet.to(accelerator.device)
        
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

        
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
    logger.info("***** preparing datasets *****")
    if ("/mnt/bn/ic-research-agic-editing2/data/laion5b_aesv2_512plus_10M" in args.data_root):
        from src.laion_bytenas import make_train_dataset
    elif "/mnt/bn/" in args.data_root:
        from src.laion_local import make_train_dataset
    elif "hdfs://" in args.data_root:
        from src.laion_hdfs import make_train_dataset
    else:
        raise NotImplementedError
    
    if args.debug:
        world_size = 64
        shuffle = False
    else:
        world_size = accelerator.state.num_processes
        shuffle = True
    
    train_dataset = make_train_dataset(
        train_data_path=args.data_root,
        size = args.resolution, 
        tokenizer=tokenizer, 
        cfg_drop_ratio = 0,
        rank=accelerator.state.process_index, 
        world_size=world_size,
        shuffle=shuffle,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collect_fn if hasattr(train_dataset, 'collect_fn') else None,
    )
    
    if args.support_cfg:
        cfg_drop_ratio = 0.1
    else:
        cfg_drop_ratio = 0.


    ## 3. Optimization
    trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    
    optimizer = torch.optim.AdamW(
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
        accelerator, unet, weight_dtype,
        height=512, width=512, num_inference_steps=5, guidance_scale=3.0,
        log_dir="", global_step=0, rank=0, sanity_check=False,
        ):
        with torch.no_grad():
            prompts = [
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a man",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a woman",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a dog",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a cat",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; green grassland and blue sky",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; mountains and trees",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a desk and a chair",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
                ["RAW photo, 8k uhd, dslr, high quality, film grain; a car on the road",   "distorted, blur, smooth, low-quality, warm, haze, over-saturated, high-contrast"],
            ]
            val_pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path, 
                vae             = vae,
                text_encoder    = text_encoder,
                tokenizer       = tokenizer,
                scheduler       = perflow_scheduler,
                torch_dtype     = weight_dtype,
                safety_checker  = None,
                ).to(device=accelerator.device)
        
            lora_state_dict = convert_peft_keys_to_kohya(accelerator.unwrap_model(unet), "lora_unet", weight_dtype,)
            val_pipeline.load_lora_weights(lora_state_dict)
            val_pipeline.fuse_lora()
            val_pipeline.to(device=accelerator.device, dtype=weight_dtype)
            
            generator = torch.Generator(device=accelerator.device)
            generator.manual_seed(123456789 + rank)

            # 8 step
            samples = []
            for _, prompt in enumerate(prompts):
                sample = val_pipeline(
                    prompt              = prompt[0],
                    negative_prompt     = prompt[1],
                    height              = height,
                    width               = width,
                    num_inference_steps = num_inference_steps,
                    guidance_scale      = guidance_scale,
                    generator           = generator,
                    output_type         = 'pt',
                ).images
                samples.append(sample)
            samples = torchvision.utils.make_grid(torch.concat(samples), nrow=4)       
        
            if not sanity_check:
                Path(f"{log_dir}/step_{global_step}").mkdir(parents=True, exist_ok=True)
                save_path = f"{log_dir}/step_{global_step}/sample-{num_inference_steps}_r{rank}.png"
            else:
                Path(f"{log_dir}/sanity_check").mkdir(parents=True, exist_ok=True)
                save_path = f"{log_dir}/sanity_check/sample-{num_inference_steps}_r{rank}.png"
            torchvision.utils.save_image(samples, save_path)
            logging.info(f"Saved samples to {save_path}")
            
            # 4 step
            samples = []
            for _, prompt in enumerate(prompts):
                sample = val_pipeline(
                    prompt              = prompt[0],
                    negative_prompt     = prompt[1],
                    height              = height,
                    width               = width,
                    num_inference_steps = 4,
                    guidance_scale      = 4.5,
                    generator           = generator,
                    output_type         = 'pt',
                ).images
                samples.append(sample)
            samples = torchvision.utils.make_grid(torch.concat(samples), nrow=4)       
        
            if not sanity_check:
                Path(f"{log_dir}/step_{global_step}").mkdir(parents=True, exist_ok=True)
                save_path = f"{log_dir}/step_{global_step}/sample-{4}_r{rank}.png"
            else:
                Path(f"{log_dir}/sanity_check").mkdir(parents=True, exist_ok=True)
                save_path = f"{log_dir}/sanity_check/sample-{4}_r{rank}.png"
            torchvision.utils.save_image(samples, save_path)
            logging.info(f"Saved samples to {save_path}")
            
        # empty cache
        del val_pipeline
        torch.cuda.empty_cache()
    
    
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
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        resume_global_step = int(path.split("-")[-1])

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
    
    
    ## 6. >>>>>>>> >>>>>>>> Start training >>>>>>>> >>>>>>>>
    unet.train()
    train_loss = 0.0
    sanity_check_flag = True
    
    while global_step < args.max_train_steps:
        for batch in train_dataloader:
            ## 6.0 sanity check
            if accelerator.sync_gradients:
                if sanity_check_flag:
                    if accelerator.state.process_index % 4 == 0:
                        logger.info("***** sanity checking *****")
                        log_validation(
                            accelerator             = accelerator,
                            unet                    = unet,
                            weight_dtype            = weight_dtype,
                            height                  = args.resolution,
                            width                   = args.resolution,
                            num_inference_steps     = args.val_steps,
                            guidance_scale          = args.val_cfg,
                            log_dir                 = os.path.join(args.output_dir, 'samples'),
                            global_step             = global_step,
                            rank                    = accelerator.state.process_index,
                            sanity_check            = sanity_check_flag,
                        )
                    sanity_check_flag = False
                    
            with accelerator.accumulate(unet):
                bsz = batch["pixel_values"].shape[0]
                ## 6.1 prepare latents and prompt embeddings
                with torch.no_grad():
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype) # (b c h w)
                    latents = vae.encode(pixel_values).latent_dist
                    latents = latents.sample()
                    latents = latents * vae.config.scaling_factor

                    prompt_ids = batch['input_ids'].to(accelerator.device)
                    text_embeddings = text_encoder(prompt_ids)[0]
                    null_text_ids = torch.stack(
                        [tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt",).input_ids[0]] * prompt_ids.shape[0]
                    ).to(accelerator.device)
                    null_embeddings = text_encoder(null_text_ids)[0]
                    
                    if args.support_cfg:
                        assert null_embeddings.shape == text_embeddings.shape
                        mask_text = torch.zeros((text_embeddings.shape[0],1,1), dtype=text_embeddings.dtype, device=text_embeddings.device)
                        for i in range(text_embeddings.shape[0]):
                            mask_text[i] = 1 if random.random() > cfg_drop_ratio else 0
                        text_embeddings_dropout =  mask_text * text_embeddings + (1-mask_text) * null_embeddings
                    else:
                        text_embeddings_dropout = text_embeddings

                ## 6.2 Sample timesteps. NOTE: t \in [1, 0]
                ## Prepare the endpoints of windows, model inputs
                with torch.no_grad():
                    timepoints = torch.rand((bsz,), device=latents.device) # [0,1)
                    if args.discrete_timesteps == -1:
                        timepoints = (timepoints * teacher_num_train_timesteps).floor() / teacher_num_train_timesteps # assert [0, 999/1000]
                    else:
                        assert isinstance(args.discrete_timesteps, int)
                        timepoints = (timepoints * args.discrete_timesteps).floor() / args.discrete_timesteps # in [0, 39/40)
                    timepoints = 1 - timepoints # [1, 1/1000], [1, 1/40]

                    noises = torch.randn_like(latents)
                    
                    t_start, t_end = perflow_scheduler.time_windows.lookup_window(timepoints)
                    latents_start = teacher_scheduler.add_noise(latents, noises, torch.clamp((t_start*teacher_num_train_timesteps).long()-1, min=0))
                    if args.cfg_sync:
                        latents_end = solver.solve(
                            latents                 = latents_start, 
                            t_start                 = t_start,
                            t_end                   = t_end,
                            unet                    = teacher_unet, 
                            prompt_embeds           = text_embeddings_dropout, 
                            negative_prompt_embeds  = null_embeddings, 
                            guidance_scale          = 1.0, 
                            num_steps               = args.solving_steps,
                            num_windows             = args.windows,
                        )
                    else:
                        latents_end = solver.solve(
                            latents                 = latents_start, 
                            t_start                 = t_start,
                            t_end                   = t_end,
                            unet                    = teacher_unet, 
                            prompt_embeds           = text_embeddings, 
                            negative_prompt_embeds  = null_embeddings, 
                            guidance_scale          = 7.5, 
                            num_steps               = args.solving_steps,
                            num_windows             = args.windows,
                        )
                    
                    latents_t = latents_start + (latents_end - latents_start) / (t_end[:,None,None,None] - t_start[:,None,None,None]) * (timepoints[:, None, None, None] - t_start[:, None, None, None])
                    latents_t = latents_t.to(weight_dtype)

                ## 6.4 prepare targets -> perform inference -> convert -> compute loss
                with torch.no_grad():
                    if args.loss_type == "velocity_matching":
                        targets = ( latents_end - latents_start ) / (t_end[:,None,None,None] - t_start[:,None,None,None])
                    elif args.loss_type == "noise_matching":
                        _, _, _, _, gamma_s_e = perflow_scheduler.get_window_alpha(timepoints.float().cpu() * teacher_num_train_timesteps)
                        gamma_s_e = gamma_s_e[:,None,None,None].to(device=latents.device)
                        targets = (latents_start - gamma_s_e * latents_end ) / ( ( 1- gamma_s_e**2)**0.5 ) #FIXME:check numerical stability
                    else:
                        raise NotImplementedError
                    
                model_pred = unet(latents_t, timepoints.float() * teacher_num_train_timesteps, text_embeddings_dropout).sample
                
                if args.pred_type == "velocity" and args.loss_type == "noise_matching":
                        d_t_to_s = (timepoints - t_start)[:, None,None,None]
                        d_e_to_t = (t_end - timepoints)[:, None,None,None]
                        model_pred = ( (1-gamma_s_e)*latents_t - (model_pred * d_t_to_s + gamma_s_e*model_pred*d_e_to_t) ) / ( ( 1- gamma_s_e**2)**0.5 )
                elif args.pred_type == "epsilon" and args.loss_type == "velocity_matching":
                    raise NotImplementedError


                loss = F.mse_loss(model_pred.float(), targets.float(), reduction="mean")


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
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        logger.info("***** Saving checkpoints *****")
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path, safe_serialization=False)
                        accelerator.unwrap_model(unet).save_pretrained(os.path.join(save_path, "unet_lora"))
                        logger.info(f"Saved state to {save_path}")
                        
                if global_step % args.validation_steps == 0:
                    if accelerator.state.process_index % 2 == 0:
                        logger.info("***** Running validation *****")
                        log_validation(
                            accelerator             = accelerator,
                            unet                    = unet,
                            weight_dtype            = weight_dtype,
                            height                  = args.resolution,
                            width                   = args.resolution,
                            num_inference_steps     = args.val_steps,
                            guidance_scale          = args.val_cfg,
                            log_dir                 = os.path.join(args.output_dir, 'samples'),
                            global_step             = global_step,
                            rank                    = accelerator.state.process_index,
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
        parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True,)
        parser.add_argument("--unet_model_path", type=str, default=None, required=True,)
        parser.add_argument("--revision", type=str, default=None, required=False,)
        parser.add_argument("--lora_rank", type=int, default=64,)
        ## loss
        parser.add_argument("--loss_type", type=str, default="velocity_matching", choices=["velocity_matching", "noise_matching"],)
        parser.add_argument("--pred_type", type=str, default='velocity', choices=['velocity', 'epsilon'],)
        parser.add_argument("--windows", type=int, default=16,)
        parser.add_argument("--solving_steps", type=int, default=2,)
        parser.add_argument("--support_cfg", action="store_true", default=False,)
        parser.add_argument("--cfg_sync", action="store_true", default=False,)
        parser.add_argument("--discrete_timesteps", type=int, default=-1,)
        ## lr
        parser.add_argument("--learning_rate", type=float, default=1e-4,)
        parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
        parser.add_argument("--lr_scheduler", type=str, default="constant",)
        parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of steps for the warmup in the lr scheduler.")
        ## optimization
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
        parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
        parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        ## misc config
        parser.add_argument("--max_train_steps", type=int, default=1_000_000,)
        parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
        parser.add_argument("--gradient_checkpointing", action="store_true",)
        parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        ## checkpointing
        parser.add_argument("--output_dir", type=str, default="sd-model-finetuned",)
        parser.add_argument("--report_to", type=str, default="tensorboard",)
        parser.add_argument("--validation_steps", type=int, default=250,)
        parser.add_argument("--val_steps", type=int, default=4, help="validation inference steps")
        parser.add_argument("--val_cfg", type=float, default=4.5,)
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
