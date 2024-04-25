accelerate launch \
    --main_process_port ??? \
    --num_processes ??? \
    --num_cpu_threads_per_process 6 \
    # perflow_accelerate_ddimeps_sdxl.py \
    #     --data_root "???"  \
    #     --resolution 1024 --dataloader_num_workers 6 --train_batch_size 8 --gradient_accumulation_steps 2 \
    #     --pretrained_model_name_or_path "../assets/public_models/StableDiffusion/stable-diffusion-xl-base-1.0" \
    #     --unet_model_path "" \
    #     --pretrained_vae_name_or_path "../assets/public_models/StableDiffusion/sdxl-vae-fp16-fix" \
    #     --pred_type "ddim_eps" --loss_type "noise_matching" \
    #     --windows 4 --solving_steps 10 --support_cfg --discrete_timesteps 200 \
    #     --learning_rate 1e-4 --lr_scheduler "constant" --lr_warmup_steps 100 \
    #     --mixed_precision "bf16" --gradient_checkpointing \
    #     --use_ema --output_dir "../_exps_/sdxl1024_perflow_4ddim10_ddimeps_xcfg" \
    #     --validation_steps 100 --inference_steps "8-4" --inference_cfg "2.0-2.0" --save_ckpt_state --checkpointing_steps 1000

    # perflow_accelerate_sd.py \
    #     --data_root "???" \
    #     --resolution 512 --dataloader_num_workers 8 --train_batch_size 32 --gradient_accumulation_steps 1 \
    #     --pretrained_model_name_or_path "../assets/public_models/DreamBooth/sd15_eps/DreamShaper_8_pruned" \
    #     --unet_model_path "" \
    #     --pred_type "diff_eps" --loss_type "noise_matching" \
    #     --windows 4 --solving_steps 8 --support_cfg --cfg_sync \
    #     --learning_rate 8e-5 --lr_scheduler "constant" --lr_warmup_steps 500 --use_ema \
    #     --mixed_precision "fp16" \
    #     --output_dir "../_exps_/sd15ds_perflow_4ddim8_diffeps_cfgsync" \
    #     --validation_steps 100 --inference_steps "8-4" --inference_cfg "7.5-4.5" --save_ckpt_state --checkpointing_steps 1000 \