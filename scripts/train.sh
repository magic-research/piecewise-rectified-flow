CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --main_process_port 10764 \
    --num_processes 1 \
    --num_cpu_threads_per_process 8 \
    --mixed_precision "fp16" \
    scripts/train_perflow.py \
        --data_root "???" --resolution 512 --dataloader_num_workers 8 \
        --train_batch_size 16 --gradient_accumulation_steps 1 \
        --pretrained_model_name_or_path "stable-diffusion-v1-5" \
        --pred_type "epsilon" --loss_type "noise_matching" \
        --windows 4 --solving_steps 8 \
        --support_cfg --cfg_sync \
        --learning_rate 8e-5 --lr_scheduler "constant" --lr_warmup_steps 1000 \
        --adam_weight_decay 1e-5 --max_grad_norm 1 \
        --max_train_steps 1000000 \
        --use_ema \
        --mixed_precision "fp16" \
        --output_dir "_exps_/tmp" \
        --debug