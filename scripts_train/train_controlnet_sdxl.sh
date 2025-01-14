export CUDA_VISIBLE_DEVICES="6"
export ACCELERATE_LOG_LEVEL="WARNING"
export HF_HOME=".../huggingface"
export NCCL_P2P_LEVEL='NVL' 

accelerate launch  \
    --config_file deepspeed_configs/default_config1_8.yaml \
    --main_process_port=20652 \
./scripts/train_controlnet_sd15.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix \
    --img_dir= \
    --img_dir_2= \
    --cond_img_path= \
    --controlnet_model_name_or_path= \
    --output_dir= \
    --resolution=768 \
    --train_batch_size=2 \
    --allow_tf32 \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --use_8bit_adam \
    --mixed_precision=bf16 \
    --logging_dir=log \
    --image_column=source \
    --caption_column=prompt \
    --tracker_project_name=tune_controlnet_5200 \
    --tracker_run_name=tune_controlnet_5200 \
    --conditioning_image_column=control_seg \
    --learning_rate=1e-7 \
    --dataloader_num_workers=2 \
    --lr_warmup_steps=100 \
    --validation_steps=50 \
    --validation_prompt=dog \
    --validation_image=dog  \
    --cfg_drop_ratio=0.1 \
    --checkpointing_steps=200 \
    --max_train_steps=72500