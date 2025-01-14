export CUDA_VISIBLE_DEVICES="6"
export ACCELERATE_LOG_LEVEL="WARNING"
export HF_HOME=".../huggingface"

export NCCL_P2P_LEVEL='NVL'  # export NCCL_P2P_LEVEL=NVL


accelerate launch  \
    --config_file ./deepspeed_configs/default_config1_8.yaml \
    --main_process_port=20652 \
./scripts/train_controlnet_sd15.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
    --dataset_file=data.json \
    --cache_dir=/no_use --train_data_dir=/no_use \
    --resolution=512 \
    --learning_rate=1e-4 \
    --tracker_project_name=train_sd15_controlnet_1e-4_4c \
    --output_dir=output_controlnet/train_sd15_controlnet_1e-4_4c \
    --allow_tf32 --set_grads_to_none --use_8bit_adam --mixed_precision=bf16 --logging_dir=log --image_column=source \
    --caption_column=prompt --conditioning_image_column=control_seg --dataloader_num_workers=8 --lr_warmup_steps=100 \
    --enable_xformers_memory_efficient_attention --validation_prompt=dog --validation_image=dog --num_validation_images=1 \
    --checkpointing_steps=2 --validation_steps=2  --train_batch_size=2 \
    --report_to=tensorboard  --num_train_epochs=200 \
    --load_weights_from_unet --add_shadow --channels_of_controlnet=4 \
    # --with_bg_cond