export CUDA_VISIBLE_DEVICES="0,1,2,3"
export ACCELERATE_LOG_LEVEL="WARNING"
export HF_HOME="huggingface"
export OPENCV_IO_ENABLE_OPENEXR="1"

# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# export CPLUS_INCLUDE_PATH="/usr/local/cuda/include/:${CPLUS_INCLUDE_PATH:-}"
# export C_INCLUDE_PATH="/usr/local/cuda/include/:${C_INCLUDE_PATH:-}"
# export PATH="/usr/local/cuda/bin:${PATH}"
# new_path="/bin"
# export PATH="${new_path}:${PATH}"
# export NCCL_P2P_LEVEL='NVL' 

export NCCL_P2P_LEVEL=NVL
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=info
# export NCCL_SOCKET_IFNAME=eth0

accelerate launch  \
    --config_file ./deepspeed_configs/default_config2_16.yaml \
    --main_process_port=20652 \
./scripts/train_sdxl_unet_9c_inpaint.py \
    --pretrained_model_name_or_path=OzzyGT/RealVisXL_V4.0_inpainting \
    --pretrained_unet_name_or_path= \
    --img_dir= \
    --img_dir_2= \
    --cond_img_path= \
    --output_dir= \
    --resolution=768 \
    --train_batch_size=2 \
    --allow_tf32 \
    --enable_xformers_memory_efficient_attention \
    --set_grads_to_none \
    --use_8bit_adam \
    --mixed_precision=fp16 \
    --logging_dir=log \
    --image_column=source \
    --caption_column=prompt \
    --tracker_project_name=tune_unet_30000 \
    --tracker_run_name=tune_unet_30000 \
    --conditioning_image_column=control_seg \
    --learning_rate=1e-7 \
    --dataloader_num_workers=2 \
    --lr_warmup_steps=100 \
    --validation_steps=50 \
    --validation_prompt=dog \
    --validation_image=dog \
    --cfg_drop_ratio=0.1 \
    --checkpointing_steps=200 \
    --max_train_steps=72500
