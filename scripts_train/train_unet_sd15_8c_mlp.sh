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
    --config_file ./deepspeed_configs/default_config4_16.yaml \
    --main_process_port=20650 \
./scripts/train_sd15_mlp.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
    --dataset_file=data.json \
    --img_dir=image \
    --cond_img_path=conditions  \
    --cache_dir=/no_use --train_data_dir=/no_use \
    --resolution=512 \
    --learning_rate=1e-6 \
    --tracker_project_name=train_sd15_1e-6 \
    --tracker_run_name=train_sd15_1e-6 \
    --output_dir=/output \
    --allow_tf32 --set_grads_to_none --use_8bit_adam --mixed_precision=bf16 --logging_dir=log --image_column=source --caption_column=prompt \
    --conditioning_image_column=control_seg --dataloader_num_workers=8 --lr_warmup_steps=100 \
    --enable_xformers_memory_efficient_attention --validation_prompt=dog --validation_image=dog  --cfg_drop_ratio=0.05 \
    --checkpointing_steps=200 --validation_steps=100 --train_batch_size=12 \
    --report_to=tensorboard --num_train_epochs=200 \
    