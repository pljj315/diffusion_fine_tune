export CUDA_VISIBLE_DEVICES="3"

export ACCELERATE_LOG_LEVEL="WARNING"
export HF_HOME="...huggingface"
export NCCL_P2P_LEVEL= NVL
export OPENCV_IO_ENABLE_OPENEXR="1"

accelerate launch  \
    --config_file ./deepspeed_configs/default_config1_8.yaml \
    --main_process_port=20650 \
./scripts/train_ipa_sd15.py \
    --pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5  \
    --image_encoder_path=models/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder \
    --data_json_file=data.json \
    --data_root_path=/no_use \
    --resolution=512 \
    --learning_rate=1e-4 \
    --output_dir=output/train_sd15_ipabg_new_1e-4_3 \
    --mixed_precision=fp16 \
    --dataloader_num_workers=8 \
    --save_steps=100 --train_batch_size=32 \
    --report_to=tensorboard --logging_dir=log --num_train_epochs=200 \
    --envmap_or_bg=bg \
    --pretrained_unet_checkpoint=diffusion_pytorch_model.safetensors \
    # --pretrained_ip_adapter_path=ip-adapter_sd15.bin \
    
    # --pretrained_mlp_checkpoint=mlp.pth
    