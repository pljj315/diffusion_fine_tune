
"""
webui不能单独加载unet(能够单独加载basemodel、controlnet、VAE、LoRA等,
而unet和text_encoder等会一同以HF_Diffusers_saved_pipeline的多文件夹形式组成basemodel)

因此训练好的unet要load到webui使用时,需要转换成HF_Diffusers_saved_pipeline的basemodel——> 这被分成两步骤：
    1. 先转为pipe整个大的safetensors: python3 1_convert_unet_sft_to_pipe_sft.py
    2. 转为webui的basemodel需要的文件组织形式: python3 2_convert_diffusers_to_original_sdxl.py --model_path aaa.safetensors --half --use_safetensors 
       来源:diffusers/scripts/convert_diffusers_to_original_sdxl.py
    3. 软链接到WEBUI存放模型的文件夹下;

"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['HF_HOME'] = 'huggingface'


import torch
import torch.utils.checkpoint
import torch.utils.cpp_extension
from diffusers import (
    StableDiffusionXLInpaintPipeline, 
    UNet2DConditionModel,
    AutoencoderKL)


if __name__ == "__main__":

    device = torch.device('cuda')
    weight_dtype = torch.float32

    pretrained_model_name_or_path = "ckpt/SDXL_inpainting" 
    pretrained_vae_model_name_or_path = 'madebyollin/sdxl-vae-fp16-fix'
    unet_step = 3500
    pretrained_unet_name_or_path = f"unet-{unet_step}"

    unet = UNet2DConditionModel.from_pretrained(pretrained_unet_name_or_path)
    unet.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(pretrained_vae_model_name_or_path,  force_upcast=False,)
    vae.requires_grad_(False)

    pipeline_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        unet = unet,
        vae= vae,
        torch_dtype=weight_dtype,
        local_files_only = True,).to(device)

    save_pretrained_unet_name_or_path = f"unetPipe_{unet_step}"
    pipeline_inpaint.save_pretrained(save_directory =save_pretrained_unet_name_or_path)

        