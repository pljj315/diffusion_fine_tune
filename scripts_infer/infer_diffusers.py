import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
os.environ['HF_HOME'] = '/huggingface'

import torch
import torch.utils.checkpoint
import torch.utils.cpp_extension

from PIL import Image, ImageOps

from diffusers import (
    StableDiffusionXLInpaintPipeline, 
    UniPCMultistepScheduler,
)
from diffusers.utils import  make_image_grid

def infer_pipe(strength, pipeline_inpaint,  num_validation_images, seed, device , output_dir, resolution):


    bg_product_prompts = []


    product_images =[]

    product_masks =[]


    negative_prompt = ""

    for  _idx, validation_image_path in enumerate(product_images):
        ori_img = Image.open(validation_image_path).convert("RGB")
        ori_mask_img =  Image.open(product_masks[_idx]).convert("L")

        with torch.autocast("cuda"):
            for i in range(len(bg_product_prompts)):
                generator = torch.Generator(device=device).manual_seed(seed)

                # new_mask: 已经反转，黑色保留
                new_mask = ImageOps.invert(ori_mask_img)  # product为黑色——对应runnaway黑色保留

                gen_images =[]
                for _ in range(num_validation_images):
                    image = pipeline_inpaint(
                        prompt = bg_product_prompts[i],
                        negative_prompt = negative_prompt,
                        
                        image= ori_img,   # RGB三通道PIL
                        mask_image = new_mask,

                        height = resolution,
                        width = resolution,
                        num_inference_steps=30, 

                        strength = strength,     # 重绘幅度

                        padding_mask_crop = None,
                        generator=generator, 
                        ).images[0]
                    gen_images.append(image)

                os.makedirs(output_dir, exist_ok=True)

                gen_list = [ori_img, new_mask] + gen_images
                grid = make_image_grid(gen_list, rows=1, cols=len(gen_list))

                grid.save('%s/%d_strength_%s_%d.jpg'%(output_dir,  i,  strength, _idx))
    print(f"----------finish inference for strength={strength}!--------")

if __name__ == "__main__":

    device = torch.device('cuda')
    weight_dtype = torch.float16

    pretrained_model_name_or_path = "OzzyGT/RealVisXL_V4.0_inpainting" #"SDXL_inpainting" 

    resolution=1024
    num_validation_images = 4
    seed = 0


    pipeline_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        local_files_only = True,
        variant = "fp16")
    pipeline_inpaint.scheduler = UniPCMultistepScheduler.from_config(pipeline_inpaint.scheduler.config)
    pipeline_inpaint = pipeline_inpaint.to(device)


    strength_list = [1, 0.99, 0.7]
    
    output_dir=f'./output'
    for strength in strength_list:
        infer_pipe(strength, pipeline_inpaint, num_validation_images, seed, device , output_dir, resolution )

    # del pipeline
    # del pipeline_sdxl
    # gc.collect()
    # torch.cuda.empty_cache()
    