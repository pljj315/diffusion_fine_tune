import os
import argparse
import gc
import logging
import random
from pathlib import Path
import json
import math
import glob
import safetensors.torch as sf
from PIL import Image
from tqdm.auto import tqdm
import cv2

import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from packaging import version
from torch.utils.tensorboard import SummaryWriter

import itertools
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo

import diffusers
from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, DPMSolverMultistepScheduler, DDPMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import make_image_grid
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import torch.nn as nn
if is_wandb_available():
    import wandb
from utils.utils_degradation import get_IBL

logger = get_logger(__name__)

# 定义 MLP 网络
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 定义各层的线性变换 16*64*3=3072
        self.fc1 = nn.Linear(3072, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 2304)  # 输出 2304=3*768 维向量

    def forward(self, x):
        # 通过每一层的前向传播，激活函数使用 Leaky ReLU
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01) # 负数输入对应的斜率
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)
        return x
    
    

class FiveLayerMLP(nn.Module):
    def __init__(self, latent_dim=64*64*4):
        super(FiveLayerMLP, self).__init__()
        # latent_dim在这里是64*64，因为MLP处理的是展平后的数据
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, latent_dim)

    def forward(self, x):
        # 展平输入
        bs, c, h, w = x.size()
        x = x.reshape(bs, -1)  # 展平为 [bs, 4*64*64]
        
        # 前向传播
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = self.fc5(x)  # 输出层无激活函数
        
        # 恢复到原始形状
        x = x.reshape(bs, 4, h, w)  # 恢复到 [bs, 4, 64, 64]
        return x

 

@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float16)

        results.append(y)
    return results

def log_validation(
        noise_scheduler, vae, text_encoder, tokenizer, models_to_opt, args, accelerator, weight_dtype, step
):
    logger.info("Running validation... ")
    # Pipeline
    unet = models_to_opt.unet
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        scheduler = noise_scheduler, 
        safety_checker=None,
        requires_safety_checker=False,
        feature_extractor=None,
        image_encoder=None,
        # local_files_only = True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention(attention_op=None)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    ###
    prompts = []
    fgs = glob.glob("/fg/*")
    fgs.sort()
    masks = glob.glob("/mask/*")
    masks.sort()
    bgs = glob.glob("/inpaintbg/*")
    bgs.sort()
    ###

    
    ### mlp
    mlp = accelerator.unwrap_model(models_to_opt.mlp)
    mlp = mlp.to(accelerator.device)
    env = Image.open("env_panorama.png").convert("RGB")
    env = np.array(env) # env_hdr = cv2.GaussianBlur(env, ksize=(33,33), sigmaX= 11)
    env_hdr = cv2.resize(env,(64,32)) / 255.0
    env_hdr = torch.from_numpy(np.stack([env_hdr], axis=0)).float() # 0~1+  #[1,3,32,64]
    env_hdr = env_hdr.movedim(-1, 1)
    env_hdr = env_hdr[:,:,:16,:]          # 只取正z半轴  1*3*16*64
    bsz = env_hdr.shape[0]
    env_hdr_flattened = env_hdr.reshape(bsz, -1).to(device=vae.device, dtype=weight_dtype) 
    env_hdr_embedding = mlp(env_hdr_flattened).reshape(bsz, 3, 768)
    ###

    image_logs = []
    for  idx, (validation_image_path, validation_mask_path, txt_path, bg_path) in enumerate(zip(fgs, masks, prompts, bgs)):
        validation_prompt = prompts[idx]
        negative_prompt  = 'lowres, bad anatomy, bad hands, cropped, worst quality'

        input_fg = Image.open(validation_image_path).convert('RGB')
        input_mask = Image.open(validation_mask_path).convert('L')
        width, height = input_fg.size
        scale = min(height/args.resolution, width/args.resolution)
        width = int(round((width /scale)  / 64.0) * 64)
        height = int(round((height /scale)  / 64.0) * 64)
        input_fg = input_fg.resize((width, height), Image.BILINEAR)
        input_mask = input_mask.resize((width, height), Image.BILINEAR)
        input_mask =np.array(input_mask)
        fg = np.array(input_fg)
        fg[input_mask<127] = 127

        concat_conds =  (torch.from_numpy(np.stack([fg], axis=0)).float() / 255.0) * 2.0 - 1.0 # 0~255 to -1~+1   #[1,h,w,c]
        concat_conds = concat_conds.movedim(-1, 1).to(device=vae.device, dtype=vae.dtype)      # [1,c,h,w]
        concat_conds = vae.encode(concat_conds).latent_dist.sample() * vae.config.scaling_factor
    
        unconds = encode_prompt(negative_prompt, text_encoder, tokenizer, vae.device)
        text_embeddings = encode_prompt(validation_prompt, text_encoder, tokenizer, vae.device) # [bs, 77, 768]
        encoder_hidden_states = torch.cat([env_hdr_embedding, text_embeddings], dim = 1) [:, :77,:]
        
        images = []
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                latents = pipeline(
                    prompt_embeds=encoder_hidden_states,
                    negative_prompt_embeds=unconds,
                    
                    num_inference_steps=30, 
                    generator=generator,
                    output_type='latent',
                    guidance_scale=7.0,
                    height = height,
                    width = width,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                ).images.to(vae.dtype) / vae.config.scaling_factor
                ic_relight_imgs = vae.decode(latents).sample
                ic_relight_imgs = pytorch2numpy(ic_relight_imgs)
            image = ic_relight_imgs[0]
            images.append(image)

            if step:
                os.makedirs('%s/tmp/'%args.output_dir, exist_ok=True)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite('%s/tmp/%d_%d_img.png'%(args.output_dir, step, idx), image)     
        image_logs.append(
            {"validation_fg": input_fg, "validation_prompt": validation_prompt, "images":images})
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_fg = log["validation_fg"]

                formatted_images = []
                formatted_images.append(np.asarray(validation_fg))
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_fg = log["validation_fg"]
                formatted_images.append(wandb.Image(validation_fg, caption="fg"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR
    snr = (alpha / sigma) ** 2
    return snr

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default='fp16',
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="The input image directory.",
    )
    parser.add_argument(
        "--cond_img_path",
        type=str,
        default=None,
        help="The input condition image directory.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default=None,
        help="The input dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='control_canny',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="noise_offset.",
    )
    parser.add_argument(
        "--cfg_drop_ratio",
        type=float,
        default=0.0,
        help="cfg_drop_ratio.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=0.0,
        help="snr_gamma.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default="sd_xl_train_run",
        help=(
            "The `run_name` argument passed to Accelerator.init_trackers for"
        ),
    )
    parser.add_argument(
        "--add_shadow",
        action="store_true",
        help="Whether or not to add_shadow.",
    )
    parser.add_argument(
        "--consistensicy_loss",
        action="store_true",
        help="Whether or not to use consistensicy_loss.",
    )
    parser.add_argument(
        "--pretrained_unet_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pretrained_mlp_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--envmap_file",
        type=str,
        default=None,
        help="The input envmap_file file.",
    )
    parser.add_argument(
        "--envmap_file_new",
        type=str,
        default=None,
        help="The input envmap_file_new file.",
    )
    # parser.add_argument(
    #     "--env_position",
    #     type=str,
    #     default="front",
    #     help=" front  or   behind    or   concatenate",
    # )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

class ComposeDataset(Dataset):
    def __init__(self, items_data, envmap_items, add_shadow=False, consistensicy_loss=False, size=512):
        self.size = size
        self.add_shadow = add_shadow
        self.consistensicy_loss = consistensicy_loss

        self.items_data = items_data
        self.envmap_items = envmap_items
        logger.info('--------dataset total num:-------',len(self.items_data), 'add_shadow==',add_shadow)
        logger.info('--------envmap_items num :-------',len(self.envmap_items))

        self.img_source_name, self.img_cond_name = 'source', 'control_seg'
        self.img_dir = {}
        self.cond_img_path = {}
        self.cache_dir_emb = {}
        

        
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]), # -1, +1
            ]
        )
        
    def __len__(self):
        return len(self.items_data)
    
    def get_random(self ):
        idx = random.randint(0, len(self.items_data))
        return self.__getitem__(idx)
    
    def __getitem__(self, index): 
        try:
            type_ = self.items_data[index]["type"]
            
            ## 模型的输入是RGB   模型的输出是RGB： 
            img_path = os.path.join(self.img_dir[type_], self.items_data[index][self.img_source_name])
            mask_path = os.path.join(self.cond_img_path[type_], self.items_data[index][self.img_cond_name])
            normal_path = os.path.join(self.normal_img_dir, type_, self.items_data[index][self.img_source_name])
            albedo_path = os.path.join(self.albedo_img_dir, type_, self.items_data[index][self.img_source_name])
            img_pil = Image.open(img_path).convert('RGB')
            mask_pil = Image.open(mask_path).convert('RGB')
            normal_pil = Image.open(normal_path).convert('RGB')
            albedo_pil = Image.open(albedo_path).convert('RGB')          

            
            # ibl_methods={'points_relighting':0.85,  'envmap_relighting':0.15},
            # env_methods = {'env_from_normal':0.6,   'diffusionlight':0.4   },
            # point_p_sets= {'1':0.1, '2':0.4, '3':0.4, '4':0.1},
            # 设置IBL参数：
            degradation_pil, IBL_label = get_IBL(img_pil, normal_pil, albedo_pil, self.envmap_items,
                    self.img_dir, self.normal_img_dir, self.albedo_img_dir, self.shadow_mask_dir, self.exr_outdir,
                    self.img_source_name, self.shadow_index_number, self.shadow_i_number,
                    new_resolution=self.size, albedo_shading_gray=False, 
                    add_shadow = self.add_shadow, shadow_p = 0.2, ambient_min=-5, ambient_max=30,
                    ibl_methods={'points_relighting':0.85,  'envmap_relighting':0.15},
                    env_methods = {'env_from_normal':0.6,   'diffusionlight':0.4   },
                    point_p_sets= {'1':0.1, '2':0.4, '3':0.4, '4':0.1},
                )
        
            #构建训练对:
            mask_pil = mask_pil.resize((degradation_pil.width, degradation_pil.height), Image.BILINEAR)
            img_pil = img_pil.resize((degradation_pil.width, degradation_pil.height), Image.BILINEAR)
            input_mask =np.array(mask_pil)
            degradation_pil = np.array(degradation_pil)
            degradation_pil[input_mask<127] = 127 
            degradation_pil = Image.fromarray(degradation_pil)
            input_gt = self.image_transforms(img_pil)          # -1~ +1
            input_fg = self.image_transforms(degradation_pil)  # -1~ +1 
            # print(input_gt.shape, input_fg.shape)     # [c,h,w]

            cache_path = os.path.join(self.cache_dir_emb[type_], self.items_data[index]['emb_cache'])
            if not os.path.isfile(cache_path) :
                if type_=="playground_v2.5":
                    cache_path = os.path.join(self.cache_dir_emb_randstr_playgroundg, self.items_data[index]['emb_cache'])
            emb_cache = torch.load(cache_path, weights_only=True)
            prompt_embeds = emb_cache['prompt_embeds'] # [77, 2048]
            prompt_embeds = prompt_embeds[:, :768]

            # consistensicy loss:
            env_path = os.path.join(self.exr_outdir, type_, os.path.splitext(self.items_data[index][self.img_source_name])[0]+'.exr')
            # env_path_new = os.path.join(self.exr_outdir_new, type_, os.path.splitext(self.items_data[index][self.img_source_name])[0]+'.exr')
            if self.consistensicy_loss:
                if os.path.isfile(env_path):
                    env_hdr = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
                    env_hdr = cv2.cvtColor(env_hdr, cv2.COLOR_BGR2RGB).astype(np.float32) # env_hdr = cv2.GaussianBlur(env_hdr, ksize=(33,33), sigmaX= 11)
                    env_hdr = cv2.resize(env_hdr,(64,32)).clip(0,10) # (32, 64, 3)

                    env_mask = np.random.uniform(low=0.0, high=1.0, size=(8,4))
                    env_mask[env_mask>0.5]= 1
                    env_mask[env_mask<0.5]= 0
                    env_mask = (cv2.resize(env_mask, (64,32))[:,:,np.newaxis]).astype(np.uint8)
                    env_hdr_1, env_hdr_2 = env_hdr * env_mask, env_hdr * (1-env_mask)

                    env_hdr = torch.from_numpy(np.stack([env_hdr], axis=0)).float()# 0~1+  #[1,3,h,w]
                    env_hdr = env_hdr.movedim(-1, 1)
                    env_hdr_1 = torch.from_numpy(np.stack([env_hdr_1], axis=0)).float()
                    env_hdr_1 = env_hdr_1.movedim(-1, 1)
                    env_hdr_2 = torch.from_numpy(np.stack([env_hdr_2], axis=0)).float()
                    env_hdr_2 = env_hdr_2.movedim(-1, 1)

                    input_mask = cv2.resize(input_mask,(self.size//8, self.size//8))
                    input_mask = cv2.cvtColor(input_mask, cv2.COLOR_RGB2GRAY)
                    input_mask[input_mask>127] =255
                    input_mask[input_mask<127] =0
                    input_mask = (input_mask / 255).astype(np.int32)
                    input_mask = torch.from_numpy(np.stack([input_mask[np.newaxis,:,:]], axis=0)).float() # binary： [1, 1, 64, 64]
                    # print("shape", input_gt.shape, input_fg.shape, env_hdr.shape, env_hdr_1.shape, env_hdr_2.shape, mask.shape)
                    return {            
                        "gt": input_gt,      # gt
                        "fg": input_fg,      # degradation_pil
                        "prompt_embeds": prompt_embeds,
                        "env_hdr":env_hdr,

                        "env_hdr_1":env_hdr_1,
                        "env_hdr_2":env_hdr_2,
                        "mask":input_mask,
                    }
                else:
                    raise ValueError(f"consistensicy_loss must use with envmap_data_file!")
            else:
                env_exist_label = 1
                if os.path.isfile(env_path):
                    env_hdr = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
                    env_hdr = cv2.cvtColor(env_hdr, cv2.COLOR_BGR2RGB).astype(np.float32) # env_hdr = cv2.GaussianBlur(env_hdr, ksize=(33,33), sigmaX= 11)
                    env_hdr = cv2.resize(env_hdr,(64,32)).clip(0,10) # (32, 64, 3)
                    env_hdr = torch.from_numpy(np.stack([env_hdr], axis=0)).float() # 0~1+  #[1, 3, 32, 64]
                    env_hdr = env_hdr.movedim(-1, 1)
                    
                    # print("shape", input_gt.shape, input_fg.shape, env_hdr.shape) # [3, 512, 512]) torch.Size([3, 512, 512]) torch.Size([1, 3, 32, 64])
                else:
                    env_hdr = torch.zeros((1, 3, 32, 64))
                    env_exist_label = 0
                return {            
                    "gt": input_gt,      # gt
                    "fg": input_fg,      # degradation_pil
                    "prompt_embeds": prompt_embeds,
                    "env_hdr":env_hdr,
                    "env_exist_label":env_exist_label,
                }
        except:
            import traceback
            print(traceback.format_exc())
            return self.get_random()

def collate_fn(examples):
    gts = torch.stack([example["gt"] for example in examples])
    gts = gts.to(memory_format=torch.contiguous_format).float()
    fgs = torch.stack([example["fg"] for example in examples])
    fgs = fgs.to(memory_format=torch.contiguous_format).float()
    prompt_ids = torch.stack([example["prompt_embeds"].clone().detach() for example in examples])
    env_hdrs = torch.cat([example["env_hdr"] for example in examples]) # [bs, 32, 64, 3]
    env_exist_labels = [example["env_exist_label"] for example in examples]

    # print("shape=====", fgs.shape, gts.shape, env_hdrs.shape, len(env_exist_labels) )

    return {
        "fgs": fgs,
        "gts": gts,
        "prompt_ids":prompt_ids,
        "env_hdrs":env_hdrs,
        "env_exist_labels":env_exist_labels,
    }

def collate_fn_2(examples):
    gts = torch.stack([example["gt"] for example in examples])
    gts = gts.to(memory_format=torch.contiguous_format).float()
    fgs = torch.stack([example["fg"] for example in examples])
    fgs = fgs.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([example["prompt_embeds"].clone().detach() for example in examples])

    env_hdrs = torch.cat([example["env_hdr"] for example in examples]) # [bs, 32, 64, 3]
    env_hdrs_1 = torch.cat([example["env_hdr_1"] for example in examples])
    env_hdrs_2 = torch.cat([example["env_hdr_2"] for example in examples])
    masks = torch.cat([example["mask"] for example in examples])

    # print("shape=====", fgs.shape, gts.shape, env_hdrs.shape, env_hdrs_1.shape, env_hdrs_2.shape, masks.shape)

    return {
        "fgs": fgs,
        "gts": gts,
        "prompt_ids":prompt_ids,

        "env_hdrs":env_hdrs,
        "env_hdrs_1":env_hdrs_1,
        "env_hdrs_2":env_hdrs_2,
        "masks":masks,
    }


@torch.inference_mode()
def encode_prompt_inner(txt: str, tokenizer, text_encoder, device):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = [[id_start] + tokens[i: i + chunk_length] + [id_end] for i in range(0, len(tokens), chunk_length)]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]
    token_ids = torch.tensor(chunks).to(dtype=torch.int64, device=device)
    conds = text_encoder(token_ids).last_hidden_state

    return conds

@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt, tokenizer, text_encoder, device):
    c = encode_prompt_inner(positive_prompt, tokenizer, text_encoder, device)
    uc = encode_prompt_inner(negative_prompt, tokenizer, text_encoder, device)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)
    return c, uc          


def encode_prompt(prompt, text_encoder, tokenizer, device):
    with torch.no_grad():
        untruncated_ids = tokenizer(prompt,  padding="longest",  return_tensors="pt").input_ids 
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", )
        text_input_ids = text_inputs.input_ids
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0]
        bs_embed, seq_len, _ = prompt_embeds.shape # [4, 77, 768]
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    return prompt_embeds


class ICwithMLP(torch.nn.Module):
    def __init__(self, unet, mlp, fivemlp=None):
        super().__init__()
        self.unet = unet
        self.mlp = mlp
        self.fivemlp = fivemlp


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # sd1.5 model ：
    sd15_name = args.pretrained_model_name_or_path
    tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")# , local_files_only = True
    text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
    ddpm_scheduler = DDPMScheduler.from_pretrained(sd15_name, subfolder="scheduler")
    noise_scheduler = ddpm_scheduler

    # unet:改为8通道
    unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d(8, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
        new_conv_in.weight.zero_()  # 零权重初始化
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        new_conv_in.bias = unet.conv_in.bias
        unet.conv_in = new_conv_in
    unet_original_forward = unet.forward
    
    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
        c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
        new_sample = torch.cat([sample, c_concat], dim=1) # 8通道
        kwargs['cross_attention_kwargs'] = {}
        return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
    unet.forward = hooked_unet_forward

    ### load_unet新unet：
    if args.pretrained_unet_checkpoint is not None:
        sd_new = sf.load_file(args.pretrained_unet_checkpoint)
        unet.load_state_dict(sd_new, strict=True)
        del sd_new
    ###

    unet.set_attn_processor(AttnProcessor2_0())
    vae.set_attn_processor(AttnProcessor2_0())

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if weights !=[]:  # jh modify:  save_state时取消了is_main_process的判断，因此会执行多卡遍save_state
                        model.save_pretrained(os.path.join(output_dir, 'unet'))
                        logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! save_model_hook')
                        weights.pop()
                    
        # `accelerator.load_state(...)`
        def load_model_hook(models, model_path):
            for _ in range(len(models)):
                logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! load_model_hook')
                model = models.pop()
                # load_model = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet") # , local_files_only = True
                # model.register_to_config(**load_model.config)
                # # model.load_state_dict(load_model.state_dict())
                # del load_model

                # with torch.no_grad():
                #     new_conv_in = torch.nn.Conv2d(8, model.conv_in.out_channels, model.conv_in.kernel_size, model.conv_in.stride, model.conv_in.padding)
                #     new_conv_in.weight.zero_()
                #     new_conv_in.weight[:, :4, :, :].copy_(model.conv_in.weight)
                #     new_conv_in.bias = model.conv_in.bias
                #     model.conv_in = new_conv_in
                # model_original_forward = model.forward
                # def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
                #     c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
                #     c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
                #     new_sample = torch.cat([sample, c_concat], dim=1) # 8通道
                #     kwargs['cross_attention_kwargs'] = {}
                #     return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)
                # model.forward = hooked_unet_forward
                
                ### load_unet新unet：
                sd_new = sf.load_file(model_path+"/diffusion_pytorch_model.safetensors")
                model.load_state_dict(sd_new, strict=True)
                del sd_new
                ###
                logger.info('==================finish loading model:unet!!!!!!!!!!')

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, 'unet'))
                    weights.pop()

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention(attention_op=None)
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    if "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config:
        accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": [args.adam_beta1, args.adam_beta2],
                "eps": args.adam_epsilon,
                "weight_decay": args.adam_weight_decay,
            },
        }
    if "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config:
        accelerator.state.deepspeed_plugin.deepspeed_config["scheduler"] = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.lr_warmup_steps,
            },
        }
    logger.info("Using DeepSpeed optimizer.")
    # Optimizer creation
    optimizer_class = accelerate.utils.DummyOptim
    
    ###================modify:
    mlp = MLP() 
    if args.pretrained_mlp_checkpoint is not None:
        mlp_new = torch.load(args.pretrained_mlp_checkpoint)['model']
        mlp.load_state_dict(mlp_new, strict=True)
        del mlp_new
    mlp.train()

    if args.consistensicy_loss:
        # fivemlp = FiveLayerMLP(latent_dim=64*64*4)
        # fivemlp.train()
        # models_to_opt = ICwithMLP(unet, mlp, fivemlp)
        # params_to_opt = itertools.chain(
        #     models_to_opt.unet.parameters(),  
        #     models_to_opt.mlp.parameters(),
        #     models_to_opt.fivemlp.parameters())
        models_to_opt = ICwithMLP(unet, mlp)
        params_to_opt = itertools.chain(
            models_to_opt.unet.parameters(),  
            models_to_opt.mlp.parameters())
    else:
        models_to_opt = ICwithMLP(unet, mlp)
        params_to_opt = itertools.chain(
            models_to_opt.unet.parameters(),  
            models_to_opt.mlp.parameters())

    optimizer = optimizer_class(params_to_opt)
    lr_scheduler = accelerate.utils.DummyScheduler(
        optimizer,
        total_num_steps=args.max_train_steps,
        warmup_num_steps=args.lr_warmup_steps,)
    ###================

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # modify:
    models_to_opt.unet.to(accelerator.device, dtype=weight_dtype)
    models_to_opt.mlp.to(accelerator.device, dtype=weight_dtype)
    # if args.consistensicy_loss:
    #     models_to_opt.fivemlp.to(accelerator.device, dtype=weight_dtype)

    logger.info('loading dataset json file')
    null_embeddings = encode_prompt([""]*args.train_batch_size, text_encoder, tokenizer, accelerator.device)
    # del text_encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    envmap_items = json.load(open(args.envmap_file,'r'))
    # envmap_items += json.load(open(args.envmap_file_new,'r'))
    if args.consistensicy_loss:
        all_items = envmap_items
    else:
        all_items = json.load(open(args.dataset_file,'r'))
    
    train_dataset = ComposeDataset(items_data = all_items, envmap_items = envmap_items, add_shadow=args.add_shadow, consistensicy_loss=args.consistensicy_loss, size=args.resolution)
    if args.consistensicy_loss:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn_2,
            batch_size=args.train_batch_size,
            drop_last=True,  # jh modify
            pin_memory=True, # jh modify
            num_workers=args.dataloader_num_workers, )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            drop_last=True,  # jh modify
            pin_memory=True, # jh modify
            num_workers=args.dataloader_num_workers, )
        
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Prepare everything with our `accelerator`. modify:
    models_to_opt, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        models_to_opt, optimizer, train_dataloader, lr_scheduler)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        # accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        accelerator.init_trackers(args.tracker_project_name, 
                                  config=tracker_config,
                                    init_kwargs={
                                        "wandbx": {
                                            "name": args.tracker_run_name,
                                            "id": "%s"%time.strftime('%X %x %Z').replace(':','_').replace(' ','__').replace('/','_'),
                                            "resume": "allow",
                                            "allow_val_change": True,
                                        }
                                    },
                                )

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("whole_unet")] # 根据保存的时候的设置更改此处
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    lr = 0.0
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(models_to_opt): # modify
                # Convert images to latent space
                if args.pretrained_vae_model_name_or_path is not None:
                    gt = batch["gts"].to(dtype=weight_dtype)  # ori_img
                    fg = batch["fgs"].to(dtype=weight_dtype)  # composite
                else:
                    gt = batch["gts"]
                    fg = batch["fgs"]
                latents = vae.encode(gt.to(accelerator.device)).latent_dist.sample() * vae.config.scaling_factor
                concat_conds = vae.encode(fg.to(accelerator.device)).latent_dist.sample() * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)
                bsz = latents.shape[0]

                # text embeds:
                text_embeddings = batch["prompt_ids"].to(weight_dtype) # [bs,77,768]
                if args.cfg_drop_ratio>0:
                    null_embeddings.to(accelerator.device)
                    assert null_embeddings.shape == text_embeddings.shape
                    mask_text = torch.zeros((text_embeddings.shape[0],1,1), dtype=text_embeddings.dtype, device=text_embeddings.device)
                    for i in range(text_embeddings.shape[0]):
                        mask_text[i] = 1 if random.random() > args.cfg_drop_ratio else 0
                    text_embeddings =  mask_text * text_embeddings + (1-mask_text) * null_embeddings

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) + args.noise_offset * torch.randn(
                        latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype
                    )

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Predict the noise residual 
                cross_attention_kwargs={'concat_conds':concat_conds}

                # ====================================env embeds:
                env_hdrs = batch["env_hdrs"].to(dtype=weight_dtype) # [bs, 3, 32, 64]
                env_hdrs = env_hdrs[:,:,:16,:] # 只取正z半轴  bs*3*16*64
                env_hdr_flattened = env_hdrs.reshape(bsz, -1) 
                env_hdr_embedding = models_to_opt.mlp(env_hdr_flattened).reshape(bsz, 3, 768)

                if args.consistensicy_loss:
                    encoder_hidden_states = torch.cat([env_hdr_embedding, text_embeddings], dim = 1) [:, :77,:]

                    env_hdrs_1 = batch["env_hdrs_1"].to(dtype=weight_dtype) # [bs, 32, 64, 3]
                    env_hdrs_1 = env_hdrs_1[:,:,:16,:] # 只取正z半轴  bs*3*16*64
                    env_hdr_flattened_1 = env_hdrs_1.reshape(bsz, -1) 
                    env_hdr_embedding_1 = models_to_opt.mlp(env_hdr_flattened_1).reshape(bsz, 3, 768)

                    env_hdrs_2 = batch["env_hdrs_2"].to(dtype=weight_dtype) # [bs, 32, 64, 3]
                    env_hdrs_2 = env_hdrs_2[:,:,:16,:] # 只取正z半轴  bs*3*16*64
                    env_hdr_flattened_2 = env_hdrs_2.reshape(bsz, -1) 
                    env_hdr_embedding_2 = models_to_opt.mlp(env_hdr_flattened_2).reshape(bsz, 3, 768)
                    
                    encoder_hidden_states_1 = torch.cat([env_hdr_embedding_1, text_embeddings], dim = 1) [:, :77,:]
                    encoder_hidden_states_2 = torch.cat([env_hdr_embedding_2, text_embeddings], dim = 1) [:, :77,:]
                    model_pred_1 = models_to_opt.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states_1,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    model_pred_2 = models_to_opt.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states_2,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    # model_pred_1_mlp = models_to_opt.fivemlp(model_pred_1) 
                    # model_pred_2_mlp = models_to_opt.fivemlp(model_pred_2)
                    model_pred_1_mlp = model_pred_1
                    model_pred_2_mlp = model_pred_2

                else:
                    env_exist_labels = batch["env_exist_labels"]
                    encoder_hidden_state = []
                    for idx, env_exist_label in enumerate(env_exist_labels):
                        if env_exist_label==1:
                            encoder_hidden_state.append(torch.cat([env_hdr_embedding[idx,:,:], text_embeddings[idx,:,:]], dim = 0) [:77,:])
                        elif env_exist_label==0:
                            encoder_hidden_state.append(text_embeddings[idx,:,:])
                    encoder_hidden_states = torch.stack(encoder_hidden_state)

                # ====================================
                model_pred = models_to_opt.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                ### modify: loss设计：
                vanilla_loss =  F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                if args.consistensicy_loss:
                    masks = batch["masks"].to(dtype=weight_dtype)
                    masked_loss_consistency =  F.mse_loss((masks*model_pred).float(), masks*(model_pred_1_mlp+model_pred_2_mlp).float(), reduction="mean")
                    loss = vanilla_loss + 0.1 * masked_loss_consistency
                else:
                    loss = vanilla_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients: 
                    # 梯度裁剪:防止梯度爆炸
                    params_to_clip = itertools.chain(
                            models_to_opt.mlp.parameters(),  
                            models_to_opt.unet.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                try:
                    ### jh modify: 用了之后deepspeed会卡住？
                    # if accelerator.is_main_process:
                    # if global_step % args.checkpointing_steps == 0:
                    #     save_dir = os.path.join(args.output_dir, f"unet_offest-{global_step}")
                    #     logger.info(f"正打算保存模型： {save_dir}")
                    #     accelerator.save_state(save_dir)
                    #     logger.info(f"Saved state to {save_dir}")

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            accelerator.unwrap_model(models_to_opt.unet).save_pretrained( os.path.join(args.output_dir, f"whole_unet-{global_step}") , safe_serialization=True)
                            mlp_save_dir = os.path.join(args.output_dir, f"mlp-{global_step}")
                            os.makedirs(mlp_save_dir, exist_ok=True)
                            torch.save({'model': accelerator.unwrap_model(models_to_opt.mlp).state_dict()}, os.path.join(mlp_save_dir, "mlp.pth"))
                            logger.info('=======================保存完整unet')
                            
                except:
                    import traceback
                    logger.error(traceback.format_exc())

                if accelerator.is_main_process:
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            noise_scheduler, vae, text_encoder, tokenizer, models_to_opt, args, accelerator, weight_dtype, global_step
                        )
                try:
                    lr = lr_scheduler.get_last_lr()[0]
                except Exception as e:
                    logger.error(
                        f"Failed to get the last learning rate from the scheduler. Error: {e}"
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        models_to_opt = accelerator.unwrap_model(models_to_opt)
        models_to_opt.unet.save_pretrained(os.path.join(args.output_dir, f"unet") )
        models_to_opt.mlp.save_pretrained(os.path.join(args.output_dir, f"mlp") )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

