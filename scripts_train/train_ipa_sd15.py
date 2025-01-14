import os
import cv2
import gc
import numpy as np
import argparse
import itertools
import json
import random
import time
from pathlib import Path
import glob
import safetensors.torch as sf

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline

from ip_adapter_official.ip_adapter import ImageProjModel
from ip_adapter_official.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter_official.attention_processor import AttnProcessor2_0 as AttnProcessor
    from ip_adapter_official.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
else:
    from ip_adapter_official.attention_processor import AttnProcessor, IPAttnProcessor
from diffusers.utils import make_image_grid
from utils.utils_degradation import get_IBL
from ip_adapter_official.ip_adapter import IPAdapter as IPAdapter_full

def log_validation( noise_scheduler, unet, vae, text_encoder, tokenizer, image_encoder, save_path, 
                   args, accelerator, weight_dtype, global_step ):
    # Pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        scheduler = noise_scheduler, 
        safety_checker=None,
        requires_safety_checker=False,
        # local_files_only = True,
    )
    pipeline.set_progress_bar_config(disable=True)
    ip_model = IPAdapter_full(pipeline, accelerator.unwrap_model(image_encoder), save_path+"/ip_adapter.bin", accelerator.device)


    ###
    prompts = []
    fgs = glob.glob("/fg/*")
    fgs.sort()
    masks = glob.glob("/mask/*")
    masks.sort()
    bgs = glob.glob("/inpaintbg/*")
    bgs.sort()
    ###

    image_logs = []
    
    # modify:
    if args.envmap_or_bg=='envmap':
        env_image = Image.open("/env_panorama.png").convert("RGB")
    elif args.envmap_or_bg=='bg':
        env_image = Image.open("/bgs/15.png").convert("RGB")
    for  idx, (validation_image_path, validation_mask_path, txt_path, bg_path) in enumerate(zip(fgs, masks, prompts, bgs)):
        validation_prompt = prompts[idx] + ', best quality'
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
        
        concat_conds =  torch.from_numpy(np.stack([fg], axis=0)).float() / 127.0 - 1.0 # 0~255 to -1~+1   #[1,h,w,c]
        concat_conds = concat_conds.clip(-1,1).movedim(-1, 1).to(device=vae.device, dtype=vae.dtype)  #[1,c,h,w]
        concat_conds = vae.encode(concat_conds).latent_dist.sample() * vae.config.scaling_factor
    
        # unconds = encode_prompt(negative_prompt, text_encoder, tokenizer, vae.device)
        # text_embeddings = encode_prompt(validation_prompt, text_encoder, tokenizer, vae.device) # [bs, 77, 768]
        # encoder_hidden_states = torch.cat([env_hdr_embedding, text_embeddings], dim = 1) [:, :77,:]

        with torch.autocast("cuda"):
            ip_scale_list = [0.2, 0.5, 0.8]
            images = []
            for ip_scale in ip_scale_list:
                image = ip_model.generate(
                    pil_image=env_image,
                    prompt=validation_prompt,
                    negative_prompt=negative_prompt,
                    scale=ip_scale,
                    
                    num_inference_steps=30, 
                    seed = 0,
                    guidance_scale=7.0,
                    height = height,
                    width = width,
                    cross_attention_kwargs={'concat_conds': concat_conds},
                )[0]
                images.append(image)
            if global_step:
                os.makedirs('%s/tmp/'%args.output_dir, exist_ok=True)
                grid = make_image_grid(images, rows=1,cols=len(images))
                grid.save('%s/tmp/%d_%d_img.png'%(args.output_dir, global_step, idx))     
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
                tracker.writer.add_images(validation_prompt, formatted_images, global_step, dataformats="NHWC")
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
            print(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs



# Dataset
class ComposeDataset(Dataset):
    def __init__(self, items_data, envmap_items, envmap_or_bg= 'bg', add_shadow=False, size=512):
        self.size = size
        self.add_shadow = add_shadow
        self.envmap_or_bg = envmap_or_bg

        self.items_data = items_data
        self.envmap_items = envmap_items

        self.img_source_name, self.img_cond_name = 'source', 'control_seg'
        self.img_dir = {}
        self.cond_img_path = {}
        self.cache_dir_emb = {}
        self.cache_dir_emb_randstr_playgroundg = ''


        self.clip_image_processor = CLIPImageProcessor()
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size),
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

            cache_path = os.path.join(self.cache_dir_emb[type_], self.items_data[index]['emb_cache'])
            if not os.path.isfile(cache_path) :
                if type_=="playground_v2.5":
                    cache_path = os.path.join(self.cache_dir_emb_randstr_playgroundg, self.items_data[index]['emb_cache'])
            emb_cache = torch.load(cache_path, weights_only=True)
            prompt_embeds = emb_cache['prompt_embeds'] # [77, 2048]
            prompt_embeds = prompt_embeds[:, :768]

            # 得到gt的环境图贴图：
            if self.envmap_or_bg =='envmap':
                env_path = os.path.join(self.exr_outdir, type_, os.path.splitext(self.items_data[index][self.img_source_name])[0]+'.exr')
                env_hdr = cv2.imread(env_path, cv2.IMREAD_UNCHANGED)
                env_hdr = cv2.cvtColor(env_hdr, cv2.COLOR_BGR2RGB).astype(np.float32)
                env_hdr = cv2.resize(env_hdr,(64,32)).clip(0, 1) # (32, 64, 3)
                env_hdr = Image.fromarray((env_hdr*255).astype(np.uint8))
                clip_image = self.clip_image_processor(images=env_hdr, return_tensors="pt").pixel_values
            elif self.envmap_or_bg =='bg':
                bg_path = os.path.join(self.inpaintbg_outdir, type_, self.items_data[index][self.img_source_name])
                bg_pil = Image.open(bg_path).convert('RGB')
                bg_pil = bg_pil.resize((degradation_pil.width, degradation_pil.height), Image.BILINEAR)
                clip_image = self.clip_image_processor(images=bg_pil, return_tensors="pt").pixel_values
            return {            
                "fg": input_fg,      # degradation_pil
                "gt": input_gt,     # gt
                "prompt_embeds": prompt_embeds,
                "clip_image":clip_image,

            }
        except:
            import traceback
            print(traceback.format_exc())
            return self.get_random()


def collate_fn(examples):
    fgs = torch.stack([example["fg"] for example in examples])
    fgs = fgs.to(memory_format=torch.contiguous_format).float()
    gts = torch.stack([example["gt"] for example in examples])
    gts = gts.to(memory_format=torch.contiguous_format).float()
    prompt_ids = torch.stack([example["prompt_embeds"].clone().detach() for example in examples])
    clip_images = torch.cat([example["clip_image"] for example in examples])

    # print("shape=====", fgs.shape, gts.shape, clip_images.shape)

    return {
        "fgs": fgs,
        "gts": gts,
        "prompt_ids":prompt_ids,
        "clip_images": clip_images,
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""

    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, cross_attention_kwargs):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
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
        "--resolution",
        type=int,
        default=512,
        help=("The resolution for input images"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=("Save a checkpoint of the training state every X updates"),
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
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--pretrained_unet_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--envmap_or_bg",
        type=str,
        default='bg',
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="ipa",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default="ipa_run",
        help=(
            "The `run_name` argument passed to Accelerator.init_trackers for"
        ),
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

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


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path) 
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ### modify: 8通道ic unet
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
    if args.pretrained_unet_checkpoint is not None:
        sd_new = sf.load_file(args.pretrained_unet_checkpoint)
        unet.load_state_dict(sd_new, strict=True)
        del sd_new
    ###

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim= image_encoder.config.projection_dim, 
        clip_extra_context_tokens=4,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    # optimizer
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    null_embeddings = encode_prompt([""]*args.train_batch_size, text_encoder, tokenizer, accelerator.device)
    gc.collect()
    torch.cuda.empty_cache()
    
    envmap_items = json.load(open(args.data_json_file,'r'))
    all_items = envmap_items

    train_dataset = ComposeDataset(all_items, envmap_items, envmap_or_bg=args.envmap_or_bg,  add_shadow=False, size=args.resolution)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

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

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                gt = batch["gts"].to(accelerator.device,dtype=weight_dtype)
                fg = batch["fgs"].to(accelerator.device,dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(gt).latent_dist.sample() * vae.config.scaling_factor
                    concat_conds = vae.encode(fg).latent_dist.sample() * vae.config.scaling_factor
                    latents = latents.to(weight_dtype)
                    bsz = latents.shape[0]

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    image_embeds = image_encoder(
                        batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                    ).image_embeds
                
                # text embeds:
                text_embeddings = batch["prompt_ids"].to(weight_dtype) # [bs,77,768]
                cross_attention_kwargs={'concat_conds':concat_conds}
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeddings, image_embeds, cross_attention_kwargs)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(
                        "Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                            epoch, step, load_data_time, time.perf_counter() - begin, avg_loss
                        )
                    )

            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    # accelerator.save_state(save_path) # 不保存ipa_model ?
                    state_dict_old = accelerator.unwrap_model(ip_adapter).state_dict()
                    state_dict_new = {"image_proj": {}, "ip_adapter": {}}
                    for key in state_dict_old.keys():
                        if key.startswith("image_proj_model."):
                            state_dict_new["image_proj"][key.replace("image_proj_model.", "")] = state_dict_old[key]
                        elif key.startswith("adapter_modules."):
                            state_dict_new["ip_adapter"][key.replace("adapter_modules.", "")] = state_dict_old[key]

                    # print(state_dict_new["image_proj"].keys(),state_dict_new["ip_adapter"].keys())
                    torch.save(state_dict_new, save_path + "/ip_adapter.bin")
                    image_logs = log_validation( noise_scheduler, ip_adapter.unet, vae, text_encoder, tokenizer, image_encoder, save_path,
                                                args, accelerator, weight_dtype, global_step )

            logs = {
                "step_loss": loss.detach().item(),
            }
            accelerator.log(logs, step=global_step)
            
            begin = time.perf_counter()


if __name__ == "__main__":
    main()
