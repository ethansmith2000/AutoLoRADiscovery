
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb
import logging
import math
import loras
from loras import patch_lora
import random
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": torch.cat(input_ids, dim=0),
        "pixel_values": pixel_values,
    }

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example



def save_model(unet, text_encoder, accelerator, save_path, args, logger, keyword="lora"):
    unet_state_dict = {k:v for k,v in unwrap_model(accelerator, unet).state_dict().items() if keyword in k}
    text_encoder_state_dict = None
    if args.train_text_encoder:
        text_encoder_state_dict = {k:v for k,v in unwrap_model(accelerator, text_encoder).state_dict().items() if keyword in k}

    state_dict = {}
    state_dict.update(unet_state_dict)
    state_dict.update(text_encoder_state_dict)

    full_state_dict ={
        # "unet": unet_state_dict,
        # "text_encoder": text_encoder_state_dict,
        "state_dict": state_dict,
        "num_params": args.num_params,
        "lora_layers": args.lora_layers,
        "lora_layers_te": args.lora_layers_te,
    }

    torch.save(full_state_dict, save_path)
    logger.info(f"Saved state to {save_path}")

class MyDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        size=512,
        center_crop=False,
        num_dataloader_repeats=100,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = [os.path.join(instance_data_root, f) for f in os.listdir(instance_data_root) if not ".txt" in f]
        random.shuffle(self.instance_images_path)
        self.instance_images_path = self.instance_images_path[:20]
        self.instance_images_path = self.instance_images_path * num_dataloader_repeats
        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images]).convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
                self.instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def log_validation(
    unet,
    text_encoder,
    weight_dtype,
    args,
    accelerator,
    pipeline_args,
    epoch,
    logger,
    is_final_validation=False,
):

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unwrap_model(accelerator, unet),
        text_encoder=unwrap_model(accelerator, text_encoder),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.safety_checker = None
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    images = []
    with torch.cuda.amp.autocast():
        images.extend(pipeline(**pipeline_args, generator=generator).images)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images


def init_train_basics(args, logger):
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator, weight_dtype


default_arguments = dict(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    revision="main",
    variant=None,
    tokenizer_name=None,
    instance_data_dir="/home/ubuntu/AutoLoRADiscovery/people/Jeff Bezos",
    num_validation_images=4,
    num_class_images=100,
    output_dir="lora-dreambooth-model",
    seed=None,
    resolution=640,
    center_crop=False,
    train_batch_size=1,
    max_train_steps=1250,
    validation_steps=250,
    num_dataloader_repeats=100,
    checkpointing_steps=500,
    checkpoints_total_limit=None,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    # learning_rate=1.0e-4,
    learning_rate=6.0e-5,
    # learning_rate=2.5e-5,
    scale_lr=False,
    lr_scheduler="constant",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    # lora_rank=16,
    lora_rank=8,
    local_rank=-1,
    num_processes=1,

    subject = "sks person",

    lora_layers = [
            "attn2.to_q", 
            "attn2.to_k", 
            "attn2.to_v", 
            "attn2.to_out",
            # "proj_in",
            # "proj_out",
            # "norm",
            #  "ff", 
    ],
    lora_layers_te = [
        "final_layer_norm",
        "7",
        "8",
        "9",
        "10",
        "11"
    ],
    train_text_encoder=True,
)



def load_models(args, accelerator, weight_dtype):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    ).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    return tokenizer, noise_scheduler, text_encoder, vae, unet




def get_train_stuff(args, params_to_optimize, tokenizer, accelerator):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = MyDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return optimizer, train_dataset, train_dataloader, lr_scheduler, num_update_steps_per_epoch



    