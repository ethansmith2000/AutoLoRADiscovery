#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
from types import SimpleNamespace

import sys
sys.path.append('/home/ubuntu/AutoLoRADiscovery/')

# from common import utils
# from common import loras
# from common import train_utils

# from utils import convert_to_multi
# from loras import patch_lora
# from train_utils import (
#     collate_fn,
#     init_train_basics,
#     log_validation,
#     unwrap_model,
#     MyDataset,
#     load_models,
#     get_train_stuff,
#     save_model,
# )

from common.utils import convert_to_multi
from common.loras import patch_lora
from common.train_utils import (
    collate_fn,
    init_train_basics,
    log_validation,
    unwrap_model,
    MyDataset,
    load_models,
    get_optimizer,
    get_dataset,
    save_model,
)



default_arguments = dict(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    revision="main",
    variant=None,
    tokenizer_name=None,
    instance_data_dir="/home/ubuntu/AutoLoRADiscovery/people/Jeff Bezos",
    num_validation_images=4,
    num_class_images=100,
    output_dir="PCLora",
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
    lora_rank=8,
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
    bundle_lora_path="/home/ubuntu/AutoLoRADiscovery/lora_bundle.pt",
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    args.instance_prompt= f"A photo of {args.subject}"
    args.validation_prompt = [f"majestic fantasy painting of {args.subject}", f"a comic book drawing of {args.subject}", f"HD cinematic photo of {args.subject}", f"oil painting of {args.subject} by van gogh"]
    accelerator, weight_dtype = init_train_basics(args, logger)

    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(args, accelerator, weight_dtype)
    
    # load in bundled lora
    bundle_lora = torch.load(args.bundle_lora_path)
    num_loras = len(bundle_lora)

    # now we will add new LoRA weights to the attention layers
    patch_lora(unet, rank=args.lora_rank, included_terms=args.lora_layers, num_loras=num_loras)

    # The text encoder comes from transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        patch_lora(text_encoder, rank=args.lora_rank, included_terms=args.lora_layers_te, num_loras=num_loras)

    if isinstance(bundle_lora, list):
        bundle_lora = [convert_to_multi(lora, i) for i, lora in enumerate(bundle_lora)]
        bundle_dict = {}
        for lora in bundle_lora:
            bundle_dict.update(lora)
    
    # save it
    torch.save(bundle_lora, args.bundle_lora_path)
    missing, unexpected = text_encoder.load_state_dict(bundle_dict, strict=False)
    missing, unexpected = unet.load_state_dict(bundle_dict, strict=False)

    # Optimizer creation
    params_to_optimize = [p for n,p in unet.named_parameters() if "each_scale" in n]
    names_to_optimize = [n for n,p in unet.named_parameters() if "each_scale" in n]
    if args.train_text_encoder:
        params_to_optimize = params_to_optimize + [p for n,p in text_encoder.named_parameters() if "each_scale" in n]
        names_to_optimize = names_to_optimize + [n for n,p in text_encoder.named_parameters() if "each_scale" in n]
    
    num_params = sum(p.numel() for p in params_to_optimize)
    args.num_params = num_params
    print("Number of parameters training: ", num_params)
    print("Training following weights: ", names_to_optimize)

    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args, tokenizer)

    # Prepare everything with our `accelerator`.
    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)

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
    initial_global_step = 0
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
        global_step = int(args.resume_from_checkpoint.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(model_input)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (noise.shape[0],), device=model_input.device
                ).long()
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device),return_dict=False,)[0]

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(unet, text_encoder,accelerator,save_path, args, logger)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                # print(global_step, global_step % args.validation_steps)
                if args.validation_prompt is not None and global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        unet,
                        text_encoder,
                        weight_dtype,
                        args,
                        accelerator,
                        pipeline_args={"prompt": args.validation_prompt, "height": args.resolution, "width": args.resolution},
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_layers.pth")
        save_model(unet, text_encoder, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)