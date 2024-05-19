import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import sys
sys.path.append('/home/ubuntu/AutoLoRADiscovery/')

from common.loras import patch_lora
from common.utils import make_weight_vector, augmentations
from common.train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    get_optimizer,
    more_init
)

from types import SimpleNamespace
from discover_lora_diffusion.models import LoraDiffusion, DiT
from torch.utils.data import Dataset
import random
import diffusers

class LoraDataset(Dataset):

    def __init__(
        self,
        lora_bundle_path,
        num_dataloader_repeats=10,
    ):
        self.lora_bundle = torch.load(lora_bundle_path) #* num_dataloader_repeats
        self.lora_bundle = [make_weight_vector(state_dict) for state_dict in self.lora_bundle]
        self.weight_dict = self.lora_bundle[0][1]
        self.lora_bundle = [x[0] for x in self.lora_bundle] * num_dataloader_repeats
        random.shuffle(self.lora_bundle)

    def __len__(self):
        return len(self.lora_bundle)

    def __getitem__(self, index):
        return self.lora_bundle[index]


def collate_fn(examples):
    return torch.stack(examples)


def get_dataset(args):
    train_dataset = LoraDataset(args.data_dir)
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

    return train_dataset, train_dataloader, num_update_steps_per_epoch

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr

def get_loss(noise_pred, noise, timesteps, noise_scheduler, snr_gamma=None):
    if snr_gamma is None:
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
    else:
        snr = compute_snr(noise_scheduler, timesteps)
        snr = snr + 2e-4
        base_weight = (
                torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        )

        if noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = base_weight + 1
        else:
            mse_loss_weights = base_weight
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    
    return loss.mean()

default_arguments = dict(
    data_dir="/home/ubuntu/AutoLoRADiscovery/lora_bundle_for_model.pt",
    output_dir="diffusion_lora",
    seed=None,
    train_batch_size=40,
    max_train_steps=30_000,
    # validation_steps=250,
    num_dataloader_repeats=100,
    checkpointing_steps=2000,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1.5e-5,
    lr_scheduler="linear",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=True,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    snr_gamma=None,
    lora_std = 0.0152
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    # lora_diffusion = LoraDiffusion(data_dim=args.data_dim,
    #                     model_dim=args.model_dim,
    #                     ff_mult=args.ff_mult,
    #                     chunks=args.chunks,
    #                     act=args.act,
    #                     encoder_layers=args.encoder_layers,
    #                     decoder_layers=args.decoder_layers
    #                     )

    lora_diffusion = DiT(
        total_dim=1_365_504,
        dim = 1536,
        num_tokens=889,
        time_dim = 384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
    )

    scheduler = diffusers.DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    params_to_optimize = list(lora_diffusion.parameters())
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)

    weight_dict = train_dataset.weight_dict

    # Prepare everything with our `accelerator`.
    lora_diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_diffusion, optimizer, train_dataloader, lr_scheduler
    )

    global_step, first_epoch, progress_bar = more_init(lora_diffusion, accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, wandb_name="diffusion_lora")


    for epoch in range(first_epoch, args.num_train_epochs):
        lora_diffusion.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_diffusion):
                batch = batch.to(accelerator.device) * (1 / args.lora_std)
                batch = augmentations(batch, weight_dict)

                noise = torch.randn_like(batch)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device
                ).long()

                noisy_model_input = scheduler.add_noise(batch, noise, timesteps)

                pred = lora_diffusion(noisy_model_input, timesteps)
                loss = get_loss(pred, noise, timesteps, scheduler, snr_gamma=args.snr_gamma)

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
                        torch.save(unwrap_model(accelerator, lora_diffusion).state_dict(), save_path)

            logs = {"mse_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_diffusion.pth")
        torch.save(unwrap_model(accelerator, lora_diffusion).state_dict(), save_path)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)