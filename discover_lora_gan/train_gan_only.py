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
from discover_lora_gan.models import Generator, Discriminator
from torch.utils.data import Dataset
import random

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


default_arguments = dict(
    data_dir="/home/ubuntu/AutoLoRADiscovery/lora_bundle_for_model.pt",
    output_dir="vae_lora",
    seed=None,
    train_batch_size=32,
    max_train_steps=6000,
    # validation_steps=250,
    num_dataloader_repeats=100,
    checkpointing_steps=2000,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1.0e-4,
    lr_scheduler="linear",
    lr_warmup_steps=200,
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

    data_dim = 1_365_504,

    kld_weight = 0.005,

    lora_std = 0.0152,
    num_latent_tokens = 1
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_generator = Generator(
                        dim = 1536,
                        num_tokens=889,
                        num_layers=6,
                        num_heads=16,
                        mlp_ratio=3.0,
                        latent_dim=64,)

    lora_discriminator = Discriminator(
                        dim = 1536,
                        num_tokens=889,
                        num_layers=4,
                        num_heads=16,
                        mlp_ratio=3.0,)

    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer_g, lr_scheduler_g = get_optimizer(args, list(lora_generator.parameters()), accelerator)
    optimizer_d, lr_scheduler_d = get_optimizer(args, list(lora_discriminator.parameters()), accelerator)

    weight_dict = train_dataset.weight_dict

    # Prepare everything with our `accelerator`.
    lora_generator, lora_discriminator, optimizer_g, optimizer_d, train_dataloader, lr_scheduler_g, lr_scheduler_d  = accelerator.prepare(
        lora_generator, lora_discriminator, optimizer_g, optimizer_d, train_dataloader, lr_scheduler_g, lr_scheduler_d
    )

    global_step, first_epoch, progress_bar = more_init([lora_generator, lora_discriminator], accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, wandb_name="gan_lora_1")

    for epoch in range(first_epoch, args.num_train_epochs):
        lora_generator.train()
        lora_discriminator.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_discriminator):
                batch = batch.to(accelerator.device) * (1 / args.lora_std)
                batch = augmentations(batch, weight_dict, slerp=True)

                with torch.no_grad():
                    latent = torch.randn(batch.size(0), 64, device=accelerator.device)
                    fake = lora_generator(latent)
                
                real = lora_discriminator(batch)
                fake = lora_discriminator(fake)

                loss_d = (F.binary_cross_entropy_with_logits(real, torch.ones_like(real)) + F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))) / 2
                loss_d.backward()
                optimizer_d.step()
                optimizer_d.zero_grad(set_to_none=True)
                lr_scheduler_d.step()

            with accelerator.accumulate(lora_generator):
                latent = torch.randn(batch.size(0), 64, device=accelerator.device)
                fake = lora_generator(latent)
                fake = lora_discriminator(fake)
                loss_g = F.binary_cross_entropy_with_logits(fake, torch.ones_like(fake))
                loss_g.backward()
                optimizer_g.step()
                optimizer_g.zero_grad(set_to_none=True)
                lr_scheduler_g.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        torch.save(unwrap_model(accelerator, lora_generator).state_dict(), save_path)

            logs = {"loss_d": loss_d.detach().item(), "lr": lr_scheduler_g.get_last_lr()[0], "loss_g": loss_g.detach().item()}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_generator.pth")
        torch.save(unwrap_model(accelerator, lora_generator).state_dict(), save_path)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)