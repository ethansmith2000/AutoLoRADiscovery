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
from common.utils import make_weight_vector
from common.train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    get_optimizer
)

from types import SimpleNamespace
from discover_lora_diffusion.models import LoraDiffusion
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


default_arguments = dict(
    data_dir="/home/ubuntu/AutoLoRADiscovery/lora_bundle_for_model.pt",
    output_dir="diffusion_lora",
    seed=None,
    train_batch_size=32,
    max_train_steps=1250,
    # validation_steps=250,
    num_dataloader_repeats=100,
    checkpointing_steps=500,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=5.0e-5,
    lr_scheduler="constant",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=True,
    adam_beta1=0.9,
    adam_beta2=0.99,
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
    model_dim = 256,
    ff_mult = 3,
    chunks = 1,
    act = torch.nn.SiLU,
    encoder_layers = 6,
    decoder_layers = 12,

    kld_weight = 0.1
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_diffusion = LoraDiffusion(data_dim=args.data_dim,
                        model_dim=args.model_dim,
                        ff_mult=args.ff_mult,
                        chunks=args.chunks,
                        act=args.act,
                        encoder_layers=args.encoder_layers,
                        decoder_layers=args.decoder_layers
                        )
    scheduler = diffusers.DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

    params_to_optimize = list(lora_diffusion.parameters())
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)

    # Prepare everything with our `accelerator`.
    lora_diffusion, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_diffusion, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("diffusion_lora", config=tracker_config)

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
        lora_diffusion.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_diffusion):
                if random.random() < 0.85:
                    coef = torch.rand(batch.shape[0]).to(batch.device)
                    batch = coef[:, None] * batch + (1 - coef[:, None]) * batch.flip(0)

                noise = torch.randn_like(batch)
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device
                ).long()

                noisy_model_input = scheduler.add_noise(batch, noise, timesteps)

                pred = lora_diffusion(noisy_model_input, timesteps)
                mse_loss = F.mse_loss(pred.float(), batch.float(), reduction="mean")

                accelerator.backward(mse_loss)
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

            logs = {"mse_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                pass

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_vae.pth")
        torch.save(unwrap_model(accelerator, lora_vae).state_dict(), save_path)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)