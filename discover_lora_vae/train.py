import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import sys
sys.path.append('..')

from common.loras import patch_lora
from common.utils import make_weight_vector, augmentations, rand_merge
from common.train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    get_optimizer,
    more_init,
    resume_model
)

from types import SimpleNamespace
from discover_lora_vae.models import LoraVAE
from torch.utils.data import Dataset
import random

import numpy as np
from scipy.stats import johnsonsu, norm


def dist(num_samples=10_000, mean=0, std=1, kurtosis=3, skewness=0.0):
    johnson_su = johnsonsu.fit(np.random.normal(size=num_samples), fa=skewness, fb=np.sqrt(kurtosis))
    samples = johnsonsu.rvs(*johnson_su, size=num_samples)
    scaled_samples = std * (samples - np.mean(samples)) / np.std(samples) + mean

    return scaled_samples

class LoraDataset(Dataset):

    def __init__(
        self,
        lora_bundle_path,
        num_dataloader_repeats=20, # this could blow up memory be careful!
    ):
        self.lora_bundle = torch.load(lora_bundle_path)
        self.lora_bundle = [make_weight_vector(state_dict) for state_dict in self.lora_bundle]
        self.weight_dict = self.lora_bundle[0][1]
        self.lora_bundle = [x[0] for x in self.lora_bundle] * num_dataloader_repeats
        random.shuffle(self.lora_bundle)

    def __len__(self):
        return len(self.lora_bundle)

    def __getitem__(self, index):
        return self.lora_bundle[index]



class DummyDataset(Dataset):
    
        def __init__(self, data_dim):
            self.data_dim = data_dim
    
        def __len__(self):
            return 1000
    
        def __getitem__(self, index):
            return torch.randn(self.data_dim)


def collate_fn(examples):
    return torch.stack(examples)


def get_dataset(args):
    # train_dataset = LoraDataset(args.data_dir)
    train_dataset = DummyDataset(1000)
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
    data_dir="/home/ubuntu/AutoLoRADiscovery/lora_bundle.pt",
    output_dir="model-output",
    seed=None,
    train_batch_size=32,
    max_train_steps=60_000,
    num_dataloader_repeats=100,
    checkpointing_steps=5000,
    resume_from_checkpoint="/home/ubuntu/AutoLoRADiscovery/discover_lora_vae/model-output/checkpoint-20000",
    # resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=1.0e-4,
    lr_scheduler="linear",
    lr_warmup_steps=200,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=9,
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

    kld_weight = 0.1,

    lora_std = 0.0152,
    use_wandb=True
)


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    lora_vae = LoraVAE(data_dim=args.data_dim,
                        model_dim=512,
                        ff_mult=3.0,
                        chunks=1,
                        # act=args.act,
                        encoder_layers=20,
                        decoder_layers=20
                        )

    params_to_optimize = list(lora_vae.parameters())
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)
    optimizer, lr_scheduler = get_optimizer(args, params_to_optimize, accelerator)
    # weight_dict = train_dataset.weight_dict

    lora_bundle = torch.load(args.data_dir)
    lora_bundle = [make_weight_vector(state_dict) for state_dict in lora_bundle]
    lora_bundle = [x[0] for x in lora_bundle]
    lora_bundle = torch.stack(lora_bundle).cuda().to(torch.bfloat16) / args.lora_std

    def get_a_lora(std=0.1, kurtosis=3.35):
        weights = dist(num_samples=lora_bundle.shape[0], mean=0, std=std, kurtosis=kurtosis, skewness=0.0)
        weights = torch.tensor(weights).to(lora_bundle.device).to(lora_bundle.dtype)
        weights = weights - weights.mean()
        a_lora = torch.sum(weights[:,None] * lora_bundle, dim=0)
        return a_lora

    # Prepare everything with our `accelerator`.
    lora_vae, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_vae, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(lora_vae, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, 
                                                    train_dataset, logger, num_update_steps_per_epoch, 
                                                    global_step, wandb_name="lora_vae")

    def train_step(first=0.4, first_with_orig=0.01, second=0.4, second_w_orig=0.01, noise_std=0.01):
        # generate random loras by mixing full dataset
        batch = [get_a_lora() for _ in range(args.train_batch_size)]
        batch = torch.stack(batch).to(torch.bfloat16)

        orig_indices = torch.randperm(lora_bundle.shape[0])[:args.train_batch_size]
        orig_batch = lora_bundle[orig_indices]

        orig_batch = orig_batch.float()
        batch = batch.float()

        mask = torch.rand(batch.shape[0]) < first
        if sum(mask) > 0:
            perm = torch.randperm(batch.shape[0])
            if random.random() < first_with_orig:
                batch[mask] = rand_merge(batch[mask], orig_batch[perm][mask])
            else:
                batch[mask] = rand_merge(batch[mask], batch[perm][mask])


        mask = torch.rand(batch.shape[0]) < second
        if sum(mask) > 0:
            perm = torch.randperm(batch.shape[0])
            if random.random() < second_w_orig:
                batch[mask] = rand_merge(batch[mask], orig_batch[perm][mask], slerp=True)
            else:
                batch[mask] = rand_merge(batch[mask], batch[perm][mask], slerp=True)

        # add noise
        # noise_factors = torch.rand(batch.shape[0], 1).to(batch.device) * (noise_std / args.lora_std)
        noise_factors = torch.rand(batch.shape[0], 1).to(batch.device) * noise_std
        batch = batch + torch.randn_like(batch) * noise_factors

        pred, mean, logvar = lora_vae(batch)
        mse_loss = F.mse_loss(pred.float(), batch.float(), reduction="mean")

        kld = torch.mean(-0.5 * torch.mean(1 + logvar - mean.float().pow(2) - logvar.float().exp(), dim = 1))
        loss = mse_loss + kld * args.kld_weight

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        return mse_loss, kld, grad_norm

    # train_step = torch.compile(train_step)

    grad_norm = 0
    for i in range(args.max_train_steps):
        with accelerator.accumulate(lora_vae):
            mse_loss, kld, grad_norm = train_step()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    torch.save(unwrap_model(accelerator, lora_vae).state_dict(), save_path)

        logs = {"mse_loss": mse_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "kld": kld.detach().item(), "grad_norm": grad_norm}
        progress_bar.set_postfix(**logs)
        if args.use_wandb:
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