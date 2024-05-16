import torch
import safetensors
from .lora_resize import svd, get_least_squares_solution, change_lora_rank
import requests
import pandas as pd
import re
import glob
from pathlib import Path
import torch

def open_weights(path):
    try: # if ".safetensors" in path:
        state_dict = {}
        with safetensors.safe_open(path, "pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    except:
        state_dict = torch.load(path, map_location="cpu")

    return state_dict


def get_num_params(state_dict):
    return sum([torch.numel(v) for v in state_dict.values()])


def convert_to_multi(state_dict, idx=0):
    new_state_dict = {}
    for k,v in state_dict.items():
        # single lora method to multi
        k = k.replace(".weight", f".{idx}")
        k = k.replace("lora_weight", f"lora_weight.{idx}")
        k = k.replace("lora_bias", f"lora_bias.{idx}")

        # changing index 
        k = re.sub(r"lora_down\.(\d+)", f"lora_down.{idx}", k)
        k = re.sub(r"lora_up\.(\d+)", f"lora_up.{idx}", k)
        k = re.sub(r"lora_weight\.(\d+)", f"lora_weight.{idx}", k)
        k = re.sub(r"lora_bias\.(\d+)", f"lora_bias.{idx}", k)

        new_state_dict[k] = v
    return new_state_dict



def aggregate_loras(path, rank):
    files = glob.glob(f"{path}/**/lora_layers.pth", recursive=True)
    all_state_dicts = []
    for f in files:
        # basically runs svd on it, this ensures down/up weights are balanced
        state_dict = torch.load(f)
        state_dict = change_lora_rank(state_dict, rank=rank)
        all_state_dicts.append(state_dict)
    return all_state_dicts


    
def make_weight_vector(state_dict):
    # ensure same ordering
    keys = sorted(list(state_dict.keys()))
    weight_dict = {}
    weights = []
    idx = 0
    for k in keys:
        weight = state_dict[k]
        flattened = weight.flatten()
        interval = (idx, idx + len(flattened))
        idx = idx + len(flattened)
        weight_dict[k] = interval
        weights.append(flattened)
    
    return torch.cat(weights), weight_dict


@torch.no_grad()
def rand_merge_simple(lora_a, lora_b):
    b, n = lora_a.shape
    coef = torch.rand(b)
    out = lora_a * coef[:,None] + lora_b * (1 -coef[:,None])
    return out


@torch.no_grad()
def rand_merge_layerwise(lora_a, lora_b):
    new_state_dict = {}
    for k in lora_a.keys():
        val_a = lora_a[k]
        val_b = lora_b[k]
        coef = torch.rand(1)
        new_state_dict[k] = val_a * coef + val_b * (1-coef)
    
    return new_state_dict


# @torch.no_grad()
# def rand_merge_layerwise(lora_a, lora_b, weight_dict):
#     b, n = a.shape

#     coef = torch.rand(b, len(weight_dict))



