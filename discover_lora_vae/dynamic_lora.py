import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLayerNorm(nn.Module):

    def __init__(self, normalized_shape, 
                        weight=None, 
                        bias=None, 
                        num_groups=32, 
                        eps=1e-5, 
                        ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(weight) if weight is not None else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        # only used for param name indexing
        self.lora_weight = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(1)])
        self.lora_bias = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(1)])

        self.weight_name = None
        self.bias_name = None
        self.weight_dict = {}

    def forward(self, x):
        lora_weight = self.weight_dict[self.weight_name]
        lora_bias = self.weight_dict[self.bias_name]
        weight = self.weight + lora_weight
        bias = self.bias + lora_bias
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class LoraGroupNorm(LoraLayerNorm):

    def forward(self, x):
        lora_weight = self.weight_dict[self.weight_name]
        lora_bias = self.weight_dict[self.bias_name]
        weight = self.weight + lora_weight
        bias = self.bias + lora_bias
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


class LoraConv2d(nn.Module):

    def __init__(self, base_layer, rank=32):
        super().__init__()
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(rank, self.base_layer.in_channels, self.base_layer.kernel_size[0], self.base_layer.kernel_size[1]) / rank) for _ in range(1)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(self.base_layer.out_channels, rank, 1, 1)) for _ in range(1)])
    
        self.down_kwargs = {
            "stride": self.base_layer.stride,
            "padding": self.base_layer.padding,
        }
        self.up_kwargs = {
            "stride": (1, 1),
        }

        self.down_name = None
        self.up_name = None
        self.weight_dict = {}


    def forward(self, x):
        orig_outs = self.base_layer(x)
        lora_down = self.weight_dict[self.down_name]
        lora_up = self.weight_dict[self.up_name]
        resid = F.conv2d(F.conv2d(x, lora_down, **self.down_kwargs), lora_up, **self.up_kwargs)
        return orig_outs + resid


class LoraLinear(nn.Module):

    def __init__(self, base_layer, rank=32):
        super().__init__()
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(rank, self.base_layer.in_features) / rank) for _ in range(1)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(self.base_layer.out_features, rank)) for _ in range(1)])
    
        self.down_name = None
        self.up_name = None
        self.weight_dict = {}

    def forward(self, x):
        orig_outs = self.base_layer(x)
        lora_down = self.weight_dict[self.down_name]
        lora_up = self.weight_dict[self.up_name]
        resid = F.linear(F.linear(x, lora_down), lora_up)
        return orig_outs + resid




def patch_lora(model, rank=32, included_terms=None, running_name=None):
    for n in list(model._modules.keys()):
        # only one depth down, and skip self
        if "." in n or n=="":
            continue

        m = model._modules[n]
        full_name = n if running_name is None else running_name + "." + n
        condition = " or ".join([f'"{term}" in full_name' for term in included_terms])
        if eval(condition):
            if any([isinstance(m, t) for t in [nn.Linear, nn.Conv2d, nn.LayerNorm, nn.GroupNorm]]):
                base_layer = deepcopy(m)
                delattr(model, n)

                if isinstance(m, nn.Linear):
                    model.add_module(n, LoraLinear(base_layer, rank))
                    layer = getattr(model, n)
                    setattr(layer, "down_name", f"{full_name}.lora_down.0")
                    setattr(layer, "up_name", f"{full_name}.lora_up.0")
                elif isinstance(m, nn.Conv2d):
                    model.add_module(n, LoraConv2d(base_layer, rank))
                    layer = getattr(model, n)
                    setattr(layer, "down_name", f"{full_name}.lora_down.0")
                    setattr(layer, "up_name", f"{full_name}.lora_up.0")
                elif isinstance(m, nn.LayerNorm):
                    model.add_module(n, LoraLayerNorm(
                        normalized_shape=base_layer.normalized_shape,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                        eps=base_layer.eps,
                    ))
                    layer = getattr(model, n)
                    setattr(layer, "weight_name", f"{full_name}.lora_weight.0")
                    setattr(layer, "bias_name", f"{full_name}.lora_bias.0")
                elif isinstance(m, nn.GroupNorm):
                    model.add_module(n, LoraGroupNorm(
                        normalized_shape=base_layer.num_channels,
                        num_groups=base_layer.num_groups,
                        eps=base_layer.eps,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                    ))
                    layer = getattr(model, n)
                    setattr(layer, "weight_name", f"{full_name}.lora_weight.0")
                    setattr(layer, "bias_name", f"{full_name}.lora_bias.0")


        if isinstance(m, nn.Module):
            names = [name for name, layer in m.named_modules() if name != ""]
            if len(names) > 0:
                patch_lora(m, rank=rank, included_terms=included_terms, running_name=full_name)


def give_weights(model, state_dict):
    for n, m in model.named_modules():
        if hasattr(m, "weight_dict"):
            m.weight_dict = state_dict