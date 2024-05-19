import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_weights_global(w1, w2, scale1, scale2, global_w, num_extra_dims=1):
    expand = [None] * num_extra_dims
    w1, w2 = map(lambda x: (torch.stack(list(x)) * global_w["global_w"][:,*expand]).sum(dim=0), [w1, w2])
    return w1, w2

def get_weights_one(w1, w2, scale1, scale2, global_w, num_extra_dims=1):
    expand = [None] * num_extra_dims
    # import pdb; pdb.set_trace()
    w1, w2 = map(lambda x: (torch.stack(list(x)) * scale1[:,*expand]).sum(dim=0), [w1, w2])
    return w1, w2

def get_weights_two(w1, w2, scale1, scale2, global_w, num_extra_dims=1):
    expand = [None] * num_extra_dims
    w1, w2 = map(lambda x, scale: (torch.stack(list(x)) * scale[:,*expand]).sum(dim=0), [(w1, scale1), (w2,scale2)])
    return w1, w2

def get_weights_none(w1, w2, scale1, scale2, global_w, num_extra_dims=1):
    w1, w2 = map(lambda x: torch.cat(list(x), dim=0), [w1, w2])
    return w1, w2


class BaseLora(nn.Module):

    def __init__(self, weight_mode, num_loras=1):
        super().__init__()
        self.global_w = {}
        self.each_scale_one = None
        self.each_scale_two = None
        if num_loras == 1:
            self.get_weights = get_weights_none
        else:
            self.get_weights = get_weights_global
            if weight_mode == "one":
                self.each_scale_one = nn.Parameter(torch.ones(num_loras) / num_loras)
                self.get_weights = get_weights_one
            elif weight_mode == "two":
                self.each_scale_one = nn.Parameter(torch.ones(num_loras) / num_loras)
                self.each_scale_two = nn.Parameter(torch.ones(num_loras) / num_loras)
                self.get_weights = get_weights_two


class LoraLayerNorm(BaseLora):

    def __init__(self, normalized_shape, 
                        weight=None, 
                        bias=None, 
                        num_groups=32, 
                        eps=1e-5, 
                        num_loras=1, 
                        weight_mode="one" # ["global", "one", "two"]
                        ):
        super().__init__(weight_mode=weight_mode, num_loras=num_loras)
        self.normalized_shape = normalized_shape
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(weight) if weight is not None else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        self.lora_weight = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(num_loras)])
        self.lora_bias = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(num_loras)])

    def forward(self, x):
        weight, bias = self.get_weights(self.lora_weight, self.lora_bias, self.each_scale_one, self.each_scale_two, self.global_w, num_extra_dims=1)
        weight = self.weight + weight
        bias = self.bias + bias
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class LoraGroupNorm(LoraLayerNorm):

    def forward(self, x):
        weight, bias = self.get_weights(self.lora_weight, self.lora_bias, self.each_scale_one, self.each_scale_two, self.global_w, num_extra_dims=1)
        weight = self.weight + weight
        bias = self.bias + bias
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


class LoraConv2d(BaseLora):

    def __init__(self, base_layer, rank=32, num_loras=1, weight_mode="one"):
        super().__init__(weight_mode=weight_mode, num_loras=num_loras)
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(rank, self.base_layer.in_channels, self.base_layer.kernel_size[0], self.base_layer.kernel_size[1]) / rank) for _ in range(num_loras)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(self.base_layer.out_channels, rank, 1, 1)) for _ in range(num_loras)])
    
        self.down_kwargs = {
            "stride": self.base_layer.stride,
            "padding": self.base_layer.padding,
        }
        self.up_kwargs = {
            "stride": (1, 1),
        }


    def forward(self, x):
        orig_outs = self.base_layer(x)
        down, up = self.get_weights(self.lora_down, self.lora_up, self.each_scale_one, self.each_scale_two, self.global_w, num_extra_dims=4)
        resid = F.conv2d(F.conv2d(x, down, **self.down_kwargs), up, **self.up_kwargs)
        return orig_outs + resid


class LoraLinear(BaseLora):

    def __init__(self, base_layer, rank=32, num_loras=1, weight_mode="one"):
        super().__init__(weight_mode=weight_mode, num_loras=num_loras)
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(rank, self.base_layer.in_features) / rank) for _ in range(num_loras)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(self.base_layer.out_features, rank)) for _ in range(num_loras)])
    
    def forward(self, x):
        orig_outs = self.base_layer(x)
        down, up = self.get_weights(self.lora_down, self.lora_up, self.each_scale_one, self.each_scale_two, self.global_w, num_extra_dims=2)
        resid = F.linear(F.linear(x, down), up)
        return orig_outs + resid




def patch_lora(model, rank=32, included_terms=None, running_name=None, num_loras=1, weight_mode="one"):
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
                    model.add_module(n, LoraLinear(base_layer, rank, num_loras, weight_mode=weight_mode))
                elif isinstance(m, nn.Conv2d):
                    model.add_module(n, LoraConv2d(base_layer, rank, num_loras, weight_mode=weight_mode))
                elif isinstance(m, nn.LayerNorm):
                    model.add_module(n, LoraLayerNorm(
                        normalized_shape=base_layer.normalized_shape,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                        eps=base_layer.eps,
                        num_loras=num_loras,
                        weight_mode=weight_mode
                    ))
                elif isinstance(m, nn.GroupNorm):
                    model.add_module(n, LoraGroupNorm(
                        normalized_shape=base_layer.num_channels,
                        num_groups=base_layer.num_groups,
                        eps=base_layer.eps,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                        num_loras=num_loras,
                        weight_mode=weight_mode
                    ))


        if isinstance(m, nn.Module):
            names = [name for name, layer in m.named_modules() if name != ""]
            if len(names) > 0:
                patch_lora(m, rank=rank, included_terms=included_terms, running_name=full_name, num_loras=num_loras, weight_mode=weight_mode)


