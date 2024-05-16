import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F


# class LoraLayerNorm(nn.Module):

#     def __init__(self, normalized_shape, weight=None, bias=None, num_groups=32, eps=1e-5):
#         super().__init__()
#         self.normalized_shape = normalized_shape
#         self.num_groups = num_groups
#         self.eps = eps
#         self.weight = torch.nn.Parameter(weight) if weight is not None else None
#         self.bias = torch.nn.Parameter(bias) if bias is not None else None

#         self.lora_weight = torch.nn.Parameter(torch.zeros(normalized_shape))
#         self.lora_bias = torch.nn.Parameter(torch.zeros(normalized_shape))

#     def get_weight(self):
#         return self.weight + self.lora_weight
    
#     def get_bias(self):
#         return self.bias + self.lora_bias

#     def forward(self, x):
#         return F.layer_norm(x, self.normalized_shape, self.get_weight(), self.get_bias(), self.eps)


# class LoraGroupNorm(LoraLayerNorm):

#     def forward(self, x):
#         return F.group_norm(x, self.num_groups, self.get_weight(), self.get_bias(), self.eps)



# class LoraLinear(torch.nn.Module):

#     def __init__(self, base_layer, rank=32):
#         super().__init__()
#         self.base_layer = base_layer
#         self.lora_down = nn.Linear(self.base_layer.in_features, rank, bias=False)
#         self.lora_up = nn.Linear(rank, self.base_layer.out_features, bias=False)

#         torch.nn.init.normal_(self.lora_down.weight, std=1 / rank)
#         torch.nn.init.zeros_(self.lora_up.weight)
    
#     def forward(self, x):
#         orig_outs = self.base_layer(x)
#         resid = self.lora_up(self.lora_down(x))
#         return orig_outs + resid



# class LoraConv2d(torch.nn.Module):

#     def __init__(self, base_layer, rank=32):
#         super(LoraConv2d, self).__init__()
#         self.base_layer = base_layer
#         self.lora_down = nn.Conv2d(self.base_layer.in_channels, rank, kernel_size=self.base_layer.kernel_size, stride=self.base_layer.stride, padding=self.base_layer.padding, bias=False)
#         self.lora_up = nn.Conv2d(rank, self.base_layer.out_channels, kernel_size=(1,1), stride=(1,1), bias=False)
#         torch.nn.init.normal_(self.lora_down.weight)
#         torch.nn.init.zeros_(self.lora_up.weight)
    
#     def forward(self, x):
#         orig_outs = self.base_layer(x)
#         resid = self.lora_up(self.lora_down(x))
#         return orig_outs + resid




class LoraLayerNorm(nn.Module):

    def __init__(self, normalized_shape, weight=None, bias=None, num_groups=32, eps=1e-5, num_loras=1):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(weight) if weight is not None else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

        self.lora_weight = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(num_loras)])
        self.lora_bias = nn.ParameterList([torch.nn.Parameter(torch.zeros(normalized_shape)) for _ in range(num_loras)])
        
        self.each_scale_weight = nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(num_loras)])
        self.each_scale_bias = nn.ParameterList([torch.nn.Parameter(torch.ones(1)) for _ in range(num_loras)])

        self.normalize = torch.nn.functional.softmax if num_loras > 1 else lambda x, dim: x

    def get_weight_bias(self):
        weight, bias, scale_weight, scale_bias = map(lambda x: torch.stack(list(x)), [self.lora_weight, self.lora_bias, self.each_scale_weight, self.each_scale_bias])

        scale_weight = self.normalize(scale_weight, dim=0)
        scale_bias = self.normalize(scale_bias, dim=0)
        weight = (weight * scale_weight).sum(dim=0)
        bias = (bias * scale_bias).sum(dim=0)

        weight = self.weight + weight
        bias = self.bias + bias
        return weight, bias

    def forward(self, x):
        weight, bias = self.get_weight_bias()
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


class LoraGroupNorm(LoraLayerNorm):

    def forward(self, x):
        weight, bias = self.get_weight_bias()
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


class LoraConv2d(torch.nn.Module):

    def __init__(self, base_layer, rank=32, num_loras=1):
        super(LoraConv2d, self).__init__()
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(self.base_layer.in_channels, rank, self.base_layer.kernel_size[0], self.base_layer.kernel_size[1]) / rank) for _ in range(num_loras)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(rank, self.base_layer.out_channels, 1, 1)) for _ in range(num_loras)])
        self.each_scale_down = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_loras)])
        self.each_scale_up = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_loras)])

        self.normalize = torch.nn.functional.softmax if num_loras > 1 else lambda x, dim: x
    
        self.down_kwargs = {
            "stride": self.base_layer.stride,
            "padding": self.base_layer.padding,
        }
        self.up_kwargs = {
            "stride": (1, 1),
        }

    def get_weight(self):
        down, up, scale_down, scale_up = map(lambda x: torch.stack(list(x)), [self.lora_down, self.lora_up, self.each_scale_down, self.each_scale_up])
        scale_down = self.normalize(scale_down, dim=0)
        scale_up = self.normalize(scale_up, dim=0)
        down = (down * scale_down).sum(dim=0)
        up = (up * scale_up).sum(dim=0)
        return down, up

    def forward(self, x):
        orig_outs = self.base_layer(x)
        down, up = self.get_weight()
        resid = F.conv2d(F.conv2d(x, down, **self.down_kwargs), up, **self.up_kwargs)
        return orig_outs + resid


class LoraLinear(LoraConv2d):

    def __init__(self, base_layer, rank=32, num_loras=1):
        super().__init__(base_layer, rank=32, num_loras=1)
        self.base_layer = base_layer
        self.lora_down = nn.ParameterList([nn.Parameter(torch.randn(self.base_layer.in_features, rank) / rank) for _ in range(num_loras)])
        self.lora_up = nn.ParameterList([nn.Parameter(torch.zeros(rank, self.base_layer.out_features)) for _ in range(num_loras)])
        self.each_scale_down = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_loras)])
        self.each_scale_up = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(num_loras)])

        self.normalize = torch.nn.functional.softmax if num_loras > 1 else lambda x, dim: x
    
    def forward(self, x):
        down, up = self.get_weight()
        orig_outs = self.base_layer(x)
        resid = F.linear(F.linear(x, down), up)
        return orig_outs + resid




def patch_lora(model, rank=32, included_terms=None, running_name=None, num_loras=1):
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
                    model.add_module(n, LoraLinear(base_layer, rank, num_loras))
                elif isinstance(m, nn.Conv2d):
                    model.add_module(n, LoraConv2d(base_layer, rank, num_loras))
                elif isinstance(m, nn.LayerNorm):
                    model.add_module(n, LoraLayerNorm(
                        normalized_shape=base_layer.normalized_shape,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                        eps=base_layer.eps,
                        num_loras=num_loras
                    ))
                elif isinstance(m, nn.GroupNorm):
                    model.add_module(n, LoraGroupNorm(
                        normalized_shape=base_layer.num_channels,
                        num_groups=base_layer.num_groups,
                        eps=base_layer.eps,
                        weight=base_layer.weight,
                        bias=base_layer.bias,
                        num_loras=num_loras
                    ))


        if isinstance(m, nn.Module):
            names = [name for name, layer in m.named_modules() if name != ""]
            if len(names) > 0:
                patch_lora(m, rank=rank, included_terms=included_terms, running_name=full_name)


