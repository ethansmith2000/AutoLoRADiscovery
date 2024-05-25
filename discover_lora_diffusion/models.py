import torch
from torch import nn
import math
import torch
import torch.nn as nn
import numpy as np
import math
import sys
sys.path.append('..')

from common.models import TimestepEmbedding, AdaNorm, ChunkFanOut, Attention, FeedForward, DiTBlock, Resnet


class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int = None,
        dropout: float = 0.0,
        ada_dim: int = 512,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = AdaNorm(in_dim, ada_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = AdaNorm(mid_dim, ada_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()


    def forward(
        self,
        hidden_states,
        ada_emb=None,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states, ada_emb)))
        hidden_states = self.linear2(self.dropout(self.act(self.norm2(hidden_states, ada_emb))))

        return hidden_states + resid


class ResnetBlock(nn.Module):

    def __init__(self, num_layers=3, in_dim=256, mid_dim=256, ada_dim=512):
        super().__init__()
        self.layers = nn.ModuleList([Resnet(in_dim, mid_dim, ada_dim=ada_dim) for _ in range(num_layers)])

    def forward(self, x, ada_emb):
        for layer in self.layers:
            x = layer(x, ada_emb)
        return x


class LoraDiffusion(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, 
                    model_dim=256, 
                    ff_mult=3, 
                    chunks=1, 
                    act=torch.nn.SiLU, 
                    num_blocks=4, 
                    layers_per_block=3
                    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(model_dim//2, model_dim//2)
        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=chunks)
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=chunks)

        self.downs = nn.ModuleList([ResnetBlock(num_layers=layers_per_block, in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])
        self.ups = nn.ModuleList([ResnetBlock(num_layers=layers_per_block,  in_dim=model_dim, mid_dim=int(model_dim * ff_mult), ada_dim=model_dim//2) for _ in range(num_blocks)])


    def forward(self, x, t):
        ada_emb = self.time_embed(t)
        x = self.in_norm(x)
        x = self.in_proj(x)
        skips = []
        for down in self.downs:
            x = down(x, ada_emb)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, ada_emb) + skip
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x



class DiT(nn.Module):
    def __init__(
        self,
        total_dim=1_365_504,
        dim = 1536,
        num_tokens=889,
        time_dim = 384,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens

        self.proj_in = torch.nn.Linear(dim, dim)
        self.time_embed =  TimestepEmbedding(time_dim, time_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([DiTBlock(dim, time_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True)
        self.proj_out = nn.Linear(dim, dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        x = x.reshape(x.size(0), -1, self.dim)
        x = self.proj_in(x) + self.pos_embed.expand(x.size(0), -1, -1)
        t = self.time_embed(t)
        for block in self.blocks:
            x = block(x, t)
        x = self.norm_out(x)
        x = self.proj_out(x)
        x = x.reshape(x.size(0), -1)
        return x






#