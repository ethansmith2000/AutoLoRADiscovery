import torch
from torch import nn
import math
import torch
import torch.nn as nn
import numpy as np
import math
import sys
sys.path.append('/home/ubuntu/AutoLoRADiscovery/')

from common.models import TimestepEmbedding, AdaNorm, ChunkFanOut, Attention, FeedForward, DiTBlock

class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        dropout: float = 0.0,
        ada_dim: int = 512,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        # self.norm1 = nn.LayerNorm(in_dim)
        self.norm1 = AdaNorm(in_dim, ada_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        # self.norm2 = nn.LayerNorm(mid_dim)
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



class Encoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, in_proj_chunks=1, act=torch.nn.SiLU, num_layers=6):
        super().__init__()
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, model_dim * ff_mult, act=act, ada_dim=model_dim//2) for _ in range(num_layers)])
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x, ada_emb):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x, ada_emb=ada_emb)
        return self.out_proj(x)


class Decoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, out_proj_chunks=1, act=torch.nn.SiLU, num_layers=6):
        super().__init__()
        self.in_proj = nn.Linear(model_dim, model_dim)
        self.resnets = nn.ModuleList([Resnet(model_dim, model_dim * ff_mult, act=act, ada_dim=model_dim//2) for _ in range(num_layers)])
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=out_proj_chunks)

    def forward(self, x, ada_emb):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x, ada_emb=ada_emb)
        x = self.out_proj(x)
        return x


class LoraDiffusion(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, chunks=1, act=torch.nn.SiLU, encoder_layers=6, decoder_layers=12):
        super().__init__()
        self.time_embed = TimestepEmbedding(model_dim//2, model_dim//2)
        self.encoder = Encoder(data_dim, model_dim, ff_mult, chunks, act, encoder_layers)
        self.decoder = Decoder(data_dim, model_dim, ff_mult, chunks, act, decoder_layers)

    def forward(self, x, t):
        ada_emb = self.time_embed(t)
        return self.decoder(self.encoder(x, ada_emb), ada_emb)




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