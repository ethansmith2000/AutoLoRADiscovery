import torch
from torch import nn


import sys
sys.path.append('/home/ubuntu/AutoLoRADiscovery/')

from common.models import ChunkFanOut, DiTBlockNoAda, AttentionResampler


class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        dropout: float = 0.0,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = nn.LayerNorm(mid_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()


    def forward(
        self,
        hidden_states,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states)))
        hidden_states = self.linear2(self.dropout(self.act(self.norm2(hidden_states))))

        return hidden_states + resid



class Encoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, in_proj_chunks=1, act=torch.nn.SiLU, num_layers=6, latent_dim=None):
        super().__init__()
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        mean, logvar = self.out_proj(x).chunk(2, dim=-1)
        return mean, logvar


class Decoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, out_proj_chunks=1, act=torch.nn.SiLU, num_layers=6):
        super().__init__()
        self.in_proj = nn.Linear(model_dim, model_dim)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=out_proj_chunks)

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_proj(x)
        return x


class LoraVAE(torch.nn.Module):

    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, chunks=1, act=torch.nn.SiLU, encoder_layers=6, decoder_layers=12, latent_dim=None):
        super().__init__()
        self.encoder = Encoder(data_dim, model_dim, ff_mult, chunks, act, encoder_layers, latent_dim)
        self.decoder = Decoder(data_dim, model_dim, ff_mult, chunks, act, decoder_layers)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return self.decoder(z), mean, logvar


class VAET(nn.Module):
    def __init__(
        self,
        total_dim=1_365_504,
        dim = 1536,
        num_tokens=889,
        encoder_depth=6,
        decoder_depth=6,
        num_heads=16,
        mlp_ratio=4.0,
        num_latent_tokens=1
    ):
        super().__init__()

        self.dim = dim
        self.num_tokens = num_tokens

        self.proj_in = torch.nn.Linear(dim, dim)

        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.encoder = nn.ModuleList([DiTBlockNoAda(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(encoder_depth)])
        self.encoder_pool = AttentionResampler(dim, num_latent_tokens)
        self.encoder_norm_out = nn.LayerNorm(dim, elementwise_affine=True)
        self.encoder_proj = nn.Linear(dim, dim * 2)
        
        self.decoder_resampler = AttentionResampler(dim, num_tokens)
        self.decoder_proj_in = nn.Linear(dim, dim)
        self.decoder = nn.ModuleList([DiTBlockNoAda(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(decoder_depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True)
        self.proj_out = nn.Linear(dim, dim)

    def encode(self, x):
        x = x.reshape(x.size(0), -1, self.dim)
        x = self.proj_in(x) + self.pos_embed.expand(x.size(0), -1, -1)
        for block in self.encoder:
            x = block(x)
        x = self.encoder_pool(x)
        x = self.encoder_norm_out(x)
        mean, logvar = self.encoder_proj(x).chunk(2, dim=-1)
        return mean, logvar

    def decode(self, x):
        x = self.decoder_resampler(x)
        x = self.decoder_proj_in(x)
        for block in self.decoder:
            x = block(x)
        x = self.norm_out(x)
        x = self.proj_out(x).reshape(x.size(0), -1)
        return x

    def forward(self, x):
        """
        Forward pass of VAET.
        x: (N, C) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        """
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return self.decode(z), mean, logvar


class Discriminator(nn.Module):
    def __init__(self, total_dim=1_365_504,
                        dim = 1536,
                        num_tokens=889,
                        num_layers=6,
                        num_heads=16,
                        mlp_ratio=4.0,):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([DiTBlockNoAda(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)])
        self.pool = AttentionResampler(dim, 1)
        self.proj_out = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.dim)
        x = self.proj_in(x) + self.pos_embed.expand(x.size(0), -1, -1)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(self.pool(x)).squeeze(-1)
        return x