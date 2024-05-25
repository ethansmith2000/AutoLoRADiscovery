import torch
from torch import nn


import sys
sys.path.append('..')

from common.models import ChunkFanOut, DiTBlockNoAda, AttentionResampler, Resnet


class Encoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, in_proj_chunks=1, act=torch.nn.SiLU, num_layers=6, latent_dim=None):
        super().__init__()
        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
        mean, logvar = self.out_proj(x).chunk(2, dim=-1)
        return mean, logvar


class Decoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, out_proj_chunks=1, act=torch.nn.SiLU, num_layers=6):
        super().__init__()
        self.in_proj = nn.Linear(model_dim, model_dim)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=out_proj_chunks)
        self.out_norm = nn.LayerNorm(model_dim)
        # self.out_norm_2 = nn.LayerNorm(data_dim) # this is a nice way to get full size parameters while still fairly cheap
        

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
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


class Discriminator(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, 
                        model_dim=256, 
                        ff_mult=3, 
                        in_proj_chunks=1, 
                        act=torch.nn.SiLU, 
                        num_layers=6, 
                        ):
        super().__init__()
        self.in_norm = nn.LayerNorm(data_dim)
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, int(model_dim * ff_mult), act=act) for _ in range(num_layers)])
        latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_norm(x)
        return self.out_proj(x)