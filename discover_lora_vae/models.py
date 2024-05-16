import torch
from torch import nn


class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        dropout: float = 0.0,
        # ada_dim: int = 512,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        # self.ada_proj = nn.Linear(temb_channels, 2 * out_channels)
        self.norm2 = nn.LayerNorm(mid_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()

    def ada_shift(self, hidden_states, ada_embed):
        ada_embed = self.act(ada_embed)
        ada_embed = self.ada_proj(ada_embed)
        return hidden_states + ada_embed

    def ada_scale_shift(self, hidden_states, ada_embed):
        ada_embed = self.act(ada_embed)
        scale, shift = self.ada_proj(ada_embed).chunk(ada_embed, 2, dim=1)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


    def forward(
        self,
        hidden_states,
        ada_embed=None,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.linear1(self.act(self.norm1(hidden_states)))
        hidden_states = self.linear2(self.dropout(self.act(self.norm2(hidden_states))))

        return hidden_states + resid


class ChunkFanIn(torch.nn.Module):

    def __init__(self, in_dim, out_dim, chunks=1):
        super().__init__()
        assert in_dim % chunks == 0
        self.projs = nn.ModuleList([nn.Linear(in_dim // chunks, out_dim) for _ in range(chunks)])
        self.in_dim = in_dim
        self.chunk_dim = in_dim // chunks

    def forward(self, x):
        return torch.stack([proj(x[..., (i * self.chunk_dim) : ((i+1) * self.chunk_dim)]) for i, proj in enumerate(self.projs)], dim=1).sum(dim=1)


class ChunkFanOut(torch.nn.Module):

    def __init__(self, in_dim, out_dim, chunks=1):
        super().__init__()
        assert out_dim % chunks == 0
        self.projs = nn.ModuleList([nn.Linear(in_dim, out_dim // chunks) for _ in range(chunks)])

    def forward(self, x):
        return torch.cat([proj(x) for proj in self.projs], dim=1)


class Encoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, ff_mult=3, in_proj_chunks=1, act=torch.nn.SiLU, num_layers=6, latent_dim=None):
        super().__init__()
        self.in_proj = ChunkFanOut(data_dim, model_dim, chunks=in_proj_chunks)
        self.resnets = nn.ModuleList([Resnet(model_dim, model_dim * ff_mult, act=act) for _ in range(num_layers)])
        latent_dim = model_dim * 2 if latent_dim is None else model_dim
        self.out_proj = nn.Linear(model_dim, latent_dim)

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        mean, logvar = self.out_proj(x).chunk(2, dim=-1)
        return mean, logvar


class Decoder(torch.nn.Module):
    def __init__(self, data_dim=1_365_504, model_dim=256, out_proj_chunks=1, ff_mult=3, act=torch.nn.SiLU, num_layers=6):
        super().__init__()
        self.in_proj = nn.Linear(model_dim, model_dim)
        self.resnets = nn.ModuleList([Resnet(model_dim, model_dim * ff_mult, act=act) for _ in range(num_layers)])
        self.out_proj = ChunkFanOut(model_dim, data_dim, chunks=out_proj_chunks)

    def forward(self, x):
        x = self.in_proj(x)
        for resnet in self.resnets:
            x = resnet(x)
        x = self.out_proj(x)
        return x