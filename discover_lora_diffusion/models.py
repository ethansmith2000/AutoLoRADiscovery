import torch
from torch import nn
import math


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    downscale_freq_shift: float = 1,
    max_period: int = 10000,
):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        bias=True,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=bias)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=bias)


    def forward(self, timestep):
        timestep = get_timestep_embedding(timestep, self.linear_1.in_features)
        timestep = self.linear_1(timestep)
        timestep = self.act(timestep)
        timestep = self.linear_2(timestep)
        return timestep


class AdaNorm(nn.Module):

    def __init__(self, in_dim, ada_dim):
        super().__init__()
        self.ada_proj = nn.Linear(ada_dim, 2 * in_dim)
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)

    def forward(self, hidden_states, ada_embed):
        hidden_states = self.norm(hidden_states)
        ada_embed = self.ada_proj(ada_embed)
        scale, shift = ada_embed.chunk(2, dim=1)
        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


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