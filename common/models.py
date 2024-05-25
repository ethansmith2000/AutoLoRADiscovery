import torch
from torch import nn
import math
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, time_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_attn1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = Attention(hidden_size, heads=num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio) # approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn1(modulate(self.norm_attn1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x


class DiTBlockNoAda(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_attn1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(hidden_size, heads=num_heads)
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mult=mlp_ratio) # approx_gelu = lambda: nn.GELU(approximate="tanh")

    def forward(self, x):
        x = x + self.attn1(self.norm_attn1(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AttentionResampler(nn.Module):

    def __init__(self, dim, num_queries=1, heads=1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, num_queries, dim))
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.h = heads
        self.dh = dim // heads
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        b, s, dim = x.shape
        norm_x = self.norm(x)
        q = self.q.expand(b, -1, -1)
        k, v = self.kv(norm_x).chunk(2, dim=-1)
        q, k, v = map(lambda t: t.view(b, -1, self.h, self.dh).transpose(1, 2), (q, k, v))

        attn_output = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, dim).to(q.dtype)
        
        return attn_output


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, int(dim * mult), bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mult), dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, 
                dim=768, 
                 heads=8, 
                 ):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.h = heads
        self.dh = dim // heads
    
    def forward(self, x):
        b, s, dim = x.shape
        q, k, v = map(lambda t: t.view(b, -1, self.h, self.dh).transpose(1, 2), self.qkv(x).chunk(3, dim=-1))
        attn_output = F.scaled_dot_product_attention(q, k, v)

        return attn_output.transpose(1, 2).reshape(b, -1, dim).to(q.dtype)


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
        timestep = get_timestep_embedding(timestep, self.linear_1.in_features).to(self.linear_1.weight.device).to(self.linear_1.weight.dtype)
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



class Resnet(nn.Module):

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        dropout: float = 0.0,
        act = torch.nn.SiLU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(mid_dim)
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = nn.Linear(mid_dim, in_dim)
        self.act = act()


    def forward(
        self,
        hidden_states,
    ) -> torch.FloatTensor:

        resid = hidden_states

        hidden_states = self.norm1(self.act(self.linear1(hidden_states)))
        hidden_states = self.norm2(self.act(self.linear2(hidden_states)))

        return hidden_states + resid