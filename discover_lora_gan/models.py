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



class Discriminator(nn.Module):
    def __init__(self, total_dim=1_365_504,
                        dim = 1536,
                        num_tokens=889,
                        num_layers=6,
                        num_heads=16,
                        mlp_ratio=4.0,):
        super().__init__()
        self.dim = dim
        self.proj_in = nn.Linear(dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([DiTBlockNoAda(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)])
        self.pool = AttentionResampler(dim, 1)
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, 1)

    def forward(self, x):
        x = x.reshape(x.size(0), -1, self.dim)
        x = self.proj_in(x) + self.pos_embed.expand(x.size(0), -1, -1)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(self.norm_out(self.pool(x))).squeeze(-1)
        return x



class Generator(nn.Module):
    def __init__(self, total_dim=1_365_504,
                        dim = 1536,
                        num_tokens=889,
                        num_layers=6,
                        num_heads=16,
                        mlp_ratio=4.0,
                        latent_dim=64,
                        ):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj_in = nn.Linear(latent_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([DiTBlockNoAda(dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.proj_in(x)[:,None,:].expand(-1, self.num_tokens, -1) + self.pos_embed.expand(x.size(0), -1, -1)
        for block in self.blocks:
            x = block(x)
        x = self.proj_out(self.norm_out(x)).reshape(x.size(0), -1)
        return x