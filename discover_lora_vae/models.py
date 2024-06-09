import torch
from torch import nn


import sys
sys.path.append('..')

from common.models import ChunkFanOut, DiTBlockNoAda, AttentionResampler, Resnet
import torch.nn.functional as F

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



class SparseAE(nn.Module):
    def __init__(self, expansion_factor=6, l0_alpha=3e-3):
        super(SparseAE, self).__init__()
        latent_dim = 512 
        self.l0_alpha = l0_alpha
        expanded_dim = round(latent_dim * expansion_factor)

        self.dictionary_size = expanded_dim
        self.up = nn.Linear(latent_dim, expanded_dim, bias=False)
        self.down = nn.Linear(expanded_dim, latent_dim, bias=False)
        self.apply(self._init_weights)

    def initiate_vae(self,vae_weights_path,  **vae_params): # initiate the base VAE, make sure same weight is used during training and inference of this SAE
        self.vae = LoraVAE(**vae_params)
        self.vae.load_state_dict(torch.load(vae_weights_path))
        self.freeze_vae()
       

    def normalize_dictionary(self):
        self.down.weight.data = F.normalize(self.down.weight.data, p=2, dim=0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module is self.down:
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu') 
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        with torch.no_grad():
            mu,logvar = self.vae.encoder(x)

        dense_dictionary = self.up(mu)
        dictionary = F.relu(dense_dictionary)
        self.normalize_dictionary()
        
        recon_x = self.down(dictionary)

        eps = 1e-3
        
        l0_loss = ((F.sigmoid((dense_dictionary - eps) * 1000.0) ).sum() / self.dictionary_size) * self.l0_alpha  # approximated l0 loss

            
        return recon_x, mu, dictionary, l0_loss





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