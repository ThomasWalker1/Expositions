import torch as t
from torch import nn
import einops
from einops.layers.torch import Rearrange
import numpy as np
from typing import Tuple

# architectures

class TiedWeightsAutoencoder(nn.Module):
    def __init__(self, in_features: int, feat_dim: int, bias=True):
        super().__init__()
        self.in_features=in_features
        self.feat_dim=feat_dim
        self.bias = bias

        sf=1/np.sqrt(in_features)

        weight=sf*(2*t.rand(feat_dim,in_features)-1)
        self.weight = nn.Parameter(weight)

        if bias:
            bias=sf*(2*t.rand(feat_dim,)-1)
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: t.Tensor) -> t.Tensor:
        w=self.weight/t.max(t.abs(self.weight))
        if self.bias is not None:
            c = (einops.einsum(x, w, "b in_feats, hid_feats in_feats -> b hid_feats")+self.bias).relu()
        else:
            c = (einops.einsum(x, w, "b in_feats, hid_feats in_feats -> b hid_feats")).relu()
        x_hat=einops.einsum(w,c,"hid_feats in_feats, b hid_feats -> b in_feats")
        return x_hat,c
    
class Scale(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x/t.max(t.abs(x))
    
class Autoencoder(nn.Module):
    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(7 * 7 * 32, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, latent_dim_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime,z
    
class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, latent_dim_size*2),
            Rearrange("b (n latent_dim) -> n b latent_dim", n=2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, hidden_dim_size),
            nn.ReLU(),
            nn.Linear(hidden_dim_size, 7 * 7 * 32),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c w h", c=32, h=7, w=7),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def sample_latent_vector(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        mu,logsigma=self.encoder(x)
        sigma=t.exp(logsigma)
        z=mu+sigma*t.randn_like(mu)
        return z,mu,logsigma

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        z, mu, logsigma = self.sample_latent_vector(x)
        x_prime = self.decoder(z)
        return x_prime, mu, logsigma, z
    
class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_sizes: list):
        super().__init__()
        self.flatten = nn.Flatten()
        self.input_layer=nn.Sequential(nn.Linear(input_size,hidden_sizes[0]),nn.ReLU())
        hidden_layers_list=[]
        for k in range(len(hidden_sizes)-1):
            layer=[nn.Linear(hidden_sizes[k],hidden_sizes[k+1]),nn.ReLU()]
            hidden_layers_list.append(nn.Sequential(*layer))
        self.hidden_layers=nn.Sequential(*hidden_layers_list)
        self.output_layer=nn.Linear(hidden_sizes[-1],output_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x=self.flatten(x)
        x=self.input_layer(x)
        x=self.hidden_layers(x)
        x=self.output_layer(x)
        return x

# Loss functions

def TiedWeights_AutoencoderLoss(inputs,target,alpha=1e-3):
    return t.norm(target-inputs[0],p=2,dim=1).pow(2)+alpha*t.norm(inputs[1],p=1,dim=1)

def VAELoss(input,mu,logsigma,target,beta_kl):   
    reconstruction_loss = nn.MSELoss()(target, input)
    kl_div_loss = (0.5 * (mu ** 2 + t.exp(2 * logsigma) - 1) - logsigma).mean() * beta_kl
    return reconstruction_loss + kl_div_loss