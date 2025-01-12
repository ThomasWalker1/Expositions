import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Autoencoder class with configurable latent space dimension
class Autoencoder(nn.Module):
    def __init__(self, latent_dim, negative_slope=0.0, encoder_size=[128,64]):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, encoder_size[0]),
            nn.ReLU(True),
            nn.Linear(encoder_size[0], encoder_size[1]),
            nn.ReLU(True),
            nn.Linear(encoder_size[1], latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Unflatten(1, (1, 28, 28))
        )

        self.negative_slope=negative_slope
        
    def forward(self, x):
        latent = self.encoder(x)
        latent = F.relu(latent)-self.negative_slope*F.relu(-latent)
        reconstructed = self.decoder(latent)
        return reconstructed

def get_dataloader(dataset_kwargs,dataloader_kwargs):
    dataset = torchvision.datasets.MNIST(**dataset_kwargs)
    loader = DataLoader(dataset,**dataloader_kwargs)
    return loader