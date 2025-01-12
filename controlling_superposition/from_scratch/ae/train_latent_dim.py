import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import os

from utils import *

# Load the MNIST dataset
train_dataset_kwargs={
    "root":"./data",
    "train":True,
    "transform":transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
    "download":True
}
train_loader_kwargs={
    "batch_size":64,
    "shuffle":True
}

OUTPUTS_PATH="controlling_superposition/from_scratch/ae/latent_dim"

def train(latent_dim,negative_slope,sample_num):
    train_loader=get_dataloader(train_dataset_kwargs,train_loader_kwargs)

    autoencoder = Autoencoder(latent_dim, negative_slope)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    pbar=tqdm(range(num_epochs))
    for epoch in pbar:
        autoencoder.train()
        running_loss = 0.0
        count = 0
        for step,(images, labels) in enumerate(train_loader):
            images = images.to(device)
            
            # Forward pass
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*len(labels)
            count += len(labels)

            if (step + 1)%10==0:
                pbar.set_description(f'{latent_dim} - {negative_slope}: Epoch [{epoch + 1}/{num_epochs}][{step + 1}/{len(train_loader)}], Loss: {running_loss/count:.4f}')
        

    # Save the trained model
    torch.save(autoencoder.state_dict(), f'{OUTPUTS_PATH}/models/{latent_dim}_{str(negative_slope).replace('.','')}_{sample_num}.pth')

sample_num=2
for latent_dim in [16,32,64]:
    for negative_slope in [0.0,0.1,0.4,0.6,0.9,1.0]:
        if os.path.exists(f"{OUTPUTS_PATH}/models/{latent_dim}_{str(negative_slope).replace('.','')}_{sample_num}.pth"):
            continue
        train(latent_dim,negative_slope,sample_num)