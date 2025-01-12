import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

from utils import *

test_dataset_kwargs={
    "root":"./data",
    "train":False,
    "transform":transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
    "download":True
}
test_loader_kwargs={
    "batch_size":64,
    "shuffle":False
}

OUTPUTS_PATH="controlling_superposition/from_scratch/ae/encoder_size"

def evaluate_autoencoder_loss(model, device = 'cpu', verbose = True):
    test_loader=get_dataloader(test_dataset_kwargs,test_loader_kwargs)
    model.eval()
    total_loss = 0.0
    count = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        if verbose:
            pbar=tqdm(test_loader,desc="evaluating reconstruction loss")
        else:
            pbar=test_loader
        for images, labels in pbar:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()*len(labels)
            count += len(labels)
    
    return total_loss / count

def decode_slope_text(text):
    return float(text[0]+"."+text[1:])

def average_model_data(encoder_size):
    with open(f"{OUTPUTS_PATH}/model_data/{encoder_size[0]}_{encoder_size[1]}","rb") as file:
        model_data=pickle.load(file)

    model_data[0.0]=sum(model_data[0.0])/len(model_data[0.0])
    model_data[1.0]=sum(model_data[1.0])/len(model_data[1.0])

    return model_data

def process_model_data():
    encoder_sizes=[[int(f.split("_")[0]),int(f.split("_")[1])] for f in os.listdir(f"{OUTPUTS_PATH}/models") if ("10" not in f)]
    pbar=tqdm(encoder_sizes)
    for encoder_size in pbar:
        encoder_size_str=f"{encoder_size[0]}_{encoder_size[1]}"
        if not(os.path.exists(f"{OUTPUTS_PATH}/model_data/{encoder_size_str}")):
            model_data={0.0:[],1.0:[]}
        else:
            with open(f"{OUTPUTS_PATH}/model_data/{encoder_size_str}","rb") as file:
                model_data=pickle.load(file)
        samples=list(set([int(f.split("_")[-1].replace('.pth','')) for f in os.listdir(f"{OUTPUTS_PATH}/models") if encoder_size_str in f]))
        for sample in samples:
            if len(model_data[0.0])>=sample:
                continue
            else:
                pbar.set_description(f"{encoder_size_str} nonlinear {sample}")
                model=Autoencoder(32,0.0,encoder_size)
                model.load_state_dict(torch.load(f"{OUTPUTS_PATH}/models/{encoder_size_str}_00_{sample}.pth",weights_only=True,map_location=torch.device('cpu')))
                loss=evaluate_autoencoder_loss(model,verbose=False)
                model_data[0.0].append(loss)

                pbar.set_description(f"{encoder_size_str} linear {sample}")
                model=Autoencoder(32,1.0,encoder_size)
                model.load_state_dict(torch.load(f"{OUTPUTS_PATH}/models/{encoder_size_str}_10_{sample}.pth",weights_only=True,map_location=torch.device('cpu')))
                loss=evaluate_autoencoder_loss(model,verbose=False)
                model_data[1.0].append(loss)

        with open(f"{OUTPUTS_PATH}/model_data/{encoder_size_str}","wb") as file:
            pickle.dump(model_data,file)


def nonlinear_linear_comparison():
    encoder_sizes=[]
    for f in os.listdir(f"{OUTPUTS_PATH}/models"):
        if [int(f.split("_")[0]),int(f.split("_")[1])] in encoder_sizes:
            continue
        else:
            encoder_sizes.append([int(f.split("_")[0]),int(f.split("_")[1])])

    encoder_sizes=sorted(encoder_sizes,key=lambda item: item[0])
        
    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(4,4))
    for j,encoder_size in enumerate(encoder_sizes):
        model_data=average_model_data(encoder_size)
        ax.scatter([model_data[0.0]],[model_data[1.0]],color=plt.cm.viridis(j/len(encoder_sizes)),label=str(encoder_size))
    ax.set_xlabel("nonlinear model loss")
    ax.set_ylabel("linear model loss")
    plt.legend()
    plt.savefig(f"{OUTPUTS_PATH}/figures/nonlinear_linear_comparison.png",bbox_inches="tight")
    plt.close()

process_model_data()

nonlinear_linear_comparison()