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

OUTPUTS_PATH="controlling_superposition/from_scratch/ae/latent_dim"

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

def process_model_data(latent_dims):
    for latent_dim in latent_dims:
        if not(os.path.exists(f"{OUTPUTS_PATH}/model_data/{latent_dim}")):
            model_data={0.0:{"losses":[],"loss":[]},0.1:{"losses":[],"loss":[]},0.4:{"losses":[],"loss":[]},0.6:{"losses":[],"loss":[]},0.9:{"losses":[],"loss":[]},1.0:{"losses":[],"loss":[]}}
        else:
            with open(f"{OUTPUTS_PATH}/model_data/{latent_dim}","rb") as file:
                model_data=pickle.load(file)
        slopes=sorted(model_data.keys())
        pbar=tqdm(slopes)
        for slope in pbar:
            samples=[int(f.split("_")[-1].replace('.pth','')) for f in os.listdir(f"{OUTPUTS_PATH}/models") if f"{latent_dim}_{str(slope).replace('.','')}" in f]
            for sample in samples:
                if len(model_data[slope]["losses"])>=sample:
                    continue
                else:
                    model_data[slope]["losses"].append([])
                    for testing_slope in slopes:
                        pbar.set_description(f"Latent dim {latent_dim} base slope {slope} sample {sample} testing slope {testing_slope}")
                        model=Autoencoder(latent_dim,testing_slope)
                        model.load_state_dict(torch.load(f"{OUTPUTS_PATH}/models/{latent_dim}_{str(slope).replace('.','')}_{sample}.pth",weights_only=True,map_location=torch.device('cpu')))
                        loss=evaluate_autoencoder_loss(model,verbose=False)
                        model_data[slope]["losses"][-1].append(loss)
                        if slope==testing_slope:
                            model_data[slope]["loss"].append(loss)

        with open(f"{OUTPUTS_PATH}/model_data/{latent_dim}","wb") as file:
            pickle.dump(model_data,file)

def average_model_data(latent_dim):
    with open(f"{OUTPUTS_PATH}/model_data/{latent_dim}","rb") as file:
        model_data=pickle.load(file)

    for k,v in model_data.items():
        num_samples=len(v["losses"])
        num_slopes=len(v["losses"][0])
        model_data[k]["losses"]=[sum([v["losses"][i][j] for i in range(num_samples)])/num_samples for j in range(num_slopes)]
        model_data[k]["loss"]=sum(v["loss"])/num_samples
    return model_data

def fixed_latent_dim(latent_dims):
    fig,axs=plt.subplots(nrows=len(latent_dims),ncols=3,figsize=(3*4,len(latent_dims)*4),layout="tight")
    for j,latent_dim in enumerate(latent_dims):
        model_data=average_model_data(latent_dim)
        slopes=sorted(model_data.keys())

        
        tops=[None,2]
        for k in range(len(tops)):
            if tops[k] is not None:
                axs[j][k].set_ylim(top=tops[k])
            for slope in slopes:
                axs[j][k].plot(slopes,model_data[slope]["losses"],color=plt.cm.viridis(slope),label=str(slope))
                axs[j][k].scatter([slope],[model_data[slope]["loss"]],c="black")
                axs[j][k].set_ylabel("loss")
            axs[j][k].legend()
        axs[j][1].set_title(str(latent_dim))
        for slope in slopes:
            axs[j][-1].plot(slopes,[e/max(model_data[slope]["losses"]) for e in model_data[slope]["losses"]],color=plt.cm.viridis(slope),label=str(slope))
            axs[j][-1].scatter([slope],[model_data[slope]["loss"]/max(model_data[slope]["losses"])],c="black")
        axs[j][-1].set_ylabel("normalised loss")
        axs[j][-1].legend()
        if latent_dim==max(latent_dims):
            for k in range(3):
                axs[j][k].set_xlabel("slope")
    plt.savefig(f"{OUTPUTS_PATH}/figures/fixed_latent_dim.png",bbox_inches="tight")
    plt.close()

def across_nonlinear_to_linear(latent_dims):
    complete_data={}
    for latent_dim in latent_dims:
        model_data=average_model_data(latent_dim)
        slopes=sorted(model_data.keys())
        slopes=sorted(model_data.keys())

        complete_data[latent_dim]=model_data
    
    fig,axs=plt.subplots(nrows=1,ncols=len(slopes),figsize=(len(slopes)*4,4),layout="tight")
    for latent_dim in latent_dims:
        
        for k,plotting_slope in enumerate(slopes):
            axs[k].set_title(str(plotting_slope))
            axs[k].plot(slopes,complete_data[latent_dim][plotting_slope]["losses"],color=plt.cm.viridis(latent_dim/max(latent_dims)),label=str(latent_dim))
            axs[k].scatter([plotting_slope],[complete_data[latent_dim][plotting_slope]["loss"]],color="black",s=5,zorder=100)
            axs[k].legend()
    for k in range(len(slopes)):
        axs[k].set_xlabel("slope")
    axs[0].set_ylabel("loss")
    plt.savefig(f"{OUTPUTS_PATH}/figures/nonlinear_to_linear.png",bbox_inches="tight")

    fig,axs=plt.subplots(nrows=1,ncols=len(slopes),figsize=(len(slopes)*4,4),layout="tight")
    for latent_dim in latent_dims:
        
        for k,plotting_slope in enumerate(slopes):
            axs[k].set_title(str(plotting_slope))
            axs[k].plot(slopes,[e/max(complete_data[latent_dim][plotting_slope]["losses"]) for e in complete_data[latent_dim][plotting_slope]["losses"]],color=plt.cm.viridis(latent_dim/max(latent_dims)),label=str(latent_dim))
            axs[k].scatter([plotting_slope],[complete_data[latent_dim][plotting_slope]["loss"]/max(complete_data[latent_dim][plotting_slope]["losses"])],color="black",s=5,zorder=100)
            axs[k].legend()
    for k in range(len(slopes)):
        axs[k].set_xlabel("slope")
    axs[0].set_ylabel("normalised loss")
    plt.savefig(f"{OUTPUTS_PATH}/figures/nonlinear_to_linear_normed.png",bbox_inches="tight")

fixed_latent_dim([8,16,32,64])
across_nonlinear_to_linear([8,16,32,64])