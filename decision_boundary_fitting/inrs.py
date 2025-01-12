import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from scipy.spatial import cKDTree

import sys
sys.path.append('../splinetheory')

import splinecam as sc

def filename_to_float(filename):
    filename=str(filename).split('_')[0].split('.')[0]
    return float(filename[0]+'.'+filename[1:])

def float_to_filename(flt,sample_num=None):
    if sample_num is not None:
        return str(flt).replace('.','')+f"_{sample_num}"
    else:
        return str(flt).replace('.','')+"_"

def chamfer_distance(set1, set2):
    tree1 = cKDTree(set1)
    tree2 = cKDTree(set2)

    distances1, _ = tree1.query(set2, k=1)
    distances2, _ = tree2.query(set1, k=1)

    chamfer_dist = np.mean(distances1**2) + np.mean(distances2**2)
    return chamfer_dist

def interior_point(coord,img):
    height,width=img.size()
    if (coord[0]==height-1) or (coord[1]==width-1) or coord[0]==0 or coord[1]==0:
        return True
    if torch.sum(img[coord[0]-1:coord[0]+2,coord[1]-1:coord[1]+2]).item()==9 or torch.sum(img[coord[0]-1:coord[0]+2,coord[1]-1:coord[1]+2]).item()==0:
        return True
    else:
        return False

def patch_img_from_samples(img,samples):
    img=torch.tensor(img)
    original_img=img.clone()
    all_coords=[[x,y] for x in range(img.shape[0]) for y in range(img.shape[1])]
    for sample in samples:
        coord=all_coords[sample]
        if interior_point(coord,original_img):
            img[coord[0],coord[1]]=0.5
    return img

class PixelDataset(Dataset):
    def __init__(self, img, patch_rate=0.01):
        height,width=img.shape
        self.height=height
        self.width=width
        coordinates=np.array([[i/height,j/width] for i in range(height) for j in range(width)])
        values=img.flatten()

        self.coordinates=torch.tensor(coordinates, dtype=torch.float32)
        self.values=torch.tensor(values, dtype=torch.float32)

        self.samples=np.random.choice(np.arange(len(values)),replace=False,size=int(len(values)*patch_rate))

        patched_img=patch_img_from_samples(img,self.samples)

        sampled_coordinates=np.array([[i/height,j/width] for i in range(height) for j in range(width) if patched_img[i,j]!=0.5])
        sampled_values=np.array([i for i in patched_img.flatten() if i!=0.5])
        self.sampled_coordinates=sampled_coordinates
        self.sampled_values=sampled_values
    
    def __len__(self):
        return len(self.sampled_values)
    
    def __getitem__(self, idx):
        coord = self.sampled_coordinates[idx]
        value = self.sampled_values[idx]
        return coord, value

def get_img_boundary(img):
    height,width=img.shape
    coordinates=np.array([[i,j] for i in range(height) for j in range(width)])
    boundary_coordinates=[np.array([coord[0]/height,coord[1]/width]) for coord in coordinates if not(interior_point(coord,img))]
    return np.array(boundary_coordinates)

img=1-plt.imread('decision_boundary_fitting/true_boundary.png')[:,:,0]
mask=(img>0.0)
img=np.zeros_like(img)
img[mask]=1
img=torch.tensor(img)
boundary_coordinates=get_img_boundary(img)

def get_sample_num(output_dir,patch_rate):
    num_samples=sum([1 for filename in os.listdir(f"{output_dir}/boundaries/") if float_to_filename(patch_rate) in filename])
    return num_samples+1

total_c_distances={}
max_samples=5
num_samples=3
patch_rates=[0.8,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.0]
for NUM_HIDDEN_DIMS in [4,8,16,32,64,96,128]:

    OUTPUT_DIRS=f"./decision_boundary_fitting/{NUM_HIDDEN_DIMS}"

    os.makedirs(OUTPUT_DIRS,exist_ok=True)
    os.makedirs(OUTPUT_DIRS+"/boundaries",exist_ok=True)
    os.makedirs(OUTPUT_DIRS+"/patched_imgs",exist_ok=True)

    ax=plt.subplot()
    ax.scatter(boundary_coordinates[:,0],boundary_coordinates[:,1],s=1)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    plt.savefig(f"{OUTPUT_DIRS}/boundary.png")
    plt.close()

    for _ in range(num_samples):
        for patch_rate in patch_rates:
            
            sample_num=get_sample_num(OUTPUT_DIRS,patch_rate)
            if sample_num>max_samples:
                continue
            dataset=PixelDataset(img,patch_rate=patch_rate)
            
            loader=DataLoader(dataset,batch_size=128,shuffle=True)

            patched_img=patch_img_from_samples(img,dataset.samples)
            torch.save(patched_img,f"{OUTPUT_DIRS}/patched_imgs/{float_to_filename(patch_rate,sample_num)}.pt")

            width = NUM_HIDDEN_DIMS
            depth = 4
            layers = [nn.Linear(2,width),nn.ReLU()]
            for i in range(depth-1):
                layers+=[nn.Linear(width,width),nn.ReLU()]
            layers.append(nn.Linear(width,1))

            model = torch.nn.Sequential(*layers)
            optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
            criterion = torch.nn.BCEWithLogitsLoss()

            epochs = 50
            model.type(torch.float64)
            pbar=tqdm(range(1, epochs + 1),desc=f"{NUM_HIDDEN_DIMS} - {patch_rate} - {sample_num}")
            for epoch in pbar:
                model.train()
                for batch_idx, (inputs,labels) in enumerate(loader, 1):
                    optimizer.zero_grad()
                    output = model(inputs)

                    loss = criterion(output.squeeze(1), labels)
                    loss.backward()
                    optimizer.step()

            domain = torch.tensor([[0,0],[0,1],[1,1],[1,0],[0,0]])
            T = torch.eye(3)[:-1]
            model.eval()
            model.type(torch.float64)

            NN = sc.wrappers.model_wrapper(model,input_shape=(2,),T = T,dtype = torch.float64)
            regions, endpoints, Abw = sc.compute.get_partitions_with_db(domain,T,NN)

            torch.save(endpoints,f"{OUTPUT_DIRS}/boundaries/{float_to_filename(patch_rate,sample_num)}.pt")

    sampled_rates=sorted(list(set([filename_to_float(filename) for filename in os.listdir(f"{OUTPUT_DIRS}/boundaries")])))

    fig,axs=plt.subplots(nrows=2,ncols=len(sampled_rates),figsize=(3*len(sampled_rates),6))
    for k,sampled_rate in enumerate(sampled_rates):
        patched_img=torch.load(f"{OUTPUT_DIRS}/patched_imgs/{float_to_filename(sampled_rate,1)}.pt")
        patched_img=np.rot90(patched_img.detach().numpy())
        axs[0,k].imshow(patched_img,cmap=ListedColormap([plt.cm.winter(0.25),'white',plt.cm.winter(0.75)]),interpolation='nearest')
        axs[0,k].axis("off")
        axs[1,k].imshow(np.rot90(img),cmap=ListedColormap([plt.cm.winter(0.25),plt.cm.winter(0.75)]),extent=(0,1,0,1))
        axs[1,k].axis("off")
        nn_decision_boundary=[]
        boundary=torch.load(f"{OUTPUT_DIRS}/boundaries/{float_to_filename(sampled_rate,1)}.pt")
        for each in boundary:
            if each is not None:
                axs[1,k].plot(each[:,0],each[:,1],c='r',zorder=1000000000,linewidth=1)
                length=np.sqrt((each[0,0]-each[1,0])**2+(each[0,1]-each[1,1])**2)
                num_points=max(int(length/0.0001),2)
                xs=np.linspace(min(each[0,0],each[1,0]),max(each[0,0],each[1,0]),num_points)
                ys=np.linspace(min(each[0,1],each[1,1]),max(each[0,1],each[1,1]),num_points)
                nn_decision_boundary.extend(np.stack((xs,ys),axis=1))
        if nn_decision_boundary!=[]:
            nn_decision_boundary=np.array(nn_decision_boundary)
            c_distance=chamfer_distance(nn_decision_boundary,boundary_coordinates)
        else:
            c_distance=0
        axs[0,k].set_title(f"{sampled_rate} - {c_distance:.4f}")
    plt.savefig(f"{OUTPUT_DIRS}/images.png")
    plt.close()

    num_sampled=5
    total_c_distances[NUM_HIDDEN_DIMS]={r:[] for r in sampled_rates}
    for sampled_rate in sampled_rates:
        num_sampled=sum([1 for filename in os.listdir(f"{OUTPUT_DIRS}/boundaries") if float_to_filename(sampled_rate) in filename])
        for k in range(1,num_sampled+1):
            nn_decision_boundary=[]
            boundary=torch.load(f"{OUTPUT_DIRS}/boundaries/{float_to_filename(sampled_rate,k)}.pt")

            for each in boundary:
                if each is not None:
                    axs[1,k].plot(each[:,0],each[:,1],c='r',zorder=1000000000,linewidth=1)

                    length=np.sqrt((each[0,0]-each[1,0])**2+(each[0,1]-each[1,1])**2)
                    num_points=max(int(length/0.0001),2)
                    xs=np.linspace(min(each[0,0],each[1,0]),max(each[0,0],each[1,0]),num_points)
                    ys=np.linspace(min(each[0,1],each[1,1]),max(each[0,1],each[1,1]),num_points)
                    nn_decision_boundary.extend(np.stack((xs,ys),axis=1))
            if nn_decision_boundary!=[]:
                nn_decision_boundary=np.array(nn_decision_boundary)
                c_distance=chamfer_distance(nn_decision_boundary,boundary_coordinates)
                total_c_distances[NUM_HIDDEN_DIMS][sampled_rate].append(c_distance)

    fig,ax=plt.subplots(figsize=(6,3))
    ax.plot(sampled_rates,[sum(v)/len(v) for v in total_c_distances[NUM_HIDDEN_DIMS].values()])
    ax.set_xticks(sampled_rates,[str(f) for f in sampled_rates],fontsize=8,rotation=45)
    plt.savefig(f"{OUTPUT_DIRS}/c_distances.png",bbox_inches="tight")
    plt.close()

fig,ax=plt.subplots(figsize=(6,3))
ax.set_xticks(patch_rates,[str(p) for p in patch_rates],rotation=45,fontsize=8)
for k,NUM_HIDDEN_DIMS in enumerate(total_c_distances.keys()):
    avg_c_distances = [sum(v)/len(v) for v in total_c_distances[NUM_HIDDEN_DIMS].values() if len(v) > 0]
    filtered_sampled_rates = [rate for rate, v in zip(sampled_rates, total_c_distances[NUM_HIDDEN_DIMS].values()) if len(v) > 0]
    ax.plot(filtered_sampled_rates, avg_c_distances, label=NUM_HIDDEN_DIMS, color=plt.cm.viridis(k/len(total_c_distances.keys())))
ax.legend()
plt.savefig(f"./decision_boundary_fitting.png",bbox_inches="tight")