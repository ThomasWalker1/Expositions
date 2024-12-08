from sae_lens import SAE
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

sae_ids=[
    "layer_0/width_16k/average_l0_105",
    "layer_1/width_16k/average_l0_102",
    "layer_2/width_16k/average_l0_141",
    "layer_3/width_16k/average_l0_59",
    "layer_4/width_16k/average_l0_124",
    "layer_5/width_16k/average_l0_68",
    "layer_6/width_16k/average_l0_70",
    "layer_7/width_16k/average_l0_69",
    "layer_8/width_16k/average_l0_71",
    "layer_9/width_16k/average_l0_73",
    "layer_10/width_16k/average_l0_77",
    "layer_11/width_16k/average_l0_80",
    "layer_12/width_16k/average_l0_41",
    "layer_13/width_16k/average_l0_84",
    "layer_14/width_16k/average_l0_84",
    "layer_15/width_16k/average_l0_78",
    "layer_16/width_16k/average_l0_78",
    "layer_17/width_16k/average_l0_77",
    "layer_18/width_16k/average_l0_74",
    "layer_19/width_16k/average_l0_73",
    "layer_20/width_16k/average_l0_71",
    "layer_21/width_16k/average_l0_70",
    "layer_22/width_16k/average_l0_72",
    "layer_23/width_16k/average_l0_75",
    "layer_24/width_16k/average_l0_73",
    "layer_25/width_16k/average_l0_116",
    ]


if not(os.path.exists("feature_geometry/similarities_2b")):
    os.makedirs("feature_geometry/similarities_2b")

embeddings=np.random.normal(size=(16384,2304))
embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True)
cosine_similarities=np.dot(embeddings,embeddings.T)
flattened_cosine_distances_random=cosine_similarities[np.tril_indices(cosine_similarities.shape[0],k=-1)]
high_sim_feats_random=np.where(np.absolute(flattened_cosine_distances_random)>0.1)[0]
random_high_sim_weight=len(high_sim_feats_random)/len(flattened_cosine_distances_random)

layers=[]
high_sim_weight=[]

fig,axs=plt.subplots(nrows=1,ncols=7,figsize=(7*3,3),layout="tight")
axs[6].hist(flattened_cosine_distances_random[high_sim_feats_random],bins=250,color="blue")
axs[6].set_title("random")
axs[6].set_xlim(-1,1)
max_freq_axis=0
pbar=tqdm(sae_ids,total=len(sae_ids))
for sae_id in pbar:
    release = "gemma-scope-2b-pt-res"
    layer=int(sae_id.split("/")[0].split("_")[1])
    pbar.set_description(str(layer))

    ground_sae = SAE.from_pretrained(release, sae_id)[0]
    embeddings=ground_sae.W_dec.detach().numpy()
    embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True)
    cosine_similarities=np.dot(embeddings,embeddings.T)

    flattened_cosine_distances=cosine_similarities[np.tril_indices(cosine_similarities.shape[0],k=-1)]
    high_sim_feats=np.where(np.absolute(flattened_cosine_distances)>0.1)[0]

    layers.append(layer)
    high_sim_weight.append(len(high_sim_feats)/len(flattened_cosine_distances))

    if layer%5==0:
        axs[layer//5].hist(flattened_cosine_distances[high_sim_feats],bins=250,color="blue")
        axs[layer//5].set_title(str(layer))
        max_freq_axis=max(max_freq_axis,axs[layer//5].get_ylim()[1])

for k in range(6):
    axs[k].set_ylim(0,max_freq_axis)
    axs[k].set_xlim(-1,1)
    if k!=0 and k!=6:
        axs[k].set_yticks([])
plt.savefig(f"feature_geometry/similarities_2b/hist_some_layers.png",bbox_inches="tight")
plt.close()

fig,ax=plt.subplots(nrows=1,ncols=1)
ax.plot(layers,high_sim_weight,color="blue")
ax.plot(layers,random_high_sim_weight*np.ones(len(layers)),color="red")
plt.savefig("feature_geometry/similarities_2b/high_sim_weight.png")