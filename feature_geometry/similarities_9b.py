from sae_lens import SAE
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

sae_ids={
    "pt":[
        "layer_9/width_16k/average_l0_100",
        "layer_20/width_16k/average_l0_11",
        "layer_31/width_16k/average_l0_114",
    ],
    "it":[
        "layer_9/width_16k/average_l0_88",
        "layer_20/width_16k/average_l0_91",
        "layer_31/width_16k/average_l0_76",
    ]
}

if not(os.path.exists("feature_geometry/similarities_9b")):
    os.makedirs("feature_geometry/similarities_9b")

high_sim_weight={}

fig,axs=plt.subplots(nrows=1,ncols=3,figsize=(3*3,3),layout="tight")
max_freq_axis=0

for model_type in ["pt","it"]:
    high_sim_weight[model_type]=[]
    pbar=tqdm(enumerate(sae_ids[model_type]),total=len(sae_ids[model_type]))
    for k,sae_id in pbar:
        release = f"gemma-scope-9b-{model_type}-res"
        layer=int(sae_id.split("/")[0].split("_")[1])
        pbar.set_description(str(layer))

        ground_sae = SAE.from_pretrained(release, sae_id)[0]
        embeddings=ground_sae.W_dec.detach().numpy()
        
        embeddings/=np.linalg.norm(embeddings,axis=1,keepdims=True)
        cosine_similarities=np.dot(embeddings,embeddings.T)

        flattened_cosine_distances=cosine_similarities[np.tril_indices(cosine_similarities.shape[0],k=-1)]
        
        high_sim_feats=np.where(np.absolute(flattened_cosine_distances)>0.1)[0]

        high_sim_weight[model_type].append(len(high_sim_feats)/len(flattened_cosine_distances))

        axs[k].hist(flattened_cosine_distances[high_sim_feats],bins=250,color="blue")
        axs[k].set_title(str(layer))
        max_freq_axis=max(max_freq_axis,axs[k].get_ylim()[1])

for k in range(3):
    axs[k].set_ylim(0,max_freq_axis)
    axs[k].set_xlim(-1,1)
    if k!=0 and k!=6:
        axs[k].set_yticks([])
plt.savefig(f"feature_geometry/similarities_9b/histograms.png",bbox_inches="tight")

fig,ax=plt.subplots(nrows=1,ncols=1)
ax.plot(range(3),high_sim_weight["pt"],color="blue",label="pretrained")
ax.plot(range(3),high_sim_weight["it"],color="red",label="instruction tuned")
ax.set_xticks([0,1,2])
ax.set_xticklabels(["9","20","31"])
plt.legend()
plt.savefig(f"feature_geometry/similarities_9b/high_sim_weight.png")