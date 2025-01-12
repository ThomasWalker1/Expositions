from utils import *
from tqdm import tqdm
import pickle

EPOCHS=100
LR=1e-3
NUM_SHAPES=3
NUM_COLORS=2
OUTPUT_DIR=f"controlling_superposition/toy_models/extended/aggregation/"
NUM_SAMPLES=16
RESOLUTION=5
NUM_HIDDEN_DIMS=[2+k for k in range(RESOLUTION)]
#DISTRIBUTION_PARAMETERS=np.linspace(0,1,RESOLUTION)
DISTRIBUTION_PARAMETERS=0.0
NEGATIVE_SLOPES=np.linspace(0,1,RESOLUTION)

COI=generate_concepts(NUM_SHAPES,NUM_COLORS).keys()

output_discrepancies={concept:{(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS} for concept in COI}
intra_feature_unnormalised_alignments={(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS}
intra_feature_normalised_alignments={(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS}
latent_max_interferences={concept:{(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS} for concept in COI}
latent_min_interferences={concept:{(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS} for concept in COI}
losses={(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS}
feature_magnitudes={k:{(ns,dp):[] for ns in NEGATIVE_SLOPES for dp in NUM_HIDDEN_DIMS} for k in range(NUM_SHAPES+NUM_COLORS)}

if not(os.path.exists(f"{OUTPUT_DIR}/data/output_discrepancies")):
    for sample in range(NUM_SAMPLES):
        pbar=tqdm(NEGATIVE_SLOPES)
        for negative_slope in pbar:
            #for distribution_parameter in DISTRIBUTION_PARAMETERS:
            for num_hidden_dim in NUM_HIDDEN_DIMS:
                pbar.set_description(f"[{sample+1}]/[{NUM_SAMPLES}] negative slope {negative_slope}, hidden dims {num_hidden_dim}")
                loader=generate_loader(NUM_SHAPES,NUM_COLORS,DISTRIBUTION_PARAMETERS)

                model=Model(NUM_SHAPES+NUM_COLORS,negative_slope,num_hidden_dim)
                model,loss=train(model,loader,EPOCHS,LR,verbose=False)
                losses[(negative_slope,num_hidden_dim)].append(loss)

                W=model.W.detach().numpy()

                for k in range(W.shape[0]):
                    feature_magnitudes[k][(negative_slope,num_hidden_dim)].append(np.linalg.norm(W[k,:],ord=2))

                ods=output_discrepancy(NUM_SHAPES,NUM_COLORS,model,return_vals=True)
                for c,v in ods.items():
                    if c in COI:
                        output_discrepancies[c][(negative_slope,num_hidden_dim)].append(v)

                ifas=intra_feature_alignment(W,return_vals=True)
                intra_feature_unnormalised_alignments[(negative_slope,num_hidden_dim)].append(ifas["unnormalised"])
                intra_feature_normalised_alignments[(negative_slope,num_hidden_dim)].append(ifas["normalised"])

                lis=latent_interference(W,model,NUM_SHAPES,NUM_COLORS,return_vals=True)
                for c,v in lis.items():
                    if c in COI:
                        latent_max_interferences[c][(negative_slope,num_hidden_dim)].append(v["max"])
                        latent_min_interferences[c][(negative_slope,num_hidden_dim)].append(v["min"])

    def save_data(data,save_dir,filename):
        os.makedirs(save_dir,exist_ok=True)
        with open(f"{save_dir}/{filename}","wb") as file:
            pickle.dump(data,file)

    save_data(output_discrepancies,OUTPUT_DIR+"/data","output_discrepancies")
    save_data(intra_feature_unnormalised_alignments,OUTPUT_DIR+"/data","intra_feature_unnormalised_alignments")
    save_data(intra_feature_normalised_alignments,OUTPUT_DIR+"/data","intra_feature_normalised_alignments")
    save_data(latent_max_interferences,OUTPUT_DIR+"/data","latent_max_interferences")
    save_data(latent_min_interferences,OUTPUT_DIR+"/data","latent_min_interferences")
    save_data(losses,OUTPUT_DIR+"/data","losses")
    save_data(feature_magnitudes,OUTPUT_DIR+"/data","feature_magnitudes")

else:
    def load_data(save_dir,filename):
        with open(f"{save_dir}/{filename}","rb") as file:
            return pickle.load(file)

    output_discrepancies=load_data(OUTPUT_DIR+"/data","output_discrepancies")
    intra_feature_unnormalised_alignments=load_data(OUTPUT_DIR+"/data","intra_feature_unnormalised_alignments")
    intra_feature_normalised_alignments=load_data(OUTPUT_DIR+"/data","intra_feature_normalised_alignments")
    latent_max_interferences=load_data(OUTPUT_DIR+"/data","latent_max_interferences")
    latent_min_interferences=load_data(OUTPUT_DIR+"/data","latent_min_interferences")  
    losses=load_data(OUTPUT_DIR+"/data","losses")
    feature_magnitudes=load_data(OUTPUT_DIR+"/data","feature_magnitudes")

CONCEPTS_FOR_PLOT=["shape_0","shape_1","shape_2","color_0","color_1"]

def image_data(data,resolution):
    img=np.zeros((resolution,resolution))

    for coords,vs in data.items():
        img[np.where(NEGATIVE_SLOPES==coords[0])[0],np.where(np.array(NUM_HIDDEN_DIMS)==coords[1])[0]]=sum(vs)/len(vs)
    return img

def plot_concept_statistic(data,save_dir,title,concepts_for_plots,locs,vmin,vmax):

    fig,axs=plt.subplots(nrows=3,ncols=3)
    fig.suptitle(title)
    axs_flat=axs.flatten()
    for ax in axs_flat:
        ax.axis("off")

    for concept,loc in zip(concepts_for_plots,locs):
        im=axs_flat[loc].imshow(image_data(data[concept],RESOLUTION),vmin=vmin,vmax=vmax,origin="lower")
        axs_flat[loc].axis("on")
        #axs_flat[loc].set_xticks([0,RESOLUTION-1],["uniform","skewed"],fontsize=6)
        axs_flat[loc].set_xticks(range(RESOLUTION),[str(dim) for dim in NUM_HIDDEN_DIMS],fontsize=6)
        axs_flat[loc].set_yticks([0,RESOLUTION-1],["non-linear","linear"],rotation=90,fontsize=6)
        axs_flat[loc].set_title(concept)

    cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im,cax=cbar_ax)
    
    os.makedirs(save_dir,exist_ok=True)
    plt.savefig(f"{save_dir}/{title}.png",bbox_inches="tight")

plot_concept_statistic(output_discrepancies,OUTPUT_DIR+"/plots","output_discrepancies",CONCEPTS_FOR_PLOT,[0,2,4,6,8],vmin=0,vmax=None)
plot_concept_statistic(latent_max_interferences,OUTPUT_DIR+"/plots","latent_max_interferences",CONCEPTS_FOR_PLOT,[0,2,4,6,8],vmin=-1,vmax=1)
plot_concept_statistic(latent_min_interferences,OUTPUT_DIR+"/plots","latent_min_interferences",CONCEPTS_FOR_PLOT,[0,2,4,6,8],vmin=-1,vmax=1)

plot_concept_statistic(feature_magnitudes,OUTPUT_DIR+"/plots","feature_magnitudes",[0,1,2,3,4,5],[0,2,4,6,8],vmin=0,vmax=None)

def plot_model_statistic(data,save_dir,title,vmin,vmax):
    fig,ax=plt.subplots(nrows=1,ncols=1)
    fig.suptitle(title)
    
    im=ax.imshow(image_data(data,RESOLUTION),vmin=vmin,vmax=vmax,origin="lower")
    #ax.set_xticks([0,RESOLUTION-1],["uniform","skewed"])
    ax.set_xticks(range(RESOLUTION),[str(dim) for dim in NUM_HIDDEN_DIMS],fontsize=6)
    ax.set_yticks([0,RESOLUTION-1],["non-linear","linear"],rotation=90)

    fig.colorbar(im)

    os.makedirs(save_dir,exist_ok=True)
    plt.savefig(f"{save_dir}/{title}.png",bbox_inches="tight")

plot_model_statistic(intra_feature_unnormalised_alignments,OUTPUT_DIR+"/plots","intra_feature_unnormalised_alignments",vmin=0,vmax=None)
plot_model_statistic(intra_feature_normalised_alignments,OUTPUT_DIR+"/plots","intra_feature_normalised_alignments",vmin=0,vmax=None)
plot_model_statistic(losses,OUTPUT_DIR+"/plots","losses",vmin=0,vmax=None)