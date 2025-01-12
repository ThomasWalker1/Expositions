import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

MACH_EPSILON=1e-8

class Model(nn.Module):
	def __init__(self,n_features,negative_slope=0.0,n_hidden=2):
		super().__init__()
		self.W = nn.Parameter(torch.empty((n_features,n_hidden)))
		nn.init.xavier_normal_(self.W)
		self.b_final = nn.Parameter(torch.zeros(n_features))

		self.negative_slope = negative_slope

	def forward(self, features):
		hidden = torch.einsum("...f,fh->...h", features, self.W)
		out = torch.einsum("...h,fh->...f", hidden, self.W)
		out = out + self.b_final
		out = F.relu(out) - self.negative_slope * F.relu(-out)
		return out, hidden
	
def generate_feature_distribution(num_features,distribution_parameter,scaling_parameter=4):
	total_sum=sum([np.exp(-k*distribution_parameter*scaling_parameter) for k in range(1,num_features+1)])
	return [np.exp(-k*distribution_parameter*scaling_parameter)/total_sum for k in range(1,num_features+1)]

def generate_loader(n_shape_features,n_color_features,distribution_parameter=None,scaling_parameter=4,sample_size=1024,batch_size=16):
	if distribution_parameter is None:
		d_shape_features=generate_feature_distribution(n_shape_features,0,scaling_parameter)
		d_color_features=generate_feature_distribution(n_color_features,0,scaling_parameter)
	else:
		d_shape_features=generate_feature_distribution(n_shape_features,distribution_parameter,scaling_parameter)
		d_color_features=generate_feature_distribution(n_color_features,distribution_parameter,scaling_parameter)
	
	if n_shape_features!=len(d_shape_features) or sum(d_shape_features)!=1:
		ValueError("Specify a valid distribution over color features")
	if n_color_features!=len(d_color_features) or sum(d_color_features)!=1:
		ValueError("Specify a valid distribution over color features")

	features=[]

	for _ in range(sample_size):
	
		color_feature=torch.zeros(n_color_features)
		color_feature[np.random.choice(range(n_color_features),p=d_color_features)]=1

		shape_feature=torch.zeros(n_shape_features)
		shape_feature[np.random.choice(range(n_shape_features),p=d_shape_features)]=1

		features.append(torch.concat([color_feature,shape_feature]).unsqueeze(0))

	features=torch.concat(features)
	dataset=TensorDataset(features,features)
	loader=DataLoader(dataset,batch_size=batch_size)
	
	return loader

def generate_concepts(n_shape_features,n_color_features):
	concepts={}
	for n_shape in range(n_shape_features):
		shape_features=torch.zeros(n_shape_features)
		shape_features[n_shape]=1
		concepts[f"shape_{n_shape}"]=torch.concat([shape_features,torch.zeros(n_color_features)])
	for n_color in range(n_color_features):
		color_features=torch.zeros(n_color_features)
		color_features[n_color]=1
		concepts[f"color_{n_color}"]=torch.concat([torch.zeros(n_shape_features),color_features])
	# for n_shape in range(n_shape_features):
	# 	shape_features=torch.zeros(n_shape_features)
	# 	shape_features[n_shape]=1
	# 	for n_color in range(n_color_features):
	# 		color_features=torch.zeros(n_color_features)
	# 		color_features[n_color]=1
	# 		concepts[f"shape_{n_shape}_color_{n_color}"]=torch.concat([shape_features,color_features])
	return concepts

def plot_features(W_shape,W_color,save_dir,save_title,d_shape_features=None,d_color_features=None):

	os.makedirs(save_dir,exist_ok=True)

	def feature_plot(ax,title,shape_features,color_features,d_shape_features=None,d_color_features=None,):
		if d_shape_features is None:
			d_shape_features=generate_feature_distribution(shape_features.shape[0],0)
		if d_color_features is None:
			d_color_features=generate_feature_distribution(color_features.shape[0],0)

		ax.axis("off")
		ax.set_title(title)
		ax.set_ylim(-1.5,1.5)
		ax.set_xlim(-1.5,1.5)

		for k in range(shape_features.shape[0]):
			ax.plot([0,shape_features[k,0]],[0,shape_features[k,1]],color=plt.cm.viridis(d_shape_features[k]/2),linewidth=2,alpha=0.5)
			ax.scatter([shape_features[k,0]],[shape_features[k,1]],color=plt.cm.viridis(d_shape_features[k]/2),s=20,alpha=0.75)

		for k in range(color_features.shape[0]):
			ax.plot([0,color_features[k,0]],[0,color_features[k,1]],color=plt.cm.viridis(1-d_color_features[k]/2),linewidth=2,alpha=0.5)
			ax.scatter([color_features[k,0]],[color_features[k,1]],color=plt.cm.viridis(1-d_color_features[k]/2),s=20,alpha=0.75)

	Wn_shape=W_shape/(np.linalg.norm(W_shape,ord=2,axis=1,keepdims=True)+MACH_EPSILON)
	Wn_color=W_color/(np.linalg.norm(W_color,ord=2,axis=1,keepdims=True)+MACH_EPSILON)
	fig,axs=plt.subplots(nrows=1,ncols=2,layout="tight",figsize=(6,3))
	feature_plot(axs[0],"unnormalised",W_shape,W_color,d_shape_features,d_color_features)
	feature_plot(axs[1],"normalised",Wn_shape,Wn_color,d_shape_features,d_color_features)
	plt.savefig(f"{save_dir}/{save_title}.png",bbox_inches="tight")
	plt.close()

def train(model,loader,epochs,lr,verbose=True):
	criterion=nn.MSELoss()
	optimizer=Adam(model.parameters(),lr=lr)

	if verbose:
		pbar=tqdm(range(epochs))
	else:
		pbar=range(epochs)
	for _ in pbar:
		epoch_loss=0
		count=0
		for input,target in loader:
			output=model(input)[0]
			loss=criterion(output,target)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss+=loss.item()*len(target)
			count+=len(target)
		if verbose:
			pbar.set_description(f"loss {epoch_loss/count:.4f}")
	return model, epoch_loss/count


def intra_feature_alignment(W,save_dir=None,save_title=None,return_vals=True):
	Wn=W/(np.linalg.norm(W,ord=2,axis=1,keepdims=True)+MACH_EPSILON)

	unnormalised_alignment=W@W.T
	normalised_alignment=Wn@Wn.T

	# aggregated_unnormalised_alignment=(np.sum(unnormalised_alignment**2)-W.shape[0])/(W.shape[0]**2-W.shape[0])
	# aggregated_normalised_alignment=(np.sum(normalised_alignment**2)-Wn.shape[0])/(Wn.shape[0]**2-Wn.shape[0])

	aggregated_unnormalised_alignment=np.max(np.triu(unnormalised_alignment,k=1))
	aggregated_normalised_alignment=np.max(np.triu(normalised_alignment,k=1))

	if save_dir is not None:
		os.makedirs(save_dir,exist_ok=True)
		fig,ax=plt.subplots(nrows=1,ncols=2,layout="tight")

		ax[0].set_title(f"unnormalised  ({aggregated_unnormalised_alignment:.3f})")
		ax[0].axis("off")
		max_abs=np.max(np.abs(unnormalised_alignment))
		im=ax[0].imshow(unnormalised_alignment,vmin=-max_abs,vmax=max_abs)
		fig.colorbar(im,ax=ax[0],fraction=0.046,pad=0.04)

		ax[1].set_title(f"normalised ({aggregated_normalised_alignment:.3f})")
		ax[1].axis("off")
		im=ax[1].imshow(normalised_alignment,vmin=-1,vmax=1)
		fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
		if save_title is None:
			plt.savefig(f"{save_dir}/{save_title}.png",bbox_inches="tight")
		else:
			plt.savefig(f"{save_dir}/{save_title}.png",bbox_inches="tight")
		plt.close()
	
	if return_vals:
		return {"unnormalised":aggregated_unnormalised_alignment,"normalised":aggregated_normalised_alignment}

def latent_interference(W,model,n_shapes,n_colors,save_dir=None,save_title=None,return_vals=True):
	concepts=generate_concepts(n_shapes,n_colors)
	
	interferences={}
	alignment=np.zeros((len(concepts),n_shapes+n_colors))

	for n,(concept,concept_input) in enumerate(concepts.items()):
		concept_latent=model(concept_input)[1].detach().numpy()
		max_interference=-np.Inf
		min_interference=np.Inf
		for m in range(W.shape[0]):
			#latent_interference=np.dot(W[m,:],concept_latent)/(np.linalg.norm(W[m,:],ord=2)*np.linalg.norm(concept_latent,ord=2)+MACH_EPSILON)
			latent_interference=np.dot(W[m,:],concept_latent)
			alignment[n,m]=latent_interference
			if concept_input[m]==0:
				max_interference=max(max_interference,latent_interference)
				min_interference=min(min_interference,latent_interference)
		interferences[concept]={"max":max_interference,"min":min_interference}

	if save_dir is not None:
		os.makedirs(save_dir,exist_ok=True)
		fig,ax=plt.subplots(nrows=1,ncols=1)
		im=ax.imshow(alignment)
		ax.set_yticks(range(len(concepts)),[f"{c} ({ints["max"]:.3f}/{ints["min"]:.3f})" for c,ints in interferences.items()])
		fig.colorbar(im)
		if save_title is None:
			plt.savefig(f"{save_dir}/latent_interference.png",bbox_inches="tight")
		else:
			plt.savefig(f"{save_dir}/{save_title}.png",bbox_inches="tight")		
		plt.close()
	
	if return_vals:
		return interferences

def output_discrepancy(n_shapes,n_colors,model,save_title=None,save_dir=None,return_vals=True):
	concepts=generate_concepts(n_shapes,n_colors)
	expected=np.zeros((len(concepts),n_shapes+n_colors))
	obtained=np.zeros((len(concepts),n_shapes+n_colors))

	discrepancies=[]

	for k,concept_input in enumerate(concepts.values()):
		output=model(concept_input)[0].detach()
		discrepancies.append(F.mse_loss(output,concept_input))
		expected[k,:]=concept_input.numpy()
		obtained[k,:]=output.numpy()

	if save_dir is not None:
		os.makedirs(save_dir,exist_ok=True)
		fig,ax=plt.subplots(nrows=1,ncols=2)

		im=ax[0].imshow(expected,vmin=0,vmax=1)
		ax[0].set_yticks(range(len(concepts)),[f"{q} - ({d:.4f})" for q,d in zip(concepts.keys(),discrepancies)])
		ax[0].set_title("expected")

		im=ax[1].imshow(obtained,vmin=0,vmax=1)
		fig.subplots_adjust(right=0.8)
		cbar_ax=fig.add_axes([0.85, 0.15, 0.05, 0.7])
		fig.colorbar(im,cax=cbar_ax)
		plt.savefig(f"{save_dir}/{save_title}.png",bbox_inches="tight")
		plt.close()
	
	if return_vals:
		return {q:d for q,d in zip(concepts.keys(),discrepancies)}