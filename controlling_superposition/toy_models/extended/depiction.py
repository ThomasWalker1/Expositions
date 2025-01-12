from utils import *

NUM_HIDDEN_DIMS=2
NUM_SHAPES=3
NUM_COLORS=2
OUTPUT_DIR="controlling_superposition/toy_models/extended/depiction"
loader=generate_loader(NUM_SHAPES,NUM_COLORS)

model=Model(NUM_SHAPES+NUM_COLORS,0.0,NUM_HIDDEN_DIMS)
model,loss=train(model,loader,100,1e-3)

W=model.W.detach().numpy()

plot_features(W[:NUM_SHAPES,:],W[NUM_SHAPES:,:],OUTPUT_DIR,"features")

output_discrepancy(NUM_SHAPES,NUM_COLORS,model,"output_discrepancy",OUTPUT_DIR,return_vals=False)

intra_feature_alignment(W,OUTPUT_DIR,"intra_feature_alignment",return_vals=False)

latent_interference(W,model,NUM_SHAPES,NUM_COLORS,OUTPUT_DIR,"latent_interference",return_vals=False)