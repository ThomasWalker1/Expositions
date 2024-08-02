import torch
import torch.nn as nn
from torch.nn import functional as F

from sklearn.datasets import make_circles,make_moons

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_dimension: int, bias=True):
        super().__init__()
        self.in_dimension=in_dimension
        self.linear=torch.nn.Linear(in_dimension,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))

class Classifier(nn.Module):
    def __init__(self, in_dimension: int, hidden_dimension, out_dimension, bias=True):
        super().__init__()
        self.in_layer=nn.Linear(in_dimension,hidden_dimension)
        self.hidden_layer=nn.Linear(hidden_dimension,hidden_dimension)
        self.out_layer=nn.Linear(hidden_dimension,out_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.in_layer(x)
        x=torch.relu(x)
        x=self.hidden_layer(x)
        x=torch.relu(x)
        x=self.out_layer(x)
        return x

class AE(nn.Module):
    def __init__(self,dimension,dropout=0.0):
        super(AE, self).__init__()
        self.dimension=dimension
        self.fc1 = nn.Linear(dimension, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, dimension)
        self.dropout = dropout
    def encode(self, x):
        h1 = F.dropout(F.relu(self.fc1(x)),p=self.dropout)
        return h1

    def decode(self, z):
        h3 = F.dropout(F.relu(self.fc2(z)),p=self.dropout)
        return self.fc3(h3)

    def forward(self, x):
        encoded = self.encode(x.view(-1, self.dimension))
        decoded = self.decode(encoded)
        return encoded,decoded
    
def AutoEncoderLoss(xs,A_xs,f_A_xs,t,alpha):
    reconstruction_loss=nn.MSELoss(reduction='mean')(xs,A_xs)
    one_hot=torch.zeros_like(f_A_xs)
    one_hot[:,t]=1
    target_loss=nn.CrossEntropyLoss(reduction='mean')(f_A_xs,one_hot)
    return reconstruction_loss+alpha*target_loss