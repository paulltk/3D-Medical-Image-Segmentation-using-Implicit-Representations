import torch
import numpy as np

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, gamma, beta):
        out = self.linear(input)
        
        out = self.omega_0 * out        
        
        out = out.permute(1, 0, 2)
        out = gamma * out + beta
        out = out.permute(1, 0, 2)
        
        out = torch.sin(out)
        
        return out
    
class Siren(nn.Module):
    def __init__(self, ARGS, in_features=3, out_features=1, 
                 first_omega_0=30., hidden_omega_0=30., 
                 final_activation="sigmoid"):
        super().__init__()
        
        self.siren_hidden_layers = ARGS.siren_hidden_layers
        self.dim_hidden = ARGS.dim_hidden
        
        self.net = nn.ModuleList([])

        self.net.append(SineLayer(in_features, ARGS.dim_hidden, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(ARGS.siren_hidden_layers):
            self.net.append(SineLayer(ARGS.dim_hidden, ARGS.dim_hidden, 
                                      is_first=False, omega_0=hidden_omega_0))

        self.final_linear = nn.Linear(ARGS.dim_hidden, out_features)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / ARGS.dim_hidden) / hidden_omega_0, 
                                          np.sqrt(6 / ARGS.dim_hidden) / hidden_omega_0)

        if final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "relu":
            self.final_activation = nn.ReLU()
        elif final_activation == None:
            self.final_activation = None
        else: 
            raise(Exception("Choose correct final activation in Siren model."))

            
    def forward(self, x, gamma, beta):
        
        coords = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        
        if gamma.shape[1:] == torch.Size([self.dim_hidden]):

                if i == 0: 
                    intermediate = sine_layer(coords, gamma, beta)
                else: 
                    intermediate = sine_layer(intermediate, gamma, beta)
            
        elif gamma.shape[1:] == torch.Size([self.siren_hidden_layers + 1, self.dim_hidden]): 
            for i, sine_layer in enumerate(self.net):    
                if i == 0: 
                    intermediate = sine_layer(coords, gamma[:, i, :], beta[:, i, :])
                else: 
                    intermediate = sine_layer(intermediate, gamma[:, i, :], beta[:, i, :])
            
        else: 
            raise(Exception("Shape of Gamma and Beta not correct")) 
    
        out = self.final_linear(intermediate)
        
        if self.final_activation: 
            out = self.final_activation(out)
            
        return out, coords