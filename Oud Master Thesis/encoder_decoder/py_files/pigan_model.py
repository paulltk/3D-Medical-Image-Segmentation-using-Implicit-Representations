#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time
import pickle

from datetime import datetime
from pathlib import Path

import math
from einops import repeat, rearrange


# ##### Siren from Sitzmann paper

# In[ ]:


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
        
        return torch.sin(out)
    
    
class Siren(nn.Module):
    def __init__(self, ARGS, in_features=3, out_features=1, 
                 first_omega_0=30., hidden_omega_0=30., 
                 final_activation="sigmoid"):
        super().__init__()
        
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
        for i, sine_layer in enumerate(self.net): 
            if gamma.shape[1:] == torch.Size([256]):
                x = sine_layer(x, gamma, beta)
            
            elif gamma.shape[1:] == torch.Size([4, 256]):    
                x = sine_layer(x, gamma[:, i, :], beta[:, i, :])
            
            else: 
                raise(Exception("Shape of Gamma and Beta not correct")) 
        
        x = self.final_linear(x)
        
        if self.final_activation: 
            x = self.final_activation(x)
            
        return x
    


# #### Complete SIREN

# In[6]:


print("Imported PI-Gan model.")


# In[ ]:





# In[ ]:




