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


# In[ ]:


def exists(val):
    return val is not None

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

def to_value(t):
    return t.clone().detach().item()

def get_module_device(module):
    return next(module.parameters()).device


# #### Mapping Network

# In[ ]:


# class EqualLinear(nn.Module):
#     def __init__(self, in_dim, out_dim, lr_mul = 0.1, bias = True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_dim))

#         self.lr_mul = lr_mul

#     def forward(self, input):
#         return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

# class MappingNetwork(nn.Module):
#     def __init__(self, ARGS, depth = 3, lr_mul = 0.1):
#         super().__init__()

#         layers = []
#         for i in range(depth):
#             layers.extend([EqualLinear(ARGS.z_dim, ARGS.z_dim, lr_mul), leaky_relu()])

#         self.net = nn.Sequential(*layers)

#         self.to_gamma = nn.Linear(ARGS.z_dim, ARGS.dim_hidden)
#         self.to_beta = nn.Linear(ARGS.z_dim, ARGS.dim_hidden)

#     def forward(self, x):
#         x = self.net(x)
#         return self.to_gamma(x), self.to_beta(x)


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
        out = gamma * out + beta
        return torch.sin(out)
    
    
class Siren(nn.Module):
    def __init__(self, ARGS, in_features=3, out_features=1):
        super().__init__()
        
        self.net = nn.ModuleList([])

        self.net.append(SineLayer(in_features, ARGS.dim_hidden, 
                                  is_first=True, omega_0=ARGS.first_omega_0))

        for i in range(ARGS.siren_hidden_layers):
            self.net.append(SineLayer(ARGS.dim_hidden, ARGS.dim_hidden, 
                                      is_first=False, omega_0=ARGS.hidden_omega_0))

        self.final_linear = nn.Linear(ARGS.dim_hidden, out_features)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / ARGS.dim_hidden) / ARGS.hidden_omega_0, 
                                          np.sqrt(6 / ARGS.dim_hidden) / ARGS.hidden_omega_0)

        self.sigmoid = nn.Sigmoid()
            
    def forward(self, x, gamma, beta):
        for sine_layer in self.net: 
            x = sine_layer(x, gamma, beta)
        
        return self.sigmoid(self.final_linear(x))
    


# #### Complete SIREN

# In[ ]:


# class SirenGenerator(nn.Module):
#     def __init__(self, ARGS):
#         super().__init__()

#         self.mapping = MappingNetwork(ARGS)

#         self.siren = Siren(ARGS, in_features=3, out_features=1)
    
#     def forward(self, latent, coords):
#         gamma, beta = self.mapping(latent)
        
#         out = self.siren(coords, gamma, beta)
        
#         return out


# In[ ]:


print("Imported PI-Gan model.")

