#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle

from PIL import Image
from datetime import datetime
from pathlib import Path


# In[3]:


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
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
        
    def forward(self, input):
        out = torch.sin(self.omega_0 * self.linear(input))
        return out
    


# In[4]:


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 first_different_init=True, outermost_layer="None", 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        
        
        # first layer 
        if first_different_init:
            self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        else: 
            self.net.append(SineLayer(in_features, hidden_features, omega_0=hidden_omega_0))

            
        # hidden layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))

            
        # last layer
        if outermost_layer.lower() == "linear":
            final_linear = nn.Linear(hidden_features, out_features)    
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)  
            self.net.append(final_linear)

        elif outermost_layer.lower() == "sigmoid":
            final_linear = nn.Linear(hidden_features, out_features)    
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)  
            self.net.append(final_linear)
            self.net.append(nn.Sigmoid())
        
        elif outermost_layer.lower() == "sine":
            self.net.append(SineLayer(hidden_features, out_features, omega_0=hidden_omega_0))
        
        else: 
            raise(Exception("Choose a correct outermost_layer"))
            
        
        # add all to network
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output


# In[15]:


# siren = Siren(in_features=3, out_features=1, hidden_features=256, 
#               hidden_layers=4, first_different_init=True, 
#               outermost_layer="sigmoid")

# print(siren.net[0].linear.weight.shape)


# In[ ]:


print("Imported SIREN model.")

