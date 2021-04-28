#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[32]:


class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
class SplitTensor(nn.Module):
    def __init__(self, n_tensors, tensor_size): 
        self.n_tensors = n_tensors
        self.tensor_size = tensor_size
        
        
    def forward(self, input):
        tensors = []
        for i in self.n_tensors: 
            tensors.append(input[:, i*self.array_size : (i+1)*self.array_size])
        return tensors
    
class CNN(nn.Module):

    def __init__(self, conv_layers):
        
        super(CNN, self).__init__()
        
        self.model = []
        self.split = False
        
        for layer in conv_layers:
            if layer["type"] == "conv": 
                self.model.append(nn.Conv3d(in_channels=layer["in"], out_channels=layer["out"], 
                                            kernel_size=layer["ker"], stride=layer["str"],
                                            padding=layer["pad"]))
                if "act" in layer.keys() and layer["act"] == "leakyrelu": 
                    self.model.append(nn.LeakyReLU(0.2))
                if "act" in layer.keys() and layer["act"] == "relu":
                    self.model.append(nn.ReLU())
                    
                
            elif layer["type"] == "lin": 
                self.model.append(nn.Linear(layer["in"], layer["out"]))
                if "act" in layer.keys() and layer["act"] == "leakyrelu": 
                    self.model.append(nn.LeakyReLU(0.2))
                if "act" in layer.keys() and layer["act"] == "relu":
                    self.model.append(nn.ReLU())
            
            elif layer["type"] == "flatten": 
                self.model.append(Flatten())  
            
            elif layer["type"] == "split": 
                self.split = True
                self.n_tensors = layer["n_tensors"]
                self.tensor_size = layer["tensor_size"]
            
            else: 
                raise(Exception("CNN layer type must be one of [conv, lin, flatten, split]."))
            
            if "max" in layer.keys(): 
                self.model.append(nn.MaxPool3d(layer["max"], stride=layer["max"], ceil_mode=True))
            
            if "norm" in layer.keys() and layer["norm"] != None: 
                if layer["norm"] == "layer": 
                    self.model.append(nn.LayerNorm(layer["ln"]))
            
            if "drop" in layer.keys() and layer["drop"] != None: 
                self.model.append(nn.Dropout(layer["drop"]))
            
        self.model = nn.Sequential(*self.model)


    def forward(self, x):
        out = self.model(x)
        
        if self.split: 
            out = [out[:, i*self.tensor_size : (i+1)*self.tensor_size] for i in range(self.n_tensors)]
            
        return out


# In[33]:


print("Imported CNN model.")


# In[58]:


# tensor = torch.randn((1, 1, 24, 64, 64))
# # print(tensor.shape)

# # output cnn torch.Size([1, 128, 2, 4, 4])
# # stride of 2, small kernel, small linear

# cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
#            , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu"} 
#            , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
#            , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
#            , {"type": "flatten"}
#            , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
#            , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
#            , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
#            , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
#            , {"type": "lin", "in": 512, "out": 512}
#            , {"type": "split", "n_tensors": 2, "tensor_size": 256}
#           ])
# print(cnn)

# out = cnn(tensor)

# print(out[0].shape)
# # print(out.shape)


# In[ ]:





# In[ ]:




