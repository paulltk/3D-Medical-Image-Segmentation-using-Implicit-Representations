#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, 
                 activation="relu", max_pool=None, layer_norm=None, batch_norm=False):
        
        super(ConvLayer, self).__init__()
        
        net = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding)]
        
        if activation == "relu":
            net.append(nn.ReLU())
        
        elif activation == "leakyrelu": 
            net.append(nn.LeakyReLU())
            
        if layer_norm: 
            net.append(nn.LayerNorm(layer_norm)) # add layer normalization
        
        if batch_norm: 
            net.append(nn.BatchNorm3d(out_channels)) # add batch normalization
        
        if max_pool: 
            net.append(nn.MaxPool3d(max_pool)) # add max_pooling
        
        self.model = nn.Sequential(*net)
        
    def forward(self, input):
        out = self.model(input)
        return out
    

class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class ReshapeTensor(nn.Module):
    def __init__(self, size): 
        super(ReshapeTensor, self).__init__()
        self.size = size
                
    def forward(self, input):
        return input.reshape([input.shape[0]] + self.size)


# ## Combi 1

# In[ ]:


class CNN1(nn.Module):

    def __init__(self):
        
        super(CNN1, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(2, 4, 4)),
            
            Flatten(),
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[176]:


class Mapping1(nn.Module):

    def __init__(self):
        super(Mapping1, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features=4096, out_features=2048, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(in_features=2048, out_features=1024, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(1024, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[177]:


class Mapping2(nn.Module):

    def __init__(self):
        super(Mapping2, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features=4096, out_features=2048, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(in_features=2048, out_features=1024, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(1024, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
                
        return out[:, :4, :], out[:, 4:, :]


# ## Combi 2 

# In[175]:


class CNN2(nn.Module):

    def __init__(self):
        
        super(CNN2, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(2, 4, 4)),
            
            Flatten(),
            
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.LeakyReLU(.2),

            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.LeakyReLU(.2),
         )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[172]:


class Mapping3(nn.Module):

    def __init__(self):
        super(Mapping3, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[ ]:


class Mapping4(nn.Module):

    def __init__(self):
        super(Mapping4, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, :4, :], out[:, 4:, :]


# ## Encoder

# In[ ]:


class Encoder(nn.Module):

    def __init__(self):
        
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(            
            ConvLayer(1,    16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      batch_norm=True, max_pool=(1, 2, 2)),
            
            ConvLayer(16,   32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,   32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(32,   64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,   64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(64,   128,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1, activation="relu"), 
            
            ConvLayer(128,   512,  kernel_size=(3, 4, 4), stride=1, padding=0)
            )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
class Encoder_1(nn.Module):

    def __init__(self):
        
        super(Encoder_1, self).__init__()
        
        self.model = nn.Sequential(            
            ConvLayer(1,    16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      batch_norm=True, max_pool=(1, 2, 2)),
            
            ConvLayer(16,   32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,   32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(32,   64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,   64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(64,   128,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True), 
            
            ConvLayer(128,   512,  kernel_size=(3, 4, 4), stride=1, padding=0)
            )

    def forward(self, x):
        out = self.model(x)
            
        return out

class Encoder_Mapping(nn.Module):

    def __init__(self):
        super(Encoder_Mapping, self).__init__()
        
        self.model = nn.Sequential(Flatten(), 
                                   nn.Linear(512, 512),
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, :4, :], out[:, 4:, :]
    
class Encoder_Mapping_1(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_1, self).__init__()
        
        self.model = nn.Sequential(Flatten(), 
                                   nn.Linear(512, 512),
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]
    
    
class Encoder_Mapping_1(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_1, self).__init__()
        
        self.gamma1 = self.gamma2 = self.gamma3 = self.gamma4 =             nn.Sequential(Flatten(), 
                          nn.Linear(512, 512),
                          nn.LeakyReLU(.2),
                          nn.Linear(512, 512), 
                          nn.LeakyReLU(.2),
                          nn.Linear(512, 512), 
                          nn.LeakyReLU(.2),
                          nn.Linear(512, 256), 
                           )
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[1]:


print("Imported CNN and Mapping functions.")


# In[ ]:




