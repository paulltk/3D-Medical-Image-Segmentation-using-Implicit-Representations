#!/usr/bin/env python
# coding: utf-8

# In[4]:


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

from torchinfo import summary


# In[5]:


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

# In[6]:


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


# In[7]:


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


# In[8]:


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

# In[9]:


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


# In[10]:


class Mapping3(nn.Module):

    def __init__(self):
        super(Mapping3, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[11]:


class Mapping4(nn.Module):

    def __init__(self):
        super(Mapping4, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, :4, :], out[:, 4:, :]


# ## Encoder

# In[12]:


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
    

class Encoder_2(nn.Module):

    def __init__(self):
        
        super(Encoder_2, self).__init__()
        
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
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1), 
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
    


# # Currently used 

# ### CNN

# #### Output 128, 3, 4, 4

# In[24]:


########################################
############ cnn_setup 5 ###############
########################################

class CNN3(nn.Module):

    def __init__(self):
        
        super(CNN3, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
############ cnn_setup 13 ##############
########################################

class CNN11(nn.Module):

    def __init__(self):
        
        super(CNN11, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup 6 ###############
########################################
class CNN4(nn.Module):

    def __init__(self):
        
        super(CNN4, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out

    
########################################
############ cnn_setup 7 ###############
########################################    
class CNN5(nn.Module):

    def __init__(self):
        
        super(CNN5, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
########################################
############ cnn_setup 8 ###############
######################################## 

class CNN6(nn.Module):

    def __init__(self):
        
        super(CNN6, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
########################################
############ cnn_setup 9 ###############
######################################## 

class CNN7(nn.Module):

    def __init__(self):
        
        super(CNN7, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,  16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,  32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,  64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128, 128, kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


    
########################################
############ cnn_setup 10 ##############
######################################## 

class CNN8(nn.Module):

    def __init__(self):
        
        super(CNN8, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup 14 ##############
######################################## 

class CNN12(nn.Module):

    def __init__(self):
        
        super(CNN12, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
    
########################################
############ cnn_setup 15 ##############
######################################## 

class CNN13(nn.Module):

    def __init__(self):
        
        super(CNN13, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
########################################
############ cnn_setup 16 ##############
######################################## 

class CNN14(nn.Module):

    def __init__(self):
        
        super(CNN14, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
    
########################################
############ cnn_setup 17 ##############
######################################## 

class LargeCNN(nn.Module):

    def __init__(self):
        
        super(LargeCNN, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out

########################################
############ cnn_setup 18 ##############
######################################## 

class LargeCNN1(nn.Module):

    def __init__(self):
        
        super(LargeCNN1, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# #### Output 512, 1, 1, 1

# In[29]:


########################################
############ cnn_setup 11 ##############
######################################## 

class CNN9(nn.Module):

    def __init__(self):
        
        super(CNN9, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
            ConvLayer(128, 512, kernel_size=(3, 4, 4), stride=1, padding=0, activation="relu", 
                      layer_norm=(1, 1, 1)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
    
########################################
############ cnn_setup 12 ##############
######################################## 

class CNN10(nn.Module):

    def __init__(self):
        
        super(CNN10, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
            ConvLayer(128, 512, kernel_size=(3, 4, 4), stride=1, padding=0,  activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[40]:


# cnn = CNN11().cuda()
# input_shape = (24, 1, 24, 64, 64)

# inp = torch.randn(input_shape).cuda()

# summary(cnn, input_size=input_shape, depth=3)


# ### Mappings

# #### Input 512, 1, 1, 1

# In[41]:


########################################
########## mapping_setup 6 #############
########################################

class Encoder_Mapping_2(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_2, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(512, 512),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(512, 512),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
             
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]

    


# #### Input 128, 3, 4, 4

# In[42]:


########################################
########## mapping_setup 7 #############
########################################

class Encoder_Mapping_3(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_3, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
            
        
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]
    
    
########################################
########## mapping_setup 8 #############
########################################

class Encoder_Mapping_4(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_4, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
            
        
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]


# In[44]:


# mapping = Encoder_Mapping_4().cuda()
# # input_shape = (24, 512, 1, 1, 1)
# input_shape = (24, 128, 3, 4, 4)
# inp = torch.randn(input_shape).cuda()

# summary(mapping, input_size=input_shape, depth=3)


# In[43]:


print("Imported CNN and Mapping functions.")

