import torch
import numpy as np

import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

####################################################################################
############################# CNN Layer Classes ####################################
####################################################################################

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


####################################################################################
############################# Final CNN setups #####################################
####################################################################################


########################################
############# CNN_Golden ###############
######################################## 

class CNN_Golden(nn.Module):

    def __init__(self):
        
        super(CNN_Golden, self).__init__()
        
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


########################################
############# CNN_Maxpool ##############
######################################## 

class CNN_MaxPool(nn.Module):

    def __init__(self):
        
        super(CNN_MaxPool, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu",
                max_pool=2, layer_norm=(24, 32, 32)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=2, layer_norm=(12, 16, 16)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=1, padding=2, activation="relu",
                max_pool=2,  layer_norm=(6, 8, 8)), 
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
############## CNN_Deep ################
######################################## 

class CNN_Deep(nn.Module):

    def __init__(self):
        
        super(CNN_Deep, self).__init__()
        
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
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu",
                layer_norm=(3, 4, 4)),    

            ConvLayer(128, 256, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(256, 256, kernel_size=5, stride=1, padding=1, activation="relu"),    
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
########### CNN_BatchNorm ##############
######################################## 

class CNN_BatchNorm(nn.Module):

    def __init__(self):
        
        super(CNN_BatchNorm, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), batch_norm=True),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),    
        )

    def forward(self, x):
        out = self.model(x)
            
        return out