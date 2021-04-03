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

import math 


# In[2]:


get_ipython().run_line_magic('run', '"custom_datasets.ipynb"')


# In[3]:


def set_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return DEVICE 

DEVICE = set_device()
DEVICE = torch.device('cpu')

print('----------------------------------')
print('Using device for training:', DEVICE)
print('----------------------------------')


# In[4]:


data = PrepareData3D(["Aorta Volunteers", "Aorta BaV", "Aorta Resvcue", "Aorta CoA"], 
                     image_size="full", norm_min_max=[0,1])

train_ds = SirenDataset(data.train, DEVICE) 
train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=0)
print(train_ds.__len__())


val_ds = SirenDataset(data.val, DEVICE) 
val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=0)
print(val_ds.__len__())


# In[5]:


small_data = PrepareData3D(["Aorta Volunteers", "Aorta BaV", "Aorta Resvcue", "Aorta CoA"], 
                     image_size="small", norm_min_max=[0,1])

small_train_ds = SirenDataset(small_data.train, DEVICE) 
small_train_dataloader = DataLoader(small_train_ds, batch_size=1, num_workers=0)
print(small_train_ds.__len__())


small_val_ds = SirenDataset(small_data.val, DEVICE) 
small_val_dataloader = DataLoader(small_val_ds, batch_size=1, num_workers=0)
print(small_val_ds.__len__())


# In[11]:


subj_i = 6
_, _, _, _, _, pcmra_array, mask_array = train_ds[subj_i]
_, _, _, _, _, small_pcmra_array, small_mask_array = small_train_ds[subj_i]

slic = 12

fig, axes = plt.subplots(2, 2, figsize=(12,12))
axes[0, 0].imshow(pcmra_array.cpu().view(128, 128, 24).detach().numpy()[:, :, slic])
axes[0, 1].imshow(mask_array.cpu().view(128, 128, 24).detach().numpy()[:, :, slic])
axes[1, 0].imshow(small_pcmra_array.cpu().view(64, 64, 24).detach().numpy()[:, :, slic])
axes[1, 1].imshow(small_mask_array.cpu().view(64, 64, 24).detach().numpy()[:, :, slic])

plt.show()


# In[ ]:




