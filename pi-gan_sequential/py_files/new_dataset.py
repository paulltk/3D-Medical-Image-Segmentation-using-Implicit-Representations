#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pydicom
import numpy as np
import torch 
import glob
import sys
import time 

import matplotlib.pyplot as plt
import matplotlib.path as mplPath

import pylab as pl

from threading import Timer

from collections import Counter
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
import random

from kornia.augmentation.augmentation3d import *
from kornia.geometry.transform import *


# In[ ]:


class SirenDataset(Dataset): 
    def __init__(self, root, subjects, DEVICE): 
        self.root = root
        self.DEVICE = DEVICE
                
        self.all_images = [image.split("__")[:3] for image in os.listdir(root) 
                           if list(image.split("__")[:2]) in subjects.tolist() 
                           and image.split("__")[3] == "pcmra.npy"]
            
        self.pcmras = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{size}__pcmra.npy")) 
                           for subj, proj, size in self.all_images]).float().to(DEVICE)

        self.masks = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{size}__mask.npy")) 
                           for subj, proj, size in self.all_images]).float().to(DEVICE)

            
    def __len__(self):
        return len(self.all_images)

    
    def __getitem__(self, idx):
        subj, proj, size = self.all_images[idx]
        pcmra = self.pcmras[idx]
        mask = self.masks[idx]
        
        pcmra = pcmra.permute(2, 0, 1).unsqueeze(0)
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        loss_cover = torch.ones(mask.shape).to(self.DEVICE)

        return idx, subj, proj, pcmra, mask, loss_cover

