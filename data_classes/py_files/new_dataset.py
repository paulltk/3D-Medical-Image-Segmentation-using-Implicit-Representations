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


# In[125]:


class SirenDataset(Dataset): 
    def __init__(self, root, subjects, DEVICE, dataset="train", max_t=(8, 8, 4)): 
        self.root = root
        self.dataset = dataset
        self.max_t = max_t
        self.DEVICE = DEVICE
        
        if self.dataset == "train":
            self.all_images = [image.split("__")[:3] for image in os.listdir(root) 
                               if list(image.split("__")[:2]) in subjects.tolist() 
                               and image.split("__")[3] == "pcmra.npy"]
        else:
            self.all_images = [image.split("__")[:3] for image in os.listdir(root) 
                               if list(image.split("__")[:2]) in subjects.tolist() 
                               and (len(image.split("__")[2].split("_")) == 1 
                                    or image.split("__")[2].split("_")[1] == "rot 0 (-, -)")
                               and image.split("__")[3] == "pcmra.npy"]
            
        self.pcmras = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{rot}__pcmra.npy")) 
                           for subj, proj, rot in self.all_images]).float().to(DEVICE)

        self.masks = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{rot}__mask.npy")) 
                           for subj, proj, rot in self.all_images]).float().to(DEVICE)

            
    def __len__(self):
        return len(self.all_images)

    
    def __getitem__(self, idx):
        subj, proj, rot = self.all_images[idx]
        pcmra = self.pcmras[idx]
        mask = self.masks[idx]
            
        if self.dataset == "train":
            shifts = self.get_random_shift()
            
            pcmra = self.translate_image(pcmra, shifts)
            mask = self.translate_image(mask, shifts)
                

        length = self.prod(pcmra.shape)

        coords = self.get_coords(*pcmra.shape).to(self.DEVICE)        
        pcmra_array = pcmra.view(length, 1)
        mask_array = mask.view(length, 1)
        
        pcmra = pcmra.permute(2, 0, 1).unsqueeze(0)
        
        return idx, subj, proj, pcmra, coords, pcmra_array, mask_array
    
    
    def get_random_shift(self):
        
        max_t = self.max_t
        
        shifts = (random.randint(-max_t[0], max_t[0]), 
                  random.randint(-max_t[1], max_t[1]), 
                  random.randint(-max_t[2], max_t[2]))
        
        return shifts
    
    
    def translate_image(self, image, shifts):

        image = torch.roll(image, shifts=shifts, dims=(0, 1, 2))

        for axis, shift in enumerate(shifts):
            idx = [[None, None], [None, None], [None, None]]

            if shift > 0: 
                idx[axis][1] = shift
            elif shift < 0: 
                idx[axis][0] = shift
            else:
                continue

            image[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1], idx[2][0]:idx[2][1]] = 0
            
        return image
    
    
    def prod(self, val) :  
        res = 1 
        for ele in val:  
            res *= ele  
        return res 

    
    def get_coords(self, *sidelengths):
        tensors = []

        for sidelen in sidelengths:
            tensors.append(torch.linspace(-1, 1, steps=sidelen))

        tensors = tuple(tensors)
        coords = torch.stack(torch.meshgrid(*tensors), dim=-1)
        
        return coords.reshape(-1, len(sidelengths))

