#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import os
import time
import argparse
import math 
import skimage
import pickle
import ast
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import pylab as pl

from datetime import datetime
from pathlib import Path
from collections import Counter
from collections import defaultdict
from threading import Timer
from PIL import Image

# from data_classes.py_files.custom_datasets import *
from data_classes.py_files.new_dataset import *

from model_classes.py_files.cnn_model import *
from model_classes.py_files.pigan_model import *


# #### ARGS class for .ipynb files

# In[2]:


class init_ARGS(object): 
    def __init__(self): 
        self.name = ""
        self.pretrained = None
        self.pretrained_best = "train"
        self.dataset = "small"
        self.norm_min_max = [0, 1]
        self.seed = 34
        self.epochs = 500
        self.acc_steps = 10
        self.eval_every = 10
        self.shuffle = True
        self.n_coords_sample = 5000
        self.cnn_setup = 1
        self.dim_hidden = 256
        self.siren_hidden_layers = 3
        self.first_omega_0 = 30.
        self.hidden_omega_0 = 30.
        self.cnn_lr = 1e-4
        self.siren_lr = 1e-4
        self.mapping_lr = 1e-4
        self.cnn_wd = 0
        self.siren_wd = 0
        self.mapping_wd = 0
        self.rotated = True


        print("WARNING: ARGS class initialized.")

    def set_args(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
          
        
def load_args(run):
    run_path = os.path.join("saved_runs", run, "ARGS.txt")

    with  open(run_path, "r") as f:
        contents = f.read()
        args_dict = ast.literal_eval(contents)
    
    ARGS = init_ARGS()
    
    old_args = vars(ARGS)
    
    for k, v in args_dict.items(): 
        if k in old_args.keys(): 
            if old_args[k] != v: 
                print(f"Changed param \t{k}: {v}.") 
        else:
            print(f"New param \t{k}: {v}.")
            
    ARGS.set_args(args_dict)
    
    return ARGS


# #### Set torch device

# In[3]:


def set_device():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return DEVICE 

DEVICE = set_device()

print('----------------------------------')
print('Using device for training:', DEVICE)
print('----------------------------------')


# ####  Model saving functions 

# In[4]:


def get_folder(ARGS): 
    now = datetime.now()
    dt = now.strftime("%d-%m-%Y %H:%M:%S")
    path = f"saved_runs/pi-gan {dt} {ARGS.name}"
    
    Path(f"{path}").mkdir(parents=True, exist_ok=True)   

    return path
    

def plot_graph(path, x, ys_and_labels, axes=("Epochs", "BCELoss"), fig_name="loss_plot"):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    
    for y, label in ys_and_labels: 
        ax.plot(x[1:], y[1:], label=label)

    plt.xlabel(axes[0])
    plt.ylabel(axes[1])
    legend = ax.legend(loc='upper right')
    
    plt.savefig(f"{path}/{fig_name}.png")
    plt.close()
    
def save_info(path, losses, dice_losses, models, optims): 
    
    np.save(f"{path}/losses.npy", losses)
    np.save(f"{path}/dice_losses.npy", dice_losses)
    
    eps = losses[:, 0]
    train_losses = losses[:, 1]
    val_losses = losses[:, 3]
    train_d_losses = dice_losses[:, 1]
    val_d_losses = dice_losses[:, 3]
    
    print(f"Train loss: \t {round(train_losses[-1], 5)}, \tdice loss: \t {round(train_d_losses[-1], 5)}.")
    print(f"Val loss: \t {round(val_losses[-1], 5)}, \tdice loss: \t {round(val_d_losses[-1], 5)}.")

    if train_losses[-1] == train_losses.min(): 
        print(f"New best train loss, saving model.")

        for model in models.keys():
            torch.save(models[model].state_dict(), f"{path}/{model}_train.pt")
            torch.save(optims[model].state_dict(), f"{path}/{model}_optim_train.pt")
        
    
    if val_losses[-1] == val_losses.min(): 
        print(f"New best val loss, saving model.")

        for model in models.keys():
            torch.save(models[model].state_dict(), f"{path}/{model}_val.pt")
            torch.save(optims[model].state_dict(), f"{path}/{model}_optim_val.pt")

    plot_graph(path, eps, [(train_losses, "Train loss"), (val_losses, "Eval loss")], 
               axes=("Epochs", "BCELoss"), fig_name="loss_plot")
    
    plot_graph(path, eps, [(train_d_losses, "Train dice loss"), (val_d_losses, "Eval dice loss")], 
               axes=("Epochs", "BCELoss"), fig_name="dice_loss_plot")
    


# #### Initialize dataloaders

# In[5]:


# def initialize_dataloaders(projects, ARGS):
#     assert(ARGS.dataset in ["full", "small"])

#     data = PrepareData3D(projects, seed=ARGS.seed, image_size=ARGS.dataset, norm_min_max=ARGS.norm_min_max)

#     train_ds = SirenDataset(data.train, DEVICE) 
#     train_dl = DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=ARGS.shuffle)
#     print("Train subjects:", train_ds.__len__())

#     val_ds = SirenDataset(data.val, DEVICE) 
#     val_dl = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)
#     print("Validation subjects:", val_ds.__len__())
    
#     test_ds = SirenDataset(data.test, DEVICE) 
#     test_dl = DataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False)
#     print("Test subjects:", test_ds.__len__())
    
#     return train_dl, val_dl, test_dl
    
def initialize_dataloaders(ARGS):
    
    assert(ARGS.dataset in ["full", "small"])
    
    root = "/home/ptenkaate/scratch/Master-Thesis/Dataset/"
    if ARGS.dataset == "small":
        root += "scaled_normalized"
    else: 
        root += "original_normalized"
    
    if ARGS.rotated: 
        root += "_rotated"
    
    subjects = [file.split("__")[:2] for file in  os.listdir(root)]
    subjects = np.array([list(subj) for subj in list(set(map(tuple, subjects)))])

    idx = list(range(subjects.shape[0]))
    split1, split2 = int(len(idx) * 0.6), int(len(idx) * 0.8)

    random.shuffle(idx) # shuffles indices
    train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:] # incides per data subset

    train_subjects, val_subjects, test_subjects =  subjects[train_idx], subjects[val_idx], subjects[test_idx]

    train_ds = SirenDataset(root, train_subjects, DEVICE)
    train_dl = DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=ARGS.shuffle)
    print("Train subjects:", train_dl.__len__())
    
    val_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="val")
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)
    print("Val subjects:", val_dl.__len__())
    
    test_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="test")
    test_dl = DataLoader(train_ds, batch_size=1, num_workers=0, shuffle=False)
    print("Test subjects:", test_ds.__len__())

    return train_dl, val_dl, test_dl


# #### Initialize models

# In[ ]:


def load_models_and_optims(ARGS):
        models = {}
        optims = {}
        
        models["cnn"] = load_cnn(ARGS).cuda()
        optims["cnn"] = torch.optim.Adam(lr=ARGS.cnn_lr, params=models["cnn"].parameters(), 
                                         weight_decay=ARGS.cnn_wd)
        
#         if ARGS.with_mapping:
#             models["mapping"] = MappingNetwork(ARGS).cuda()
#             optims["mapping"] = torch.optim.Adam(lr=ARGS.mapping_lr, params=models["mapping"].parameters(), 
#                                                  weight_decay=ARGS.mapping_wd)
        
        models["siren"] = Siren(ARGS, in_features=3, out_features=1).cuda()
        optims["siren"] = torch.optim.Adam(lr=ARGS.siren_lr, params=models["siren"].parameters(), 
                                           weight_decay=ARGS.siren_wd)
        
        for model, struct in models.items(): 
            print(struct)
            
        return models, optims


# #####  Random coords subsample

# In[6]:


def choose_random_coords(*arrays, n=1000): 
    
    mx = arrays[0].shape[1]
    rand_idx = random.sample(range(mx), n)
    
    arrays = [array.detach().clone()[:, rand_idx, :] for array in arrays]
    
    return arrays


# #### Dice loss

# In[ ]:


def calc_dice_loss(pred, target):
    
    smooth = 0.

    pred = torch.round(pred)

    pflat = pred.flatten()
    tflat = target.flatten()
    intersection = (pflat * tflat).sum()

    A_sum = torch.sum(pflat * pflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


# #### Train and validation epoch functions

# In[7]:


def train_epoch(dataloader, models, optims, criterion, batch_count, ARGS):
    losses = []
    
    for _, _, _, pcmra, coords, _, mask_array in dataloader:
        siren_in, siren_labels = choose_random_coords(coords, mask_array, n=ARGS.n_coords_sample)

        cnn_out = models["cnn"](pcmra)
        if "mapping" in models.keys(): 
            gamma, beta = models["mapping"](cnn_out)
        else: 
            gamma, beta = cnn_out
            
        siren_out = models["siren"](siren_in, gamma, beta)
         
        loss = criterion(siren_out, siren_labels) 
                
        losses.append(loss.item())
        loss = loss / ARGS.acc_steps
        loss.backward()

        batch_count += 1
        if batch_count % ARGS.acc_steps == 0: 
            for _, optim in optims.items():
                optim.step()
                optim.zero_grad()
    
    
    mean, std = round(np.mean(losses), 6), round(np.std(losses), 6)
    
    return mean, std, batch_count


def val_epoch(dataloader, models, criterion, n_eval=100):
    losses = []
    d_losses = []

    i = 0
    
    for idx, subj, proj, pcmra, coords, pcmra_array, mask_array in dataloader:    
        siren_out = get_complete_image(models, pcmra, coords)
        
        loss = criterion(siren_out, mask_array)  
        d_loss = calc_dice_loss(siren_out, mask_array) 

        losses.append(loss.item())
        d_losses.append(d_loss.item())
        
        i += 1
        if i == n_eval:
            break    
    
    loss_mean, loss_std = round(np.mean(losses), 6), round(np.std(losses), 6)
    d_loss_mean, d_loss_std = round(np.mean(d_losses), 6), round(np.std(d_losses), 6)
    
    return loss_mean, loss_std, d_loss_mean, d_loss_std


def get_complete_image(models, pcmra, coords, val_n = 10000): 
    for model in models.values(): 
        model.eval() #evaluation mode    
        
    image = torch.Tensor([]).cuda() # initialize results tensor
    
    cnn_out = models["cnn"](pcmra)
    
    if "mapping" in models.keys(): 
        gamma, beta = models["mapping"](cnn_out)
    else: 
        gamma, beta = cnn_out
                
    n_slices = math.ceil(coords.shape[1] / val_n) # number of batches
    for i in range(n_slices):
        coords_in = coords[:, (i*val_n) : ((i+1)*val_n), :]
        siren_out = models["siren"](coords_in, gamma, beta)
        image = torch.cat((image, siren_out.detach()), 1)
    
    for model in models.values(): 
        model.train() #train mode
    
    return image 


# #### Load pretrained models

# In[8]:


def load_models(folder, best, models, optims): 
    path = f"saved_runs/{folder}/"

    for key in models.keys(): 
        models[key].load_state_dict(torch.load(f"{path}/{key}_{best}.pt"))
        optims[key].load_state_dict(torch.load(f"{path}/{key}_optim_{best}.pt"))


# In[9]:


print("Loaded all helper functions.")


# #### Load CNN setup

# In[1]:


def load_cnn(ARGS): 
    
    if ARGS.cnn_setup == 1: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # maxpool,  small kernel, small linear
        
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 128, "act": "leakyrelu"}
                   , {"type": "lin", "in": 128, "out": 128, "act": "leakyrelu"}
                   , {"type": "lin", "in": 128, "out": 128, "act": "leakyrelu"}
                   , {"type": "lin", "in": 128, "out": 128, "act": "leakyrelu"}
                   , {"type": "lin", "in": 128, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
        
    elif ARGS.cnn_setup == 2: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # maxpool,  small kernel, medium linear
        
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 3: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # maxpool,  small kernel, large linear
        
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 4: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # maxpool,  small kernel, large linear but short linear
        
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 1, "pad": 1, "max": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
        
    
    elif ARGS.cnn_setup == 5: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # stride of 2, small kernel, small linear
        
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])

    elif ARGS.cnn_setup == 6: 
        # output cnn torch.Size([1, 128, 2, 4, 4])
        # stride of 1, 2, small kernel, small linear

        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 1, "pad": 1, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu"} 
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 1, "pad": 1, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu"} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 1, "pad": 1, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])

    elif ARGS.cnn_setup == 7: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 7, "str": 2, "pad": 3, "act": "relu"}
               , {"type": "conv", "in": 16, "out": 32, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "conv", "in": 32, "out": 64, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "conv", "in": 64, "out": 128, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "flatten"}
               , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
               , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
               , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
               , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
               , {"type": "lin", "in": 512, "out": 512}
               , {"type": "split", "n_tensors": 2, "tensor_size": 256}
              ])
    
    elif ARGS.cnn_setup == 8: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 7, "str": 2, "pad": 3, "act": "relu"}
               , {"type": "conv", "in": 16, "out": 32, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "conv", "in": 32, "out": 64, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "conv", "in": 64, "out": 128, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
               , {"type": "flatten"}
               , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
               , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
               , {"type": "lin", "in": 1024, "out": 512}
               , {"type": "split", "n_tensors": 2, "tensor_size": 256}
              ])
        
    elif ARGS.cnn_setup == 9: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 7, "str": 2, "pad": 3, "act": "relu", 
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)} 
                   , {"type": "conv", "in": 64, "out": 128, "ker": 7, "str": 2, "pad": 3, "act": "relu"} 
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])

    elif ARGS.cnn_setup == 10: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])

    
    elif ARGS.cnn_setup == 11: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512, "act": "leakyrelu"}
                   , {"type": "lin", "in": 512, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 12:
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 3, "str": 2, "pad": 1, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 3, "str": 2, "pad": 1, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 13: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 14: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 15: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 1, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 1, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 1, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 64, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 1, "pad": 2, "act": "relu"}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 16: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                   "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                     "norm": "layer", "ln": (2, 4, 4)}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 17: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (24, 64, 64)}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 64, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu", 
                    "norm": "layer", "ln": (2, 4, 4)}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 18: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (24, 64, 64)}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 64, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 19: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (24, 64, 64)}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 64, "out": 64, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 7, "str": 2, "pad": 3, "act": "relu"}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
        
    elif ARGS.cnn_setup == 20: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (24, 64, 64)}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 64, "out": 64, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 7, "str": 1, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 7, "str": 2, "pad": 3, "act": "relu",
                    "norm": "layer", "ln": (2, 4, 4)}
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
    elif ARGS.cnn_setup == 21: 
        cnn = CNN([{"type": "conv", "in": 1, "out": 16, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (24, 64, 64)}
                   , {"type": "conv", "in": 16, "out": 16, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "max": 2, "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 16, "out": 32, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (12, 32, 32)}
                   , {"type": "conv", "in": 32, "out": 32, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 32, "out": 64, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (6, 16, 16)} 
                   , {"type": "conv", "in": 64, "out": 64, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 64, "out": 128, "ker": 5, "str": 1, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (3, 8, 8)}
                   , {"type": "conv", "in": 128, "out": 128, "ker": 5, "str": 2, "pad": 2, "act": "relu",
                    "norm": "layer", "ln": (2, 4, 4)}   
                   , {"type": "flatten"}
                   , {"type": "lin", "in": 4096, "out": 2048, "act": "leakyrelu"}
                   , {"type": "lin", "in": 2048, "out": 1024, "act": "leakyrelu"}
                   , {"type": "lin", "in": 1024, "out": 512}
                   , {"type": "split", "n_tensors": 2, "tensor_size": 256}
                  ])
    
        
        
    
    return cnn.cuda()


# In[ ]:


class Show_images(object):
    """
    Scroll through slices. Takes an unspecified number of subfigures per figure.
    suptitles: either a str or a list. Represents the 
    main title of a figure. 
    images_titles: a list with tuples, each tuple an np.array and a 
    title for the array subfigure. 
    """
    def __init__(self, suptitles, *images_titles):
        # if string if given, make list with that title for 
        # each slice.
        if type(suptitles) == str: 
            self.suptitles = []
            for i in range(images_titles[0][0].shape[2]): 
                self.suptitles.append(suptitles)
        else: 
            self.suptitles = suptitles
                    
        self.fig, self.ax = plt.subplots(1,len(images_titles))

        # split tuples with (image, title) into lists
        self.images = [x[0] for x in images_titles]
        self.titles = [x[1] for x in images_titles]

        # get the number of slices that are to be shown
        rows, cols, self.slices = self.images[0].shape        
        self.ind = 0

        self.fig.suptitle(self.suptitles[self.ind]) # set title 

        self.plots = []
        
        # start at slice 10 if more than 20 slices, 
        # otherwise start at middle slice.
        if self.images[0].shape[2] > 20: 
            self.ind = 10
        else:
            self.ind = self.images[0].shape[2] // 2
        
        # make sure ax is an np array
        if type(self.ax) == np.ndarray:
            pass
        else: 
            self.ax = np.array([self.ax])
        
        # create title for each subfigure in slice
        for (sub_ax, image, title) in zip(self.ax, self.images, self.titles): 
            sub_ax.set_title(title)
            plot = sub_ax.imshow(image[:, :, self.ind], vmin=0, vmax=1)
            self.plots.append(plot)

            
        # link figure to mouse scroll movement
        self.plot_show = self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
        

    def onscroll(self, event):
        """
        Shows next or previous slice with mouse scroll.
        """
        if event.button == 'up':
            self.ind = (self.ind - 1) % self.slices
        else:
            self.ind = (self.ind + 1) % self.slices
        
        self.update()
        

    def update(self):
        """
        Updates the figure.
        """
        self.fig.suptitle(self.suptitles[self.ind])
        
        for plot, image in zip(self.plots, self.images):
            plot.set_data(image[:, :, self.ind])
        
        self.ax[0].set_ylabel('Slice Number: %s' % self.ind)
        self.plots[0].axes.figure.canvas.draw()

