#!/usr/bin/env python
# coding: utf-8

# In[7]:


# %run convert_ipynb_to_py_files.ipynb


# In[2]:


import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import argparse
import os
import math 
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import time
import pickle

from datetime import datetime
from pathlib import Path

# from data_classes.py_files.custom_datasets import *
from data_classes.py_files.new_dataset import *

from model_classes.py_files.cnn_model import *
from model_classes.py_files.pigan_model import *

from functions import *


# #### Import classes

# In[3]:


DEVICE = set_device()

print('----------------------------------')
print('Using device for training:', DEVICE)
print('----------------------------------')


# #### Train the model

# In[4]:


def train():  
    
    ##### path to wich the model should be saved #####
    path = get_folder(ARGS)
    
    ##### save ARGS #####
    with open(f"{path}/ARGS.txt", "w") as f:
        print(vars(ARGS), file=f)
        
    ##### data preparation #####
    train_dl, val_dl, test_dl = initialize_dataloaders(ARGS)  
    
#       train_dl, val_dl, test_dl = initialize_dataloaders(["Aorta Volunteers", "Aorta BaV",
#                                                "Aorta Resvcue", "Aorta CoA"], ARGS)  
    
    
    ##### initialize models and optimizers #####
    models, optims = load_models_and_optims(ARGS)

    ##### load pretrained model #####
    if ARGS.pretrained: 
        print(f"Loading pretrained model from '{ARGS.pretrained}'.")
        load_models(ARGS.pretrained, ARGS.pretrained_best, 
                    models, optims)
    
    ##### loss function #####
    criterion = nn.BCELoss()
    
    ##### epoch, train loss mean, train loss std, #####
    ##### val loss mean, val loss std #####
    losses = np.empty((0, 5))
    dice_losses = np.empty((0, 5))

    batch_count = 0     
    
    for ep in range(ARGS.epochs):
    
        t = time.time() 

        for model in models.values():
            model.train()

        t_loss_mean, t_loss_std, batch_count = train_epoch(train_dl, models, optims,
                                                           criterion, batch_count, ARGS)
        
        print(f"Epoch {ep}, train loss: {t_loss_mean}")
        
        if ep % ARGS.eval_every == 0: 

            print(f"Epoch {ep} took {round(time.time() - t)} seconds.")
            
            t_loss_mean, t_loss_std, t_dice_mean, t_dice_std = val_epoch(train_dl, models, criterion)
            v_loss_mean, v_loss_std, v_dice_mean, v_dice_std = val_epoch(val_dl, models, criterion)
            
            losses = np.append(losses, [[ep ,t_loss_mean, t_loss_std, 
                                         v_loss_mean, v_loss_std]], axis=0)
            
            dice_losses = np.append(dice_losses, [[ep ,t_dice_mean, t_dice_std, 
                                         v_dice_mean, v_dice_std]], axis=0)
            
            save_info(path, losses, dice_losses, models, optims)


# ## Run as .ipynb

# In[6]:


# ARGS = init_ARGS()
# ARGS.acc_steps = 64
# ARGS.epochs = 50
# ARGS.seed = 1
# ARGS.eval_every = 5

# ARGS.cnn_setup = 17

# train()  


# In[ ]:





# In[ ]:





# In[ ]:





# ## Run as .py

# In[ ]:


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    # Arguments for training
    PARSER.add_argument('--name', type=str, default="", 
                        help='Name of the folder where the output should be saved.')
    
    PARSER.add_argument('--pretrained', type=str, default=None, 
                        help='Folder name of pretrained model that should be loaded.')
    
    PARSER.add_argument('--pretrained_best', type=str, default="train", 
                        help='Pretrained model with lowest [train, val] loss.')
    
    # data
    PARSER.add_argument('--dataset', type=str, default="small", 
                        help='The dataset which we train on.')
    
    PARSER.add_argument('--norm_min_max', type=list, default=[0, 1], 
                        help='List with min and max for normalizing input.')
    
    PARSER.add_argument('--seed', type=int, default=34, 
                        help='List with min and max for normalizing input.')
    
    # train variables
    PARSER.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs.')
    
    PARSER.add_argument('--acc_steps', type=int, default=64, 
                        help='Number of subjects that the gradient is \
                        accumulated over before taking an optim step.')
    
    PARSER.add_argument('--eval_every', type=int, default=5, 
                        help='Set the # epochs after which evaluation should be done.')
    
    PARSER.add_argument('--shuffle', type=bool, default=True, 
                        help='Shuffle the train dataloader?')
    
    PARSER.add_argument('--n_coords_sample', type=int, default=5000, 
                        help='Number of coordinates that should be sampled for each subject.')
    
    
    # CNN
    PARSER.add_argument('--cnn_setup', type=int, default=17, 
                        help='Setup of the CNN.')
    
    # SIREN
    PARSER.add_argument('--dim_hidden', type=int, default=256, 
                        help='Dimension of hidden SIREN layers.')
    
    PARSER.add_argument('--siren_hidden_layers', type=int, default=3, 
                        help='Number of hidden SIREN layers.')
    
    PARSER.add_argument('--first_omega_0', type=float, default=30., 
                        help='Omega_0 of first layer.')
    
    PARSER.add_argument('--hidden_omega_0', type=float, default=30., 
                        help='Omega_0 of hidden layer.')
    
    
    # optimizers
    PARSER.add_argument('--cnn_lr', type=float, default=1e-4, 
                        help='Learning rate of cnn optim.')

    PARSER.add_argument('--siren_lr', type=float, default=1e-4, 
                        help='Learning rate of siren optim.')

    PARSER.add_argument('--mapping_lr', type=float, default=1e-4, 
                        help='Learning rate of mapping optim.')

    PARSER.add_argument('--cnn_wd', type=float, default=0, 
                        help='Weight decay of cnn optim.')

    PARSER.add_argument('--siren_wd', type=float, default=0, 
                        help='Weight decay of siren optim.')
    
    PARSER.add_argument('--mapping_wd', type=float, default=0, 
                        help='Weight decay of mapping optim.')
    
    
    
    
    
    
#     PARSER.add_argument('--z_dim', type=int, default=256, 
#                         help='Size of the latent pcmra representation.')
    
    
#     # MAPPING network
#     PARSER.add_argument('--with_mapping', type=bool, default=False, 
#                         help='Use the mapping network to produce a gamma and beta.')  
    
    
    ARGS = PARSER.parse_args()
    
    train()
