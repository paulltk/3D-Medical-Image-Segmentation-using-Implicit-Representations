#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'convert_ipynb_to_py_files.ipynb')


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


# #### Train the model

# In[3]:


def train():  
    
    ##### path to wich the model should be saved #####
    path = get_folder(ARGS)
    
    ##### save ARGS #####
    with open(f"{path}/ARGS.txt", "w") as f:
        print(vars(ARGS), file=f)
        
    ##### data preparation #####
    train_dl, val_dl, test_dl = initialize_dataloaders(ARGS) 
    
            
    ##### initialize models and optimizers #####
    models, optims, schedulers = load_models_and_optims(ARGS)

    ##### load pretrained model #####
    if ARGS.pretrained: 
        print(f"Loading pretrained model from '{ARGS.pretrained}'.")
        load_models(ARGS.pretrained, ARGS.pretrained_best, 
                    models, optims)
    
    ##### loss function #####
    criterions = [nn.BCELoss(), nn.MSELoss()]
#     criterions = [nn.BCELoss(), nn.L1Loss()]
        
    ##### epoch, train loss mean, train loss std, #####
    ##### val loss mean, val loss std #####
    mask_losses = np.empty((0, 5))
    pcmra_losses = np.empty((0, 5))
    dice_losses = np.empty((0, 5))

    batch_count = 0     
    
    for ep in range(ARGS.epochs):
    
        t = time.time() 

        for model in models.values():
            model.train()

        t_loss_mean, t_loss_std,         t_p_loss_mean, t_p_loss_std,         batch_count = train_epoch(train_dl, models, optims, schedulers,
                                  criterions, batch_count, ARGS)
        
        print(f"Epoch {ep}, train loss: {t_loss_mean}, train pcmra loss: {t_p_loss_mean}")
        
        if ep % ARGS.eval_every == 0: 

            print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
            
            t_mask_mean, t_mask_std,             t_pcmra_mean, t_pcmra_std,             t_dice_mean, t_dice_std = val_epoch(train_dl, models, criterions, ARGS)
            
            v_mask_mean, v_mask_std,             v_pcmra_mean, v_pcmra_std,             v_dice_mean, v_dice_std = val_epoch(val_dl, models, criterions, ARGS)
            
            mask_losses = np.append(mask_losses, [[ep ,t_mask_mean, t_mask_std, 
                                         v_mask_mean, v_mask_std]], axis=0)
            
            pcmra_losses = np.append(pcmra_losses, [[ep ,t_pcmra_mean, t_pcmra_std, 
                                         v_pcmra_mean, v_pcmra_std]], axis=0)
            
            dice_losses = np.append(dice_losses, [[ep ,t_dice_mean, t_dice_std, 
                                         v_dice_mean, v_dice_std]], axis=0)
            
            save_info(path, mask_losses, pcmra_losses, dice_losses, models, optims, save_last=True)


# ## Run as .ipynb

# In[ ]:


torch.cuda.empty_cache()    

# ARGS.n_coords_sample=-1
# ARGS.batch_size = 4

versions = [(20000, 12), (5000, 24)]

ARGS = init_ARGS()


for ARGS.n_coords_sample, ARGS.batch_size in versions: 
    for ARGS.pcmra_lambda in [5, 10]:
        ARGS.epochs = 70
        ARGS.print_models = False
        ARGS.rotate, ARGS.translate, ARGS.flip = True, True, False


        ARGS.scheduler_on = "combined"
        ARGS.reconstruction = "pcmra"

        ARGS.share_mapping = False

        ARGS.cnn_setup = 1
        ARGS.mapping_setup = 2

        print(vars(ARGS))

        train()  

        torch.cuda.empty_cache()    


# In[ ]:


torch.cuda.empty_cache()    

# ARGS.n_coords_sample=-1
# ARGS.batch_size = 4

versions = [(20000, 12), (5000, 24)]

ARGS = init_ARGS()


for ARGS.n_coords_sample, ARGS.batch_size in versions: 
    for ARGS.pcmra_lambda in [5, 10]:
        ARGS.epochs = 70
        ARGS.print_models = False
        ARGS.rotate, ARGS.translate, ARGS.flip = True, True, False


        ARGS.scheduler_on = "combined"
        ARGS.reconstruction = "mask"

        ARGS.share_mapping = False

        ARGS.cnn_setup = 1
        ARGS.mapping_setup = 2

        print(vars(ARGS))

        train()  

        torch.cuda.empty_cache()    


# In[ ]:





# In[ ]:


# ARGS = init_ARGS()

# ARGS.print_models = False
# ARGS.epochs = 250
# ARGS.eval_every = 20

# cnn_mapping_combis = [[1, 1], [1, 2], [2, 3], [2, 4]]
# transformations = [(False, False, False)]
# share_mapping = [False, True]
# reconstruction = ["pcmra", "both", "mask"]
# lambdas = [(1, 1), (1, 10)]
# omega_0s = [5, 30, 100]

# for ARGS.cnn_setup, ARGS.mapping_setup in cnn_mapping_combis: 
#     for ARGS.rotate, ARGS.translate, ARGS.flip in transformations: 
#         for ARGS.share_mapping in share_mapping: 
#             for ARGS.reconstruction in reconstruction: 
#                 for ARGS.mask_lambda, ARGS.pcmra_lambda in lambdas: 
#                     for ARGS.first_omega_0 in omega_0s: 
#                         print(vars(ARGS))
#                         train()  
                    


# In[ ]:


# ARGS = init_ARGS()

# ARGS.print_models = False
# ARGS.epochs = 51

# transformations = [(True, True, False), (True, True, True)]
# cnn_mapping_combis = [[1, 2], [2, 3], [2, 4]]
# share_mapping = [False, True]
# reconstruction = ["pcmra", "both", "mask"]
# lambdas = [(1, 1), (1, 10)]
# omega_0s = [30]

# for ARGS.rotate, ARGS.translate, ARGS.flip in transformations: 
#     for ARGS.cnn_setup, ARGS.mapping_setup in cnn_mapping_combis: 
#         for ARGS.share_mapping in share_mapping: 
#             for ARGS.reconstruction in reconstruction: 
#                 for ARGS.mask_lambda, ARGS.pcmra_lambda in lambdas: 
#                     for ARGS.first_omega_0 in omega_0s: 
#                         print(vars(ARGS))
#                         train()  
                    


# ## Run as .py

# In[ ]:


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    # Arguments for training
    PARSER.add_argument('--device', type=str, default="GPU", 
                        help='Device that should be used.')

    PARSER.add_argument('--print_models', type=str, default="GPU", 
                        help='Print the models after initialization or not.')

    PARSER.add_argument('--name', type=str, default="", 
                        help='Name of the folder where the output should be saved.')
    
    PARSER.add_argument('--pretrained', type=str, default=None, 
                        help='Folder name of pretrained model that should be loaded.')
    
    PARSER.add_argument('--pretrained_best', type=str, default="train", 
                        help='Pretrained model with lowest [train, val] loss.')
    
    
    # PCMRA and Mask reconstruction
    PARSER.add_argument('--reconstruction', type=str, default="pcmra", 
                        help='Do we want the CNN trained on reconstruction of pcmra/mask/both.')
    
    PARSER.add_argument('--share_mapping', type=bool, default=False, 
                        help='Do we want to share the mapping from latent repr to gamma and beta?')
    
    PARSER.add_argument('--pcmra_lambda', type=float, default=1., 
                        help='Multiplier for pcmra reconstruction loss')
    
    PARSER.add_argument('--mask_lambda', type=float, default=1., 
                        help='Multiplier for mask reconstruction loss')
    
    
    # data
    PARSER.add_argument('--dataset', type=str, default="small", 
                        help='The dataset which we train on.')
    
    PARSER.add_argument('--rotate', type=bool, default=True, 
                        help='Rotations of the same image')
    
    PARSER.add_argument('--translate', type=bool, default=True, 
                        help='Translations of the same image')
    
    PARSER.add_argument('--flip', type=bool, default=True, 
                        help='Flips of the same image')
    
    PARSER.add_argument('--norm_min_max', type=list, default=[0, 1], 
                        help='List with min and max for normalizing input.')
    
    PARSER.add_argument('--seed', type=int, default=34, 
                        help='Seed for initializig dataloader')
    
    
    # train variables
    PARSER.add_argument('--epochs', type=int, default=51, 
                        help='Number of epochs.')
    
    PARSER.add_argument('--batch_size', type=int, default=24, 
                        help='Number of epochs.')
        
    PARSER.add_argument('--eval_every', type=int, default=5, 
                        help='Set the # epochs after which evaluation should be done.')
    
    PARSER.add_argument('--shuffle', type=bool, default=True, 
                        help='Shuffle the train dataloader?')
    
    PARSER.add_argument('--n_coords_sample', type=int, default=5000, 
                        help='Number of coordinates that should be sampled for each subject.')
    
    
    # CNN
    PARSER.add_argument('--cnn_setup', type=int, default=1, 
                        help='Setup of the CNN.')

    
    # Mapping
    PARSER.add_argument('--mapping_setup', type=int, default=1, 
                        help='Setup of the Mapping network.')

    
    # SIREN
    PARSER.add_argument('--dim_hidden', type=int, default=256, 
                        help='Dimension of hidden SIREN layers.')
    
    PARSER.add_argument('--siren_hidden_layers', type=int, default=3, 
                        help='Number of hidden SIREN layers.')
    
    
    PARSER.add_argument('--first_omega_0', type=float, default=30., 
                        help='Omega_0 of first layer.')
    
    PARSER.add_argument('--hidden_omega_0', type=float, default=30., 
                        help='Omega_0 of hidden layer.')
    
    
    PARSER.add_argument('--pcmra_first_omega_0', type=float, default=30., 
                        help='Omega_0 of first layer of PCMRA siren.')
    
    PARSER.add_argument('--pcmra_hidden_omega_0', type=float, default=30., 
                        help='Omega_0 of hidden layer of PCMRA siren.')
    
    
    # optimizers
    PARSER.add_argument('--cnn_lr', type=float, default=1e-4, 
                        help='Learning rate of cnn optim.')

    PARSER.add_argument('--cnn_wd', type=float, default=0, 
                        help='Weight decay of cnn optim.')

    
    PARSER.add_argument('--mapping_lr', type=float, default=1e-4, 
                        help='Learning rate of siren optim.')
    
    PARSER.add_argument('--pcmra_mapping_lr', type=float, default=1e-4, 
                        help='Learning rate of siren optim.')
    

    PARSER.add_argument('--siren_lr', type=float, default=1e-4, 
                        help='Learning rate of siren optim.')

    PARSER.add_argument('--siren_wd', type=float, default=0, 
                        help='Weight decay of siren optim.')
    
    
    PARSER.add_argument('--pcmra_siren_lr', type=float, default=1e-4, 
                        help='Learning rate of PCMRA siren optim.')    
    
    PARSER.add_argument('--pcmra_siren_wd', type=float, default=0, 
                        help='Weight decay of PCMRA siren optim.')
    
    
    PARSER.add_argument('--scheduler_on', type=str, default="combined", 
                        help='Schedule lr on pcmra/mask/combined loss.')
    
 
    ARGS = PARSER.parse_args()
    
    train()


# In[ ]:




