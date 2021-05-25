#!/usr/bin/env python

# coding: utf-8

import warnings

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

from py_files.new_dataset import *

from py_files.cnn_model import *
from py_files.pigan_model import *

from py_files.seq_pi_gan_functions import *


def train():  
    
    warnings.filterwarnings("ignore")
    
    ##### path to wich the model should be saved #####
    path = get_folder(ARGS)
    
    ##### save ARGS #####
    with open(f"{path}/ARGS.txt", "w") as f:
        print(vars(ARGS), file=f)
        
    ##### data preparation #####
    train_dl, val_dl, test_dl = initialize_dataloaders(ARGS)
    print(next(iter(test_dl))[1])
            
    ##### initialize models and optimizers #####
    models, optims, schedulers = load_models_and_optims(ARGS)
    
    
    ##### load pretrained model #####
    if ARGS.pretrained: 
        print(f"Loading pretrained model from '{ARGS.pretrained}'.")
        load_pretrained_models(ARGS.pretrained, ARGS.pretrained_best_dataset, ARGS.pretrained_best_loss,
                    models, optims, pretrained_models = ARGS.pretrained_models)
    
        if ARGS.pretrained_lr_reset:
            orig_lr = {"cnn": ARGS.cnn_lr, "mapping": ARGS.mapping_lr, "siren": ARGS.siren_lr, 
                       "pcmra_mapping": ARGS.pcmra_mapping_lr, "pcmra_siren": ARGS.pcmra_siren_lr}
            for name, optim in optims.items():
                for param_group in optim.param_groups: 
                    if param_group["lr"] != orig_lr[name]: 
                        param_group["lr"] = ARGS.pretrained_lr_reset
                print(f"{name} lr: {optim.param_groups[0]['lr']}")

    ##### loss function #####
    criterions = [nn.BCELoss(), nn.MSELoss()]
        
    ##### epoch, train loss mean, train loss std, val loss mean, val loss std #####
    mask_losses, pcmra_losses, dice_losses = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
    
    for ep in range(ARGS.pcmra_epochs):
    
        t = time.time() 

        for model in models.values():
            model.train()

        loss, _ = train_model(train_dl, models, optims, schedulers, criterions[1], ARGS, output="pcmra")
        
        
        if ep % ARGS.eval_every == 0: 

            print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
            
            t_pcmra_mean, t_pcmra_std, _, _ =                 val_model(train_dl, models, criterions[1], ARGS, output="pcmra", n_eval=100)
            
            v_pcmra_mean, v_pcmra_std, _, _ =                 val_model(val_dl, models, criterions[1], ARGS, output="pcmra", n_eval=100)

            pcmra_losses = np.append(pcmra_losses, [[ep ,t_pcmra_mean, t_pcmra_std, 
                                         v_pcmra_mean, v_pcmra_std]], axis=0)
            
            save_loss(path, pcmra_losses, models, optims, name="pcmra_loss", 
                      save_models=True)
        
    
    for ep in range(ARGS.mask_epochs):
    
        t = time.time() 

        for model in models.values():
            model.train()

        loss, _ = train_model(train_dl, models, optims, schedulers, criterions[0], ARGS, output="mask")
        
        
        if ep % ARGS.eval_every == 0: 

            print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
            
            t_mask_mean, t_mask_std, t_dice_mean, t_dice_std =                 val_model(train_dl, models, criterions[0], ARGS, output="mask", n_eval=100)
            
            v_mask_mean, v_mask_std, v_dice_mean, v_dice_std =                 val_model(val_dl, models, criterions[0], ARGS, output="mask", n_eval=100)

            mask_losses = np.append(mask_losses, [[ep ,t_mask_mean, t_mask_std, 
                                         v_mask_mean, v_mask_std]], axis=0)
            
            dice_losses = np.append(dice_losses, [[ep ,t_dice_mean, t_dice_std, 
                                         v_dice_mean, v_dice_std]], axis=0)
            
            save_loss(path, mask_losses, models, optims, name="mask_loss", 
                      save_models=True)
            
            save_loss(path, dice_losses, models, optims, name="dice_loss", 
                      save_models=False)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()

    # Arguments for training
    PARSER.add_argument('--device', type=str, default="GPU", 
                        help='Device that should be used.')

    PARSER.add_argument('--print_models', type=bool, default=False, 
                        help='Print the models after initialization or not.')

    PARSER.add_argument('--name', type=str, default="", 
                        help='Name of the folder where the output should be saved.')
    

    # pretrained params 
    
    PARSER.add_argument('--pretrained', type=str, default=None, 
                        help='Folder name of pretrained model that should be loaded.')
    
    PARSER.add_argument('--pretrained_best_dataset', type=str, default="train", 
                        help='Pretrained model with lowest [train, val] loss.')
    
    PARSER.add_argument('--pretrained_best_loss', type=str, default="mask", 
                        help='Pretrained model with lowest [train, val] loss.')
    
    PARSER.add_argument('--pretrained_models', type=str, default=None, 
                        help='Choose which pretrained models to load. None = all models')
    
    PARSER.add_argument('--pretrained_lr_reset', type=str, default=None, 
                        help='Reset the lr to a value.')
    
    
    
    # data
    PARSER.add_argument('--dataset', type=str, default="small", 
                        help='The dataset which we train on.')
    
    PARSER.add_argument('--rotate', type=bool, default=True, 
                        help='Rotations of the same image')
    
    PARSER.add_argument('--translate', type=bool, default=True, 
                        help='Translations of the same image')
    
    PARSER.add_argument('--flip', type=bool, default=True, 
                        help='Flips the train image')
    
    PARSER.add_argument('--crop', type=bool, default=True, 
                        help='Crops the train image')

    PARSER.add_argument('--stretch', type=bool, default=True, 
                        help='Stretches the train image')

    PARSER.add_argument('--norm_min_max', type=list, default=[0, 1], 
                        help='List with min and max for normalizing input.')
    
    PARSER.add_argument('--seed', type=int, default=34, 
                        help='Seed for initializig dataloader')
    
    
    # train variables
    PARSER.add_argument('--pcmra_epochs', type=int, default=5000, 
                        help='Number of epochs for pcmra training.')

    PARSER.add_argument('--mask_epochs', type=int, default=2500, 
                        help='Number of epochs for mask training.')
    
    PARSER.add_argument('--batch_size', type=int, default=24, 
                        help='Number of epochs.')
        
    PARSER.add_argument('--eval_every', type=int, default=50, 
                        help='Set the # epochs after which evaluation should be done.')
    
    PARSER.add_argument('--shuffle', type=bool, default=True, 
                        help='Shuffle the train dataloader?')
    
    PARSER.add_argument('--n_coords_sample', type=int, default=5000, 
                        help='Number of coordinates that should be sampled for each subject.')
    
    
    # CNN
    PARSER.add_argument('--cnn_setup', type=int, default=1, 
                        help='Setup of the CNN.')
    
    PARSER.add_argument('--pcmra_train_cnn', type=bool, default=True, 
                        help='Whether to also train the cnn during pcmra reconstruction.')

    PARSER.add_argument('--mask_train_cnn', type=bool, default=False, 
                        help='Whether to also train the cnn during mask segmentation.')


    
    # Mapping
    PARSER.add_argument('--mapping_setup', type=int, default=2, 
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
    
    PARSER.add_argument('--patience', type=int, default=100, 
                        help='Patience of the LR scheduler.')
    
    
    
    ARGS = PARSER.parse_args()
    
    train()
