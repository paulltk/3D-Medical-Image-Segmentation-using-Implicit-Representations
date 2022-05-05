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

from py_files.new_dataset import *

from py_files.cnn_model import *
from py_files.pigan_model import *


# #### ARGS class for .ipynb files

# In[6]:


class init_ARGS(object): 
    def __init__(self): 
        self.device = "GPU"
        self.print_models = "GPU"
        self.name = ""
        self.pretrained = None
        self.pretrained_best = "train"
        self.reconstruction = "pcmra"
        self.share_mapping = True
        self.pcmra_lambda = 1.
        self.mask_lambda = 1.
        self.dataset = "small"
        self.rotate = True
        self.translate = True
        self.flip = True
        self.norm_min_max = [0, 1]
        self.seed = 34
        self.epochs = 51
        self.batch_size = 24
        self.eval_every = 5
        self.shuffle = True
        self.n_coords_sample = 5000
        self.cnn_setup = 1
        self.mapping_setup = 1
        self.dim_hidden = 256
        self.siren_hidden_layers = 3
        self.first_omega_0 = 30.
        self.hidden_omega_0 = 30.
        self.pcmra_first_omega_0 = 30.
        self.pcmra_hidden_omega_0 = 30.
        self.cnn_lr = 1e-4
        self.cnn_wd = 0
        self.mapping_lr = 1e-4
        self.pcmra_mapping_lr = 1e-4
        self.siren_lr = 1e-4
        self.siren_wd = 0
        self.pcmra_siren_lr = 1e-4
        self.pcmra_siren_wd = 0
        self.scheduler_on = "combined"

        print("WARNING: ARGS class initialized.")

    def set_args(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
          
        
def load_args(run, print_changed=True):
    run_path = os.path.join("saved_runs", run, "ARGS.txt")

    with  open(run_path, "r") as f:
        contents = f.read()
        args_dict = ast.literal_eval(contents)
    
    ARGS = init_ARGS()
    
    old_args = vars(ARGS)
    
    if print_changed:
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
    
    print('----------------------------------')
    print('Using device for training:', DEVICE)
    print('----------------------------------')

    return DEVICE 

# DEVICE = set_device()


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

    
def save_info(path, mask_losses, pcmra_losses, dice_losses, models, optims, save_last=False): 
    
    np.save(f"{path}/losses.npy", mask_losses)
    np.save(f"{path}/pcmra_losses.npy", pcmra_losses)
    np.save(f"{path}/dice_losses.npy", dice_losses)
    
    eps = mask_losses[:, 0]
    
    train_m_losses = mask_losses[:, 1]
    val_m_losses = mask_losses[:, 3]
    
    train_p_losses = pcmra_losses[:, 1]
    val_p_losses = pcmra_losses[:, 3]
    
    train_d_losses = dice_losses[:, 1]
    val_d_losses = dice_losses[:, 3]
    
    print(f"Train mask loss: \t {round(train_m_losses[-1], 5)},     pcmra loss: \t {round(train_p_losses[-1], 5)},     \tdice loss: \t {round(train_d_losses[-1], 5)}.")
    
    print(f"Eval  mask loss: \t {round(val_m_losses[-1], 5)},     pcmra loss: \t {round(val_p_losses[-1], 5)},     \tdice loss: \t {round(val_d_losses[-1], 5)}.")

    if train_m_losses[-1] == train_m_losses.min() or save_last: 
        print(f"New best train loss, saving model.")

        for model in models.keys():
            torch.save(models[model].state_dict(), f"{path}/{model}_train.pt")
            torch.save(optims[model].state_dict(), f"{path}/{model}_optim_train.pt")
        
    
    if val_m_losses[-1] == val_m_losses.min(): 
        print(f"New best val loss, saving model.")

        for model in models.keys():
            torch.save(models[model].state_dict(), f"{path}/{model}_val.pt")
            torch.save(optims[model].state_dict(), f"{path}/{model}_optim_val.pt")

    plot_graph(path, eps, [(train_m_losses, "Train loss"), (val_m_losses, "Eval loss")], 
               axes=("Epochs", "BCELoss"), fig_name="loss_plot")
    
    plot_graph(path, eps, [(train_p_losses, "Train loss"), (val_p_losses, "Eval loss")], 
               axes=("Epochs", "MSELoss"), fig_name="pcmra_loss_plot")
    
    plot_graph(path, eps, [(train_d_losses, "Train dice loss"), (val_d_losses, "Eval dice loss")], 
               axes=("Epochs", "Dice Loss"), fig_name="dice_loss_plot")
    


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
    
    if ARGS.device.lower() == "cpu": 
        DEVICE = torch.device("cpu")
        
        print('----------------------------------')
        print('Using device for training:', DEVICE)
        print('----------------------------------')
    
    else: 
        DEVICE = set_device()
    
    
    assert(ARGS.dataset in ["full", "small"])
    
    root = "/home/ptenkaate/scratch/Master-Thesis/Dataset/"
    if ARGS.dataset == "small":
        root += "scaled_normalized"
    else: 
        root += "original_normalized"
    
    if ARGS.rotate: 
        root += "_rotated"
        
    subjects = [file.split("__")[:2] for file in  sorted(os.listdir(root))]
    subjects = np.array(sorted([list(subj) for subj in list(set(map(tuple, subjects)))]))
    
    idx = list(range(subjects.shape[0]))
    split1, split2 = int(len(idx) * 0.6), int(len(idx) * 0.8)
    
    random.seed(ARGS.seed)

    random.shuffle(idx) # shuffles indices
    train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:] # incides per data subset

    train_subjects, val_subjects, test_subjects =  subjects[train_idx], subjects[val_idx], subjects[test_idx]

    train_ds = SirenDataset(root, train_subjects, DEVICE, dataset="train", 
                            translate=ARGS.translate, flip=ARGS.flip)
    train_dl = DataLoader(train_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=ARGS.shuffle)
    print("Train subjects:", train_ds.__len__())
    print(train_ds.all_images[:10])
    
    val_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="val")
    val_dl = DataLoader(val_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Val subjects:", val_ds.__len__())
    
    test_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="test")
    test_dl = DataLoader(test_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Test subjects:", test_ds.__len__())

    return train_dl, val_dl, test_dl


# #### Initialize models

# In[2]:


def load_models_and_optims(ARGS):
    
    if ARGS.device.lower() == "cpu": 
        DEVICE = torch.device("cpu")
        
        print('----------------------------------')
        print('Using device for training:', DEVICE)
        print('----------------------------------')

    else: 
        DEVICE = set_device()
        

    models = {}
    optims = {}
    schedulers = {}

    models["cnn"] = load_cnn(ARGS).to(DEVICE)
    optims["cnn"] = torch.optim.Adam(lr=ARGS.cnn_lr, params=models["cnn"].parameters(), 
                                     weight_decay=ARGS.cnn_wd)
    schedulers["cnn"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optims["cnn"], patience=5, verbose=True)
    
    models["mapping"] = load_mapping(ARGS).to(DEVICE)
    optims["mapping"] = torch.optim.Adam(lr=ARGS.mapping_lr, params=models["mapping"].parameters())
    schedulers["mapping"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optims["mapping"], patience=5, verbose=True)

    models["siren"] = Siren(ARGS, in_features=3, out_features=1,first_omega_0=ARGS.first_omega_0, 
                            hidden_omega_0=ARGS.hidden_omega_0).to(DEVICE)
    optims["siren"] = torch.optim.Adam(lr=ARGS.siren_lr, params=models["siren"].parameters(),
                                       weight_decay=ARGS.siren_wd)           
    schedulers["siren"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optims["siren"], patience=5, verbose=True)
    
    if not ARGS.share_mapping: 
        models["pcmra_mapping"] = load_mapping(ARGS).to(DEVICE)
        optims["pcmra_mapping"] = torch.optim.Adam(lr=ARGS.pcmra_mapping_lr, params=models["pcmra_mapping"].parameters())
        schedulers["pcmra_mapping"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optims["pcmra_mapping"], patience=5, verbose=True)
    
    if ARGS.reconstruction == "pcmra" or ARGS.reconstruction == "both":
        models["pcmra_siren"] = Siren(ARGS, in_features=3, out_features=1, 
                                      first_omega_0=ARGS.pcmra_first_omega_0, 
                                      hidden_omega_0=ARGS.pcmra_hidden_omega_0,
                                      final_activation=None).to(DEVICE)
        optims["pcmra_siren"] = torch.optim.Adam(lr=ARGS.pcmra_siren_lr, params=models["pcmra_siren"].parameters(),
                                                 weight_decay=ARGS.pcmra_siren_wd)
        schedulers["pcmra_siren"] = torch.optim.lr_scheduler.ReduceLROnPlateau(optims["pcmra_siren"], patience=5, verbose=True)
    

    for model, struct in models.items(): 
        print(model.upper())
        if ARGS.print_models:
            print(struct)

    return models, optims, schedulers


# #####  Random coords subsample

# In[3]:


def choose_random_coords(*arrays, n=1000): 
    
    if not n == -1:
        mx = arrays[0].shape[1]
        rand_idx = random.sample(range(mx), n)
    
        arrays = [array.detach().clone()[:, rand_idx, :] for array in arrays]
    
    return arrays


# #### Dice loss

# In[4]:


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

# In[5]:


def train_epoch(dataloader, models, optims, schedulers, criterions, batch_count, ARGS):
    mask_losses = []
    pcmra_losses = []
    combined_losses = []
    
    for _, _, _, pcmra, coords, pcmra_array, mask_array in dataloader:
        #choose random coords
        siren_in, pcmra_labels, mask_labels = choose_random_coords(coords, pcmra_array, 
                                                                   mask_array, n=ARGS.n_coords_sample)

        # get latent representations
        latent_rep = models["cnn"](pcmra)   
        
        # get reconstructions 
        if ARGS.reconstruction == "pcmra":
            if ARGS.share_mapping: 
                gamma, beta = models["mapping"](latent_rep)
                
                mask_out = models["siren"](siren_in, gamma.detach(), beta.detach())
                pcmra_out = models["pcmra_siren"](siren_in, gamma, beta)
            
            if not ARGS.share_mapping: 
                gamma, beta = models["mapping"](latent_rep.detach())
                mask_out = models["siren"](siren_in, gamma, beta)

                pcmra_gamma, pcmra_beta = models["pcmra_mapping"](latent_rep)
                pcmra_out = models["pcmra_siren"](siren_in, pcmra_gamma, pcmra_beta)
        
        if ARGS.reconstruction == "both":
            if ARGS.share_mapping: 
                gamma, beta = models["mapping"](latent_rep)
                
                mask_out = models["siren"](siren_in, gamma, beta)
                pcmra_out = models["pcmra_siren"](siren_in, gamma, beta)
            
            if not ARGS.share_mapping: 
                gamma, beta = models["mapping"](latent_rep)
                mask_out = models["siren"](siren_in, gamma, beta)

                pcmra_gamma, pcmra_beta = models["pcmra_mapping"](latent_rep)
                pcmra_out = models["pcmra_siren"](siren_in, pcmra_gamma, pcmra_beta)
                
        if ARGS.reconstruction == "mask":
            gamma, beta = models["mapping"](latent_rep)
                
            mask_out = models["siren"](siren_in, gamma, beta)

        #calculate losses
        mask_loss = criterions[0](mask_out, mask_labels) 
        
        if ARGS.reconstruction == "pcmra" or ARGS.reconstruction == "both":
            pcmra_loss = criterions[1](pcmra_out, pcmra_labels)
            loss = ARGS.mask_lambda * mask_loss + ARGS.pcmra_lambda * pcmra_loss
            
        else: 
            pcmra_loss = torch.Tensor([0])
            loss = ARGS.mask_lambda * mask_loss
        
        mask_losses.append(mask_loss.item())
        pcmra_losses.append(pcmra_loss.item())
        combined_losses.append(loss.item())

        loss.backward()

        for _, optim in optims.items():
            optim.step()
            optim.zero_grad()
    
    for _, scheduler in schedulers.items():
        if ARGS.scheduler_on == "pcmra":
            scheduler.step(np.mean(pcmra_losses))
        elif ARGS.scheduler_on == "combined":
            scheduler.step(np.mean(combined_losses))
        elif ARGS.scheduler_on == "mask":
            scheduler.step(np.mean(mask_losses))  
        else: 
            raise(Exception("Choose valid scheduler_on for ARGS."))
    
    mean, std = round(np.mean(mask_losses), 6), round(np.std(mask_losses), 6)
    p_mean, p_std = round(np.mean(pcmra_losses), 6), round(np.std(pcmra_losses), 6)
    
    return mean, std, p_mean, p_std, batch_count


def val_epoch(dataloader, models, criterions, ARGS, n_eval=100):
    with torch.no_grad():
        mask_losses = []
        pcmra_losses = []
        d_losses = []

        i = 0

        for idx, subj, proj, pcmra, coords, pcmra_array, mask_array in dataloader:    
            mask_out = get_complete_image(models, pcmra, coords, ARGS)
            
            mask_loss = criterions[0](mask_out, mask_array)  
            mask_losses.append(mask_loss.item())

            d_loss = calc_dice_loss(mask_out, mask_array) 
            d_losses.append(d_loss.item())
            
            if ARGS.reconstruction != "mask": 
                pcmra_out = get_complete_image(models, pcmra, coords, ARGS, output="pcmra")
                
                pcmra_loss = criterions[1](pcmra_out, pcmra_array)                  
                pcmra_losses.append(pcmra_loss.item())
            
            else: 
                pcmra_losses.append(0)
                

            i += ARGS.batch_size
            if i > n_eval:
                break    

        m_loss_mean, m_loss_std = round(np.mean(mask_losses), 6), round(np.std(mask_losses), 6)
        p_loss_mean, p_loss_std = round(np.mean(pcmra_losses), 6), round(np.std(pcmra_losses), 6)
        d_loss_mean, d_loss_std = round(np.mean(d_losses), 6), round(np.std(d_losses), 6)
    
    return m_loss_mean, m_loss_std, p_loss_mean, p_loss_std, d_loss_mean, d_loss_std


def get_complete_image(models, pcmra, coords, ARGS, val_n=10000, output="mask"): 
    for model in models.values(): 
        model.eval() #evaluation mode    
        
    latent_rep = models["cnn"](pcmra)
                
    n_slices = math.ceil(coords.shape[1] / val_n) # number of batches
    for i in range(n_slices):
        coords_in = coords[:, (i*val_n) : ((i+1)*val_n), :]
        
        if output == "mask":
            gamma, beta = models["mapping"](latent_rep)
            siren_out = models["siren"](coords_in, gamma, beta)
        
        elif output == "pcmra": 
            if ARGS.share_mapping: 
                gamma, beta = models["mapping"](latent_rep)
            if not ARGS.share_mapping:
                gamma, beta = models["pcmra_mapping"](latent_rep)
                
            siren_out = models["pcmra_siren"](coords_in, gamma, beta)
            
        if i == 0: 
            image = siren_out.detach()
        else:
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


# #### Load CNN and Mapping setup

# In[1]:


def load_cnn(ARGS): 
    
    if ARGS.cnn_setup == 1: 
        cnn = CNN1()
    elif ARGS.cnn_setup == 2: 
        cnn = CNN2()
    else: 
        raise(Exception("Choose existing CNN setup"))
        
    return cnn

def load_mapping(ARGS): 
    
    if ARGS.mapping_setup == 1: 
        mapping = Mapping1()
    elif ARGS.mapping_setup == 2: 
        mapping = Mapping2()
    elif ARGS.mapping_setup == 3: 
        mapping = Mapping3()
    elif ARGS.mapping_setup == 4: 
        mapping = Mapping4()
    else: 
        raise(Exception("Choose existing mapping setup"))
        
    return mapping


# #### Function to scroll through output

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


# In[ ]:





# In[1]:


print("Loaded all helper functions.")

