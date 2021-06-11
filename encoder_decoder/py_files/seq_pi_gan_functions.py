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

from kornia.augmentation.augmentation3d import *
from kornia.geometry.transform import *


# #### ARGS class for .ipynb files

# In[2]:


class init_ARGS(object): 
    def __init__(self): 
        self.device = "GPU"
        self.print_models = False
        self.name = ""
        self.pretrained = None
        self.pretrained_best = "train"
        self.dataset = "small"
        self.rotate = True
        self.translate = True
        self.flip = True
        self.crop = True
        self.stretch = True
        self.norm_min_max = [0, 1]
        self.seed = 34
        self.epochs = 501
        self.batch_size = 24
        self.eval_every = 20
        self.shuffle = True
        self.n_coords_sample = 5000
        self.cnn_setup = 0
        self.mapping_setup = 0
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
        self.patience = 20
        self.train_cnn = True

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

    
def save_loss(path, loss, models, optims, name="loss", save_models=True):
    np.save(f"{path}/{name}.npy", loss)
    
    eps, t_loss, v_loss = loss[:, 0], loss[:, 1], loss[:, 3]
    
    print(f"{name.ljust(15)} Train: {round(t_loss[-1], 5)}, \t Eval: {round(v_loss[-1], 5)}")
    
    if t_loss[-1] == t_loss.min(): 
        print(f"New best train loss, saving model.")
        if save_models:
            for model in models.keys():
                torch.save(models[model].state_dict(), f"{path}/{model}_train.pt")
                torch.save(optims[model].state_dict(), f"{path}/{model}_optim_train.pt")

    if v_loss[-1] == v_loss.min(): 
        print(f"New best eval  loss, saving model.")
        if save_models:
            for model in models.keys():
                torch.save(models[model].state_dict(), f"{path}/{model}_val.pt")
                torch.save(optims[model].state_dict(), f"{path}/{model}_optim_val.pt")
        
    plot_graph(path, eps, [(t_loss, "Train loss"), (v_loss, "Eval loss")], 
               axes=("Epochs", "Loss"), fig_name=f"{name}_plot")


# #### Initialize dataloaders

# In[5]:


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
#     print(train_ds.all_images[:10])
    
    val_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="val")
    val_dl = DataLoader(val_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Val subjects:", val_ds.__len__())
    
    test_ds =  SirenDataset(root, val_subjects, DEVICE, dataset="test")
    test_dl = DataLoader(test_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Test subjects:", test_ds.__len__())

    return train_dl, val_dl, test_dl


# #### Image transformations

# In[1]:


##### TRANSLATE FUNCTIONS #####

def get_random_shift(max_t=(4, 8, 8)):
        shifts = (random.randint(-max_t[0], max_t[0]), 
                  random.randint(-max_t[1], max_t[1]), 
                  random.randint(-max_t[2], max_t[2]))
        return shifts
     
def translate_image(image, shifts):
    
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

def translate_batch(batch): 
    idx, subj, proj, pcmras, masks = batch
    
    new_pcmras = new_masks = torch.empty((0)).to(pcmras.device)
    
    for pcmra, mask in zip(pcmras, masks): 
        pcmra, mask = pcmra.squeeze(), mask.squeeze()
        
        shifts = get_random_shift()
        pcmra = translate_image(pcmra, shifts).unsqueeze(0).unsqueeze(0)
        mask = translate_image(mask, shifts).unsqueeze(0).unsqueeze(0)

        new_pcmras = torch.cat((new_pcmras, pcmra), 0)
        new_masks = torch.cat((new_masks, mask), 0)
    
    return idx, subj, proj, new_pcmras, new_masks
    
##### FLIP FUNCTION #####

def flip_batch(batch): 
    d_flip, h_flip, v_flip = RandomDepthicalFlip3D(), RandomHorizontalFlip3D(), RandomVerticalFlip3D()
    
    idx, subj, proj, pcmra, mask = batch
    pcmra_mask = torch.cat((pcmra, mask), 1)
    
    pcmra_mask = d_flip(pcmra_mask)
    pcmra_mask = h_flip(pcmra_mask)
    pcmra_mask = v_flip(pcmra_mask)
    
    pcmra, mask = pcmra_mask.split(1, dim=1)

    return idx, subj, proj, pcmra, mask
    
##### ROTATION FUNCTION #####

def rotate_batch(batch):
    rotate = RandomRotation3D((10., 15., 15.), p=1.0)
    
    idx, subj, proj, pcmra, mask = batch
    pcmra_mask = torch.cat((pcmra, mask), 1)
    
    pcmra_mask = rotate(pcmra_mask)
    pcmra, mask = pcmra_mask.split(1, dim=1)
    
    return idx, subj, proj, pcmra, mask

##### CROP FUNCTIONS #####

def crop_batch(batch, stretch=True):
    idx, subj, proj, pcmras, masks = batch
    
    orig_shape = pcmras.shape[2:]
    
    crop_sample = RandomCrop3D(orig_shape, p=1.)
    
    rand = random.uniform
    inc = 1.4
    if stretch:
        resize = [rand(1., inc), rand(1., inc), rand(1., inc)]
    else: 
        resize = [rand(1., inc)] * 3
    
    size = tuple([int(i * j) for i, j in zip(orig_shape, resize)])
    
    pcmras = F.interpolate(pcmras, size=size, mode="trilinear")
    masks = F.interpolate(masks, size=size, mode="trilinear")
    
    pcmra_masks = torch.cat((pcmras, masks), 1)
    pcmra_masks = crop_sample(pcmra_masks)

    pcmras, masks = pcmra_masks.split(1, dim=1)

    return idx, subj, proj, pcmras, masks


##### complete transformation #####

def transform_batch(batch, ARGS): 
    
    if ARGS.rotate: 
        batch = rotate_batch(batch)
    if ARGS.translate: 
        batch = translate_batch(batch)
    if ARGS.crop: 
        batch = crop_batch(batch, ARGS.stretch)
    if ARGS.flip: 
        batch = flip_batch(batch)
    
    return batch


# #### Create siren arrays 

# In[ ]:


def get_siren_batch(batch): 
    idx, subj, proj, pcmras, masks = batch

    length = prod(pcmras.shape[2:])
    
    coords = get_coords(*pcmras.shape[2:]).to(pcmras.device).unsqueeze(0)
#     print("c", coords.shape)
    coords = coords.repeat(pcmras.shape[0], 1, 1)
    
    pcmra_array = pcmras.view(pcmras.shape[0], length, 1)
    mask_array = masks.view(pcmras.shape[0], length, 1)
    
    return idx, subj, proj, pcmras, coords, pcmra_array, mask_array

def prod(val) :  
    res = 1 
    for ele in val:  
        res *= ele  
    return res 

    
def get_coords(*sidelengths):
    tensors = []

    for sidelen in sidelengths:
        tensors.append(torch.linspace(-1, 1, steps=sidelen))

    tensors = tuple(tensors)
    coords = torch.stack(torch.meshgrid(*tensors), dim=-1)

    return coords.reshape(-1, len(sidelengths))


# #### Initialize models

# In[2]:


def cnn_model_optim_scheduler(ARGS, DEVICE): 
    model = load_cnn(ARGS).to(DEVICE)
    optim = torch.optim.Adam(lr=ARGS.cnn_lr, params=model.parameters(), weight_decay=ARGS.cnn_wd)
    scheduler= torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, verbose=True, min_lr=1e-5)
    
    return model, optim, scheduler


def mapping_model_optim_scheduler(ARGS, lr, DEVICE):
    model= load_mapping(ARGS).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, verbose=True, min_lr=1e-5)

    return model, optim, scheduler
    

def siren_model_optim_scheduler(ARGS, first_omega_0, hidden_omega_0, lr, wd, final_activation, DEVICE):
    model = Siren(ARGS, in_features=3, out_features=1,first_omega_0=first_omega_0, 
                            hidden_omega_0=hidden_omega_0, final_activation=final_activation).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=wd)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, verbose=True, min_lr=1e-5)
    
    return model, optim, scheduler


def load_models_and_optims(ARGS):
    
    if ARGS.device.lower() == "cpu": 
        DEVICE = torch.device("cpu")
        
        print('----------------------------------')
        print('Using device for training:', DEVICE)
        print('----------------------------------')

    else: 
        DEVICE = set_device()
        
    models, optims, schedulers = {}, {}, {}
    
    models["cnn"], optims["cnn"], schedulers["cnn"] = cnn_model_optim_scheduler(ARGS, DEVICE)
    
    models["mapping"], optims["mapping"],         schedulers["mapping"] = mapping_model_optim_scheduler(ARGS, ARGS.mapping_lr, DEVICE)
    
    
    models["siren"], optims["siren"], schedulers["siren"] =         siren_model_optim_scheduler(ARGS, ARGS.first_omega_0, ARGS.hidden_omega_0, 
                                    ARGS.siren_lr, ARGS.siren_wd, "sigmoid", DEVICE)
    
    models["pcmra_mapping"], optims["pcmra_mapping"],         schedulers["pcmra_mapping"] = mapping_model_optim_scheduler(ARGS, ARGS.pcmra_mapping_lr, DEVICE)
   
    models["pcmra_siren"], optims["pcmra_siren"], schedulers["pcmra_siren"] =         siren_model_optim_scheduler(ARGS, ARGS.pcmra_first_omega_0, ARGS.pcmra_hidden_omega_0, 
                                    ARGS.pcmra_siren_lr, ARGS.pcmra_siren_wd, None, DEVICE)

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

# In[ ]:


def train_model(dataloader, models, optims, schedulers, criterion, ARGS, output="pcmra"): 
    losses = [] 
    
    for batch in dataloader:
                    
        batch = transform_batch(batch, ARGS)            
        _, _, _, pcmra, coords, pcmra_array, mask_array = get_siren_batch(batch)
        
        latent_rep = models["cnn"](pcmra) # get latent representation
        
        if output == "pcmra": 
            siren_in, labels = choose_random_coords(coords, pcmra_array, n=ARGS.n_coords_sample)
            
            if ARGS.train_cnn:
                gamma, beta = models["pcmra_mapping"](latent_rep)
                model_keys = ["cnn", "pcmra_mapping", "pcmra_siren"]
                
            else:
                gamma, beta = models["pcmra_mapping"](latent_rep.detach())
                model_keys = ["pcmra_mapping", "pcmra_siren"]
            
            out = models["pcmra_siren"](siren_in, gamma, beta)            
            
        elif output == "mask": 
            siren_in, labels = choose_random_coords(coords, mask_array, n=ARGS.n_coords_sample)
            gamma, beta = models["mapping"](latent_rep.detach())
            out = models["siren"](siren_in, gamma, beta)
            
            model_keys = ["mapping", "siren"]
            
        #calculate losses
        loss = criterion(out, labels) 
        losses.append(loss.item())
        loss.backward()
        
        for key in model_keys: 
            optims[key].step()
            optims[key].zero_grad()
    
    mean, std = round(np.mean(losses), 5), round(np.std(losses), 5)

    for key in model_keys: 
        if optims[key].param_groups[0]["lr"] > 1e-5:
            schedulers[key].step(mean)
    
    return mean, std


# In[21]:


def val_model(dataloader, models, criterion, ARGS, output="pcmra", n_eval=100):
    with torch.no_grad():
        losses = []
        d_losses = []
        
        i = 0
        
        for batch in dataloader:
                    
            _, _, _, pcmra, coords, pcmra_array, mask_array = get_siren_batch(batch)

            i += pcmra.shape[0]
            
            labels = [pcmra_array if output=="pcmra" else mask_array][0]
            
            out = get_complete_image(models, pcmra, coords, ARGS, output=output)
            
            loss = criterion(out, labels)  
            losses.append(loss.item())
            
            if output=="mask":
                d_loss = calc_dice_loss(out, labels) 
            else:
                d_loss = torch.tensor([0])

            d_losses.append(d_loss.item())
                
            if i > n_eval:
                break    

        loss_mean, loss_std = round(np.mean(losses), 5), round(np.std(losses), 5)
        d_loss_mean, d_loss_std = round(np.mean(d_losses), 5), round(np.std(d_losses), 5)
    
    return loss_mean, loss_std, d_loss_mean, d_loss_std


# In[22]:


def get_complete_image(models, pcmra, coords, ARGS, val_n=10000, output="mask"): 
    for model in models.values(): 
        model.eval()  # evaluation mode    
    
    n_slices = math.ceil(coords.shape[1] / val_n) # number of batches
    
    latent_rep = models["cnn"](pcmra)                
    
    for i in range(n_slices):
        coords_in = coords[:, (i*val_n) : ((i+1)*val_n), :]
        
        if output == "mask":
            gamma, beta = models["mapping"](latent_rep)
            siren_out = models["siren"](coords_in, gamma, beta)
        
        elif output == "pcmra": 
            gamma, beta = models["pcmra_mapping"](latent_rep)    
            siren_out = models["pcmra_siren"](coords_in, gamma, beta)
        
        if i == 0: 
            image = siren_out.detach()
        else:
            image = torch.cat((image, siren_out.detach()), 1)
    
    for model in models.values(): 
        model.train()  # train mode
    
    return image 


# #### Load pretrained models

# In[8]:


def load_pretrained_models(folder, best, models, optims, pretrained_models=None): 
    path = f"saved_runs/{folder}/"

    for key in models.keys():
        if pretrained_models == None or key in pretrained_models:
            if os.path.exists(f"{path}/{key}_{best}.pt"):
                print(f"Loading params from {key}")
                models[key].load_state_dict(torch.load(f"{path}/{key}_{best}.pt"))
                optims[key].load_state_dict(torch.load(f"{path}/{key}_optim_{best}.pt"))


# #### Load CNN and Mapping setup

# In[1]:


def load_cnn(ARGS): 
    
    if ARGS.cnn_setup == 1: 
        cnn = CNN1()
    elif ARGS.cnn_setup == 2: 
        cnn = CNN2()
    elif ARGS.cnn_setup == 0: 
        cnn = Encoder()
    elif ARGS.cnn_setup == 3: 
        cnn = Encoder_1()
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
    elif ARGS.mapping_setup == 0: 
        mapping = Encoder_Mapping()
    elif ARGS.mapping_setup == 5: 
        mapping = Encoder_Mapping_1()
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

