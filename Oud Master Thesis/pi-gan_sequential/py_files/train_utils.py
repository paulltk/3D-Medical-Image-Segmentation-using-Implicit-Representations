import torch
import math
import numpy as np

from py_files.data_utils import *
from py_files.loss_utils import *


def train_model(dataloader, models, optims, schedulers, criterion, blur_layer, ARGS, output="pcmra"): 
    losses = [] 
    
    for batch in dataloader:
                    
        batch = transform_batch(batch, ARGS)            
        _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = get_siren_batch(batch, blur_layer, ARGS.n_coords_sample, ARGS)
        
        latent_rep = models["cnn"](pcmra) # get latent representation
        
        if output == "pcmra": 
            siren_in, labels = coords_array, pcmra_array
            
            if ARGS.pcmra_train_cnn:
                gamma, beta = models["pcmra_mapping"](latent_rep)
                model_keys = ["cnn", "pcmra_mapping", "pcmra_siren"]
                
            else:
                gamma, beta = models["pcmra_mapping"](latent_rep.detach())
                model_keys = ["pcmra_mapping", "pcmra_siren"]
            
            out, _ = models["pcmra_siren"](siren_in, gamma, beta)            
            
        elif output == "mask": 
            siren_in, labels = coords_array, mask_array

            if ARGS.mask_train_cnn:
                gamma, beta = models["mapping"](latent_rep)
                model_keys = ["cnn", "mapping", "siren"]
                
            else:
                gamma, beta = models["mapping"](latent_rep.detach())
                model_keys = ["mapping", "siren"]
            
            out, coords_bp = models["siren"](siren_in, gamma, beta)
            
        if not ARGS.sdf: 
            loss = criterion(out, labels)  
        elif ARGS.sdf: 
            loss = criterion(coords_bp, out, surface_array, norm_array)

        losses.append(loss.item())
        loss.backward()
        
        for key in model_keys: 
            optims[key].step()
            optims[key].zero_grad()
    
    mean, std = round(np.mean(losses), 6), round(np.std(losses), 6)

    for key in model_keys: 
        schedulers[key].step(mean)
    
    return mean, std


def val_model(dataloader, models, criterion, blur_layer, ARGS, output="pcmra"):
    with torch.no_grad():
        losses = []
        d_losses = []
        
        for batch in dataloader:
                    
            _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = get_siren_batch(batch, blur_layer, -1, ARGS)
    
            labels = [pcmra_array if output=="pcmra" else mask_array][0]
            
            out, _ = get_complete_image(models, pcmra, coords_array, ARGS, output=output)
            
            if ARGS.sdf: 
                out = 1 - F.sigmoid(out)
                
            for s_out, s_labels in zip(out, labels):
            
                loss = criterion(s_out, s_labels)  
                losses.append(loss.item())
            
                if output=="mask":
                    d_loss = calc_dice_loss(s_out, s_labels) 
                else:
                    d_loss = torch.tensor([0])

                d_losses.append(d_loss.item())

        loss_mean, loss_std = round(np.mean(losses), 6), round(np.std(losses), 6)
        d_loss_mean, d_loss_std = round(np.mean(d_losses), 6), round(np.std(d_losses), 6)
    
    return loss_mean, loss_std, d_loss_mean, d_loss_std


def get_complete_image(models, pcmra, coords, ARGS, val_n=10000, output="mask"): 
    with torch.no_grad():
        for model in models.values(): 
            model.eval()  # evaluation mode    
        
        n_slices = math.ceil(coords.shape[1] / val_n) # number of batches
        
        coords_backprop = None

        latent_rep = models["cnn"](pcmra)                
        
        for i in range(n_slices):
            coords_in = coords[:, (i*val_n) : ((i+1)*val_n), :]
            
            if output == "mask":
                gamma, beta = models["mapping"](latent_rep)
                siren_out, coords_out = models["siren"](coords_in, gamma, beta)
            
            elif output == "pcmra": 
                gamma, beta = models["pcmra_mapping"](latent_rep)    
                siren_out, _ = models["pcmra_siren"](coords_in, gamma, beta)
            
            if i == 0: 
                # if ARGS.sdf: 
                #     coords_backprop = coords_out
                # if not ARGS.sdf: 
                #     siren_out = siren_out.detach()
                # image = siren_out
                image = siren_out.detach()
                
            else:
                # if ARGS.sdf: 
                #     coords_backprop = torch.cat((coords_backprop, coords_out), 1)
                # if not ARGS.sdf: 
                #     siren_out = siren_out.detach()
                # image = torch.cat((image, siren_out), 1)
                image = torch.cat((image, siren_out.detach()), 1)
        
        for model in models.values(): 
            model.train()  # train mode
    
    return image, coords_backprop