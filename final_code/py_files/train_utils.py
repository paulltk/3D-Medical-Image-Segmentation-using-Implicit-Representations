import torch
import torch.nn as nn
import math
import numpy as np
import time
import wandb

from py_files.load_utils import *
from py_files.data_utils import *
from py_files.loss_utils import *
from py_files.save_utils import *


#################################################
############## Complete training ################
#################################################

def complete_training(ARGS): 
    if ARGS.training_setup == "reconstruction":
        reconstruction(ARGS)
    elif ARGS.training_setup == "segmentation":
        segmentation(ARGS)
    elif ARGS.training_setup == "combined":
        combined(ARGS)
    elif ARGS.training_setup == "consecutively":
        consecutively(ARGS)
    else: 
        raise(Exception("Choose a correct training setup."))


def reconstruction(ARGS): 
    
    # Log in to your W&B account
    wandb.login()

    wandb.init(project=f"{ARGS.training_setup}_{ARGS.segmentation}",
               name=f"{ARGS.name}_{ARGS.cnn_setup}_{ARGS.mapping_setup}_{ARGS.first_omega_0}", 
               config=vars(ARGS))
    
    path, DEVICE = initialize_path_and_device(ARGS)   
    
    ##### data preparation #####
    train_dl, val_dl, test_dl = initialize_dataloaders(ARGS, DEVICE)
            
    ##### initialize models and optimizers #####
    models, optims, schedulers = initialize_models_and_optims(ARGS, DEVICE)
    
    blur_layer = initialize_blurring_layer(1.0, DEVICE)

    ##### load pretrained model #####
    load_from_saved_run(models, optims, DEVICE, ARGS)
    
    ##### epoch, train loss mean, train loss std, val loss mean, val loss std #####
    mask_losses, pcmra_losses, dice_losses = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
    
    for ep in range(ARGS.pcmra_epochs):
        t = time.time() 

        for model in models.values():
            model.train()

        loss, _ = train_reconstruction(train_dl, models, optims, schedulers, blur_layer, ARGS)
        
        wandb.log({"epoch_sampling_mse_loss": loss, "epochs": ep}, step=ep)
    
        if ep % 10 == 0: 
            print(f"Epoch {ep}, train loss {loss.round(6)}")

        if ep % ARGS.eval_every == 0: 

            print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")

            pcmra_losses, _ = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, pcmra_losses, None, "pcmra", ARGS) 
            
            wandb.log({"train_mse_loss": pcmra_losses[-1, 1], 
                "eval_mse_loss": pcmra_losses[-1, 3], 
                "epochs": ep}, step=ep)

            # check if training improved last 200 epochs
            if (pcmra_losses.shape[0] - np.argmin(pcmra_losses[:, 1])) > (200 / ARGS.eval_every) and \
            optims["cnn"].param_groups[0]['lr'] == ARGS.min_lr:
                break

    losses = test_epoch(test_dl, models, blur_layer, ARGS, DEVICE)

    wandb.log({"test_mse_loss": losses[0], 
        "test_bce_loss": losses[2], 
        "test_dice_loss": losses[4]}, step=ep)


    wandb.finish()


def segmentation(ARGS):
    
    again = True
    while again == True: 
        again = False
        # Log in to your W&B account
        wandb.login()
        wandb.init(project=f"{ARGS.training_setup}_{ARGS.segmentation}",
               name=f"{ARGS.name}_{ARGS.cnn_setup}_{ARGS.mapping_setup}_{ARGS.first_omega_0}", 
               config=vars(ARGS))
    
        path, DEVICE = initialize_path_and_device(ARGS)   
        
        ##### data preparation #####
        train_dl, val_dl, test_dl = initialize_dataloaders(ARGS, DEVICE)
                
        ##### initialize models and optimizers #####
        models, optims, schedulers = initialize_models_and_optims(ARGS, DEVICE)
        
        blur_layer = initialize_blurring_layer(1.0, DEVICE)
        
        ##### load pretrained model #####
        load_from_saved_run(models, optims, DEVICE, ARGS)
        
        ##### epoch, train loss mean, train loss std, val loss mean, val loss std #####
        mask_losses, pcmra_losses, dice_losses = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
        
        for ep in range(ARGS.mask_epochs):
            t = time.time() 
    
            for model in models.values():
                model.train()
    
            loss, _ = train_segmentation(train_dl, models, optims, schedulers, blur_layer, ARGS)
            
            wandb.log({"epoch_sampling_bce_loss": loss, "epochs": ep}, step=ep)
    
            if ep % 10 == 0: 
                print(f"Epoch {ep}, train loss {loss.round(6)}")
    
            if ep % ARGS.eval_every == 0: 
                
                print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
                
                mask_losses, dice_losses = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, mask_losses, dice_losses, "mask", ARGS)
                
                wandb.log({"train_bce_loss": mask_losses[-1, 1], 
                    "eval_bce_loss": mask_losses[-1, 3], 
                    "train_dice_loss": dice_losses[-1, 1], 
                    "eval_dice_loss": dice_losses[-1, 3], 
                    "epochs": ep}, step=ep)

                if ARGS.pretrained == None and ep == 0 and dice_losses[0, 1] < 1: 
                    again = True
                    break 
                
                # check if training improved last 200 epochs
                if ARGS.segmentation == "binary" and \
                (mask_losses.shape[0] - np.argmin(mask_losses[:, 1])) > (200 / ARGS.eval_every)  and \
                optims["cnn"].param_groups[0]['lr'] == ARGS.min_lr:
                    break
                if ARGS.segmentation == "sdf" and \
                (dice_losses.shape[0] - np.argmin(dice_losses[:, 1])) > (200 / ARGS.eval_every) and \
                optims["cnn"].param_groups[0]['lr'] == ARGS.min_lr:
                    break
        
        
        losses = test_epoch(test_dl, models, blur_layer, ARGS, DEVICE)

        wandb.log({"test_mse_loss": losses[0], 
            "test_bce_loss": losses[2], 
            "test_dice_loss": losses[4]}, step=ep)

        wandb.finish()    
        
def combined(ARGS): 
    
    again = True
    while again == True: 
        again = False

        # Log in to your W&B account
        wandb.login()
        wandb.init(project=f"{ARGS.training_setup}_{ARGS.segmentation}",
                   name=f"{ARGS.name}_{ARGS.cnn_setup}_{ARGS.mapping_setup}_{ARGS.first_omega_0}", 
                   config=vars(ARGS))
        
        path, DEVICE = initialize_path_and_device(ARGS)   
        
        ##### data preparation #####
        train_dl, val_dl, test_dl = initialize_dataloaders(ARGS, DEVICE)
                
        ##### initialize models and optimizers #####
        models, optims, schedulers = initialize_models_and_optims(ARGS, DEVICE)
        
        blur_layer = initialize_blurring_layer(1.0, DEVICE)
        
        ##### load pretrained model #####
        load_from_saved_run(models, optims, DEVICE, ARGS)
        
        ##### epoch, train loss mean, train loss std, val loss mean, val loss std #####
        mask_losses, pcmra_losses, dice_losses = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
        
        for ep in range(ARGS.pcmra_epochs):
            t = time.time() 

            for model in models.values():
                model.train()

            loss, _ = train_combined(train_dl, models, optims, schedulers, blur_layer, ARGS)
            
            wandb.log({"epoch_sampling_combined_loss": loss, 
                "epochs": ep}, step=ep)

            if ep % 10 == 0: 
                print(f"Epoch {ep}, train loss {loss.round(6)}")
                
            if ep % ARGS.eval_every == 0: 
        
                print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
                
                pcmra_losses, _ = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, pcmra_losses, None, "pcmra", ARGS)
                mask_losses, dice_losses =validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, mask_losses, dice_losses, "mask", ARGS)
                
                wandb.log({"train_mse_loss": pcmra_losses[-1, 1], 
                    "eval_mse_loss": pcmra_losses[-1, 3],
                    "train_bce_loss": mask_losses[-1, 1], 
                    "eval_bce_loss": mask_losses[-1, 3], 
                    "train_dice_loss": dice_losses[-1, 1], 
                    "eval_dice_loss": dice_losses[-1, 3], 
                    "epochs": ep}, step=ep)

                if ep == 0 and dice_losses[0, 1] < 1: 
                    again = True
                    break 
                
                # check if training improved last 200 epochs
                if ARGS.segmentation == "binary" and (mask_losses.shape[0] - np.argmin(mask_losses[:, 1])) > (200 / ARGS.eval_every) and \
                (pcmra_losses.shape[0] - np.argmin(pcmra_losses[:, 1])) > (200 / ARGS.eval_every) and \
                optims["cnn"].param_groups[0]['lr'] == ARGS.min_lr:
                    break
                if ARGS.segmentation == "sdf" and (dice_losses.shape[0] - np.argmin(dice_losses[:, 1])) > (200 / ARGS.eval_every) and \
                (pcmra_losses.shape[0] - np.argmin(pcmra_losses[:, 1])) > (200 / ARGS.eval_every) and \
                optims["cnn"].param_groups[0]['lr'] == ARGS.min_lr:
                    break

        losses = test_epoch(test_dl, models, blur_layer, ARGS, DEVICE)

        wandb.log({"test_mse_loss": losses[0], 
            "test_bce_loss": losses[2], 
            "test_dice_loss": losses[4]}, step=ep)

        wandb.finish()    


def consecutively(ARGS): 
    
    # Log in to your W&B account
    wandb.login()
    wandb.init(project=f"{ARGS.training_setup}_{ARGS.segmentation}",
               name=f"{ARGS.name}_{ARGS.cnn_setup}_{ARGS.mapping_setup}_{ARGS.first_omega_0}", 
               config=vars(ARGS))
    
    path, DEVICE = initialize_path_and_device(ARGS)   
    
    ##### data preparation #####
    train_dl, val_dl, test_dl = initialize_dataloaders(ARGS, DEVICE)
            
    ##### initialize models and optimizers #####
    models, optims, schedulers = initialize_models_and_optims(ARGS, DEVICE)
    
    blur_layer = initialize_blurring_layer(1.0, DEVICE)
    
    ##### load pretrained model #####
    load_from_saved_run(models, optims, DEVICE, ARGS)
    
    ##### epoch, train loss mean, train loss std, val loss mean, val loss std #####
    mask_losses, pcmra_losses, dice_losses = np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 5))
    
    for ep in range(1):
        t = time.time() 

        for model in models.values():
            model.train()

        # loss, _ = train_reconstruction(train_dl, models, optims, schedulers, ARGS)
        
        # wandb.log({"epoch_sampling_mse_loss": loss, 
        #     "epochs": ep}, step=ep)

        # if ep % 10 == 0: 
        #     print(f"Epoch {ep}, train loss {loss.round(6)}")

        if ep % ARGS.eval_every == 0: 
            print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")
            
            pcmra_losses, _ = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, pcmra_losses, None, "pcmra", ARGS)

            wandb.log({"train_mse_loss": pcmra_losses[-1, 1], 
                "eval_mse_loss": pcmra_losses[-1, 3], 
                "epochs": ep}, step=ep)

            print("This was a test epoch for reconstruction.")

    
    again = True
    while again == True: 
        again = False
    
        for ep in range(ARGS.mask_epochs):
            t = time.time() 

            for model in models.values():
                model.train()

            loss, _ = train_segmentation(train_dl, models, optims, schedulers, blur_layer, ARGS)
            
            if ep == 0: 
                mask_losses, dice_losses = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, mask_losses, dice_losses, "mask", ARGS)     
                
                if dice_losses[0, 1] < 1: 

                    ##### initialize models and optimizers #####
                    new_models, new_optims, new_schedulers = initialize_models_and_optims(ARGS, DEVICE)

                    models["mapping"] = new_models["mapping"]
                    models["siren"] = new_models["siren"]
                    optims["mapping"] = new_optims["mapping"]
                    optims["siren"] = new_optims["siren"]
                    schedulers["mapping"] = new_schedulers["mapping"]
                    schedulers["siren"] = new_schedulers["siren"]
                    
                    mask_losses, dice_losses =  np.empty((0, 5)), np.empty((0, 5))
                    
                    again = True

                    break 

            wandb.log({"epoch_sampling_bce_loss": loss, 
                "epochs": ep}, step=ep)

            if ep % 10 == 0: 
                print(f"Epoch {ep}, train loss {loss.round(6)}")

            if ep % ARGS.eval_every == 0: 

                print(f"Epoch {ep} took {round(time.time() - t, 2)} seconds.")

                mask_losses, dice_losses = validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, mask_losses, dice_losses, "mask", ARGS)     

                wandb.log({"train_bce_loss": mask_losses[-1, 1], 
                    "eval_bce_loss": mask_losses[-1, 3], 
                    "train_dice_loss": dice_losses[-1, 1], 
                    "eval_dice_loss": dice_losses[-1, 3], 
                    "epochs": ep}, step=ep)

                # check if training improved last 200 epochs
                if ARGS.segmentation == "binary" and (mask_losses.shape[0] - np.argmin(mask_losses[:, 1])) > (200 / ARGS.eval_every)  and \
                optims["siren"].param_groups[0]['lr'] == ARGS.min_lr:
                    break
                if ARGS.segmentation == "sdf" and (dice_losses.shape[0] - np.argmin(dice_losses[:, 1])) > (200 / ARGS.eval_every) and \
                optims["siren"].param_groups[0]['lr'] == ARGS.min_lr:
                    break

    losses = test_epoch(test_dl, models, blur_layer, ARGS, DEVICE)

    wandb.log({"test_mse_loss": losses[0], 
        "test_bce_loss": losses[2], 
        "test_dice_loss": losses[4]}, step=ep)

    wandb.finish()    

            

#################################################
################ Single epoch ###################
#################################################

def train_reconstruction(dataloader, models, optims, schedulers, blur_layer, ARGS): 
    
    mse_criterion = nn.MSELoss()

    losses = [] 
    
    for batch in dataloader:
                    
        batch = transform_batch(batch, ARGS)            
        _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = \
            get_siren_batch(batch, blur_layer, ARGS.n_coords_sample, ARGS)
        
        model_keys = ["cnn", "pcmra_mapping", "pcmra_siren"]

        # Forward pass
        latent_rep = models["cnn"](pcmra) # get latent representation 
        
        # print("latent_rep", latent_rep.shape)
        gamma, beta = models["pcmra_mapping"](latent_rep)
        # print("gamma", gamma.shape)
        out, _ = models["pcmra_siren"](coords_array, gamma, beta)            
        # print("out", out.shape)
        # Backward pass
        loss = mse_criterion(out, pcmra_array)  
        # print("Calculated loss.")
        loss.backward()
        # print("Did backprop.")
        losses.append(loss.item())
        
        for key in model_keys: 
            optims[key].step()
            optims[key].zero_grad()
    
    mean, std = round(np.mean(losses), 6), round(np.std(losses), 6)

    for key in model_keys: 
        schedulers[key].step(mean)
        
    return mean, std


def train_segmentation(dataloader, models, optims, schedulers, blur_layer, ARGS): 
    
    bce_criterion = nn.BCELoss()

    losses = [] 
    
    for batch in dataloader:
                    
        batch = transform_batch(batch, ARGS)            
        _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = get_siren_batch(batch, blur_layer, ARGS.n_coords_sample, ARGS)
        
        model_keys = ["cnn", "mapping", "siren"]

        # Forward pass
        latent_rep = models["cnn"](pcmra) # get latent representation 
        if ARGS.train_encoder_seg: 
            gamma, beta = models["mapping"](latent_rep)
        else:
            gamma, beta = models["mapping"](latent_rep.detach())
            
        out, coords_backprop = models["siren"](coords_array, gamma, beta)            
        
        # Backward pass
        if ARGS.segmentation.lower() == "binary": 
            loss = bce_criterion(out, mask_array)  
        elif ARGS.segmentation.lower() == "sdf": 
            loss = sdf_criterion(coords_backprop, out, surface_array, norm_array, ARGS)
        loss.backward()

        losses.append(loss.item())
        
        for key in model_keys: 
            optims[key].step()
            optims[key].zero_grad()
    
    mean, std = round(np.mean(losses), 6), round(np.std(losses), 6)

    for key in model_keys: 
        schedulers[key].step(mean)
    
    return mean, std


def train_combined(dataloader, models, optims, schedulers, blur_layer, ARGS): 
    
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()

    losses = [] 
    
    for batch in dataloader:
                    
        batch = transform_batch(batch, ARGS)            
        _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = get_siren_batch(batch, blur_layer, ARGS.n_coords_sample, ARGS)
        
        model_keys = ["cnn", "mapping", "siren", "pcmra_mapping", "pcmra_siren"]

        # Forward pass
        latent_rep = models["cnn"](pcmra) # get latent representation 
        
        r_gamma, r_beta = models["pcmra_mapping"](latent_rep)
        r_out, _ = models["pcmra_siren"](coords_array, r_gamma, r_beta)            
        
        if ARGS.train_encoder_seg: 
            s_gamma, s_beta = models["mapping"](latent_rep)
        else:
            s_gamma, s_beta = models["mapping"](latent_rep.detach())
        s_out, coords_backprop = models["siren"](coords_array, s_gamma, s_beta)            
        
        # Backward pass
        r_loss = mse_criterion(r_out, pcmra_array)  
        
        if ARGS.segmentation.lower() == "binary": 
            s_loss = bce_criterion(s_out, mask_array)  
        elif ARGS.segmentation.lower() == "sdf": 
            s_loss = sdf_criterion(coords_backprop, s_out, surface_array, norm_array, ARGS)
            
        loss = r_loss + ARGS.seg_loss_lambda * s_loss
        loss.backward()

        losses.append(loss.item())
        
        for key in model_keys: 
            optims[key].step()
            optims[key].zero_grad()
    
    mean, std = round(np.mean(losses), 6), round(np.std(losses), 6)

    for key in model_keys: 
        schedulers[key].step(mean)
    
    return mean, std


##################################################
############### Test functions ###################
##################################################

def test_epoch(test_dl, models, blur_layer, ARGS, DEVICE): 
    
    print("Testing models.")
    
    print("Loading best pcmra models.")
    for key in models.keys():
        if os.path.exists(f"{ARGS.path}/{key}_pcmra_loss_val.pt"):
            print(f"Loading params from {key}")
            models[key].load_state_dict(torch.load(f"{ARGS.path}/{key}_pcmra_loss_val.pt", 
                                        map_location=torch.device(DEVICE)))
            
    pcmra_mean, pcmra_std, _, _ = val_model(test_dl, models, nn.MSELoss(), blur_layer, ARGS, output="pcmra")
    
    print("Loading best mask models.")
    for key in models.keys():
        if os.path.exists(f"{ARGS.path}/{key}_dice_loss_val.pt"):
            print(f"Loading params from {key}")
            models[key].load_state_dict(torch.load(f"{ARGS.path}/{key}_dice_loss_val.pt", 
                                        map_location=torch.device(DEVICE)))
    
    mask_mean, mask_std, dice_mean, dice_std = val_model(test_dl, models, nn.BCELoss(), blur_layer, ARGS, output="mask")
    
    losses = np.array([pcmra_mean, pcmra_std, mask_mean, mask_std, dice_mean, dice_std])

    print(losses)

    np.save(f"{ARGS.path}/test_results.npy", losses)
    
    return losses

##################################################
########### Validation functions #################
##################################################

def validation_epoch(ep, train_dl, val_dl, models, optims, blur_layer, losses, dice_losses, output, ARGS): 
        
    if output == "pcmra":         
        
        t_pcmra_mean, t_pcmra_std, _, _ = \
            val_model(train_dl, models, nn.MSELoss(), blur_layer, ARGS, output=output)
        v_pcmra_mean, v_pcmra_std, _, _ = \
            val_model(val_dl, models, nn.MSELoss(), blur_layer, ARGS, output=output)

        losses = np.append(losses, [[ep ,t_pcmra_mean, t_pcmra_std, 
                                     v_pcmra_mean, v_pcmra_std]], axis=0)

        save_loss(ARGS.path, losses, models, optims, name="pcmra_loss", 
                  save_models=True)
        
    elif output == "mask":
        
        t_mask_mean, t_mask_std, t_dice_mean, t_dice_std = \
            val_model(train_dl, models, nn.BCELoss(), blur_layer, ARGS, output=output)
        v_mask_mean, v_mask_std, v_dice_mean, v_dice_std = \
            val_model(val_dl, models, nn.BCELoss(), blur_layer, ARGS, output=output)

        losses = np.append(losses, [[ep ,t_mask_mean, t_mask_std, 
                                     v_mask_mean, v_mask_std]], axis=0)
        dice_losses = np.append(dice_losses, [[ep ,t_dice_mean, t_dice_std, 
                                     v_dice_mean, v_dice_std]], axis=0)

        save_loss(ARGS.path, losses, models, optims, name="mask_loss", 
                  save_models=ARGS.save_models)
        save_loss(ARGS.path, dice_losses, models, optims, name="dice_loss", 
                  save_models=ARGS.save_models)
    
    return losses, dice_losses


def val_model(dataloader, models, criterion, blur_layer, ARGS, output="pcmra"):
    with torch.no_grad():
        losses = []
        d_losses = []
        
        for batch in dataloader:
                    
            _, _, _, pcmra, coords_array, pcmra_array, mask_array, surface_array, norm_array = get_siren_batch(batch, blur_layer, -1, ARGS)
    
            labels = [pcmra_array if output=="pcmra" else mask_array][0]
            
            out = get_complete_image(models, pcmra, coords_array, ARGS, output=output)
            
            if ARGS.segmentation.lower() == "sdf" and output=="mask": 
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
                image = siren_out.detach()   
            else:
                image = torch.cat((image, siren_out.detach()), 1)
        
        for model in models.values(): 
            model.train()  # train mode
    
    return image