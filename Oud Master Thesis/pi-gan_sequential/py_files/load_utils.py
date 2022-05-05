import torch
import os
import numpy as np
import random

from torch.utils.data import Dataset, DataLoader

from py_files.dataset import *
from py_files.cnn_model import *
from py_files.pigan_model import *

##################################################
############# Initialize dataloader ##############
##################################################

def initialize_dataloaders(ARGS, DEVICE):
        
    assert(ARGS.dataset in ["full", "small", "new"])
    
    
    folder = {"small": "scaled_normalized", "full": "original_normalized", 
              "new": "new_original"}

    root = os.path.join(os.path.abspath('..'), "Dataset", folder[ARGS.dataset])
            
    subjects = [file.split("__")[:2] for file in  sorted(os.listdir(root))]
    subjects = np.array(sorted([list(subj) for subj in list(set(map(tuple, subjects)))]))
    
    idx = list(range(subjects.shape[0]))
    split1, split2 = int(len(idx) * 0.6), int(len(idx) * 0.8)
    
    random.seed(ARGS.seed)
    random.shuffle(idx) # shuffles indices
    train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:] # incides per data subset

    train_subjects, val_subjects, test_subjects =  subjects[train_idx], subjects[val_idx], subjects[test_idx]

    train_ds = SirenDataset(root, train_subjects, DEVICE)
    train_dl = DataLoader(train_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=ARGS.shuffle)
    print("Train subjects:", train_ds.__len__())
    
    val_ds =  SirenDataset(root, val_subjects, DEVICE)
    val_dl = DataLoader(val_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Val subjects:", val_ds.__len__())
    
    test_ds =  SirenDataset(root, test_subjects, DEVICE)
    test_dl = DataLoader(test_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)

    print("Test subjects:", test_ds.__len__())

    return train_dl, val_dl, test_dl


##################################################
############### Initialize models ################
##################################################


def cnn_model_optim_scheduler(ARGS, DEVICE): 
    model = load_cnn(ARGS).to(DEVICE)
    optim = torch.optim.Adam(lr=ARGS.cnn_lr, params=model.parameters(), weight_decay=ARGS.cnn_wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, 
                                                           factor=.5, verbose=True, min_lr=ARGS.min_lr)
    
    return model, optim, scheduler


def mapping_model_optim_scheduler(ARGS, lr, DEVICE):
    model= load_mapping(ARGS).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, factor=.5, 
                                                           verbose=True, min_lr=ARGS.min_lr)

    return model, optim, scheduler
    

def siren_model_optim_scheduler(ARGS, first_omega_0, hidden_omega_0, lr, wd, final_activation, DEVICE):
    model = Siren(ARGS, in_features=3, out_features=1,first_omega_0=first_omega_0, 
                            hidden_omega_0=hidden_omega_0, final_activation=final_activation).to(DEVICE)
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=wd)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=ARGS.patience, factor=.5, 
                                                           verbose=True, min_lr=ARGS.min_lr)
    
    return model, optim, scheduler


def initialize_models_and_optims(ARGS, DEVICE):
        
    models, optims, schedulers = {}, {}, {}
    
    models["cnn"], optims["cnn"], schedulers["cnn"] = cnn_model_optim_scheduler(ARGS, DEVICE)
    
    models["mapping"], optims["mapping"], schedulers["mapping"] = \
        mapping_model_optim_scheduler(ARGS, ARGS.mapping_lr, DEVICE)
    
    models["siren"], optims["siren"], schedulers["siren"] = \
        siren_model_optim_scheduler(ARGS, ARGS.first_omega_0, ARGS.hidden_omega_0, 
                                    ARGS.siren_lr, ARGS.siren_wd, ARGS.mask_siren_final_activation, DEVICE)
    
    models["pcmra_mapping"], optims["pcmra_mapping"], schedulers["pcmra_mapping"] = \
        mapping_model_optim_scheduler(ARGS, ARGS.pcmra_mapping_lr, DEVICE)
   
    models["pcmra_siren"], optims["pcmra_siren"], schedulers["pcmra_siren"] = \
        siren_model_optim_scheduler(ARGS, ARGS.pcmra_first_omega_0, ARGS.pcmra_hidden_omega_0,
                                    ARGS.pcmra_siren_lr, ARGS.pcmra_siren_wd, None, DEVICE)

    for model, struct in models.items(): 
        print(model.upper())
        if ARGS.print_models:
            print(struct)

    return models, optims, schedulers

##################################################
################## Load models ###################
##################################################

def load_from_saved_run(models, optims, DEVICE, ARGS):
    if ARGS.pretrained: 
        print(f"Loading pretrained model from '{ARGS.pretrained}'.")
        load_pretrained_models(ARGS.pretrained, ARGS.pretrained_best_dataset, ARGS.pretrained_best_loss,
                    models, optims, DEVICE, pretrained_models = ARGS.pretrained_models)
    
        if ARGS.pretrained_lr_reset:
            for name, optim in optims.items():
                for param_group in optim.param_groups: 
                    param_group["lr"] = ARGS.pretrained_lr_reset
                print(f"{name} lr reset to: {optim.param_groups[0]['lr']}")


def load_pretrained_models(folder, best_dataset, best_loss, models, optims, DEVICE, pretrained_models=None): 
    path = f"saved_runs/{folder}/"

    for key in models.keys():
        if pretrained_models == None or key in pretrained_models:
            if os.path.exists(f"{path}/{key}_{best_loss}_loss_{best_dataset}.pt"):
                print(f"Loading params from {key}")
                models[key].load_state_dict(torch.load(f"{path}/{key}_{best_loss}_loss_{best_dataset}.pt", 
                                            map_location=torch.device(DEVICE)))
                optims[key].load_state_dict(torch.load(f"{path}/{key}_optim_{best_loss}_loss_{best_dataset}.pt", 
                                            map_location=torch.device(DEVICE)))

                
##################################################
######### Choose a CNN and mapping setup #########
##################################################

def load_cnn(ARGS): 
   
    if ARGS.cnn_setup == -1: 
        cnn = LargeCNN1()
    elif ARGS.cnn_setup == -2: 
        cnn = LargeCNN()
    elif ARGS.cnn_setup == -3: 
        cnn = LargeCNN3()
    elif ARGS.cnn_setup == -4: 
        cnn = LargeCNN4()
    elif ARGS.cnn_setup == -5: 
        cnn = LargeCNN5()
    elif ARGS.cnn_setup == -6: 
        cnn = LargeCNN6()
    elif ARGS.cnn_setup == -20: 
        cnn = LargeCNN20()

        
    elif ARGS.cnn_setup == 0: 
        cnn = Encoder()
    elif ARGS.cnn_setup == 1: 
        cnn = CNN1()
    elif ARGS.cnn_setup == 2: 
        cnn = CNN2()
    elif ARGS.cnn_setup == 3: 
        cnn = Encoder_1()
    elif ARGS.cnn_setup == 4: 
        cnn = Encoder_2()
    elif ARGS.cnn_setup == 5: 
        cnn = CNN3()     
    elif ARGS.cnn_setup == 6: 
        cnn = CNN4()
    elif ARGS.cnn_setup == 7: 
        cnn = CNN5()
    elif ARGS.cnn_setup == 8: 
        cnn = CNN6()
    elif ARGS.cnn_setup == 9: 
        cnn = CNN7()
    elif ARGS.cnn_setup == 10: 
        cnn = CNN8()
    elif ARGS.cnn_setup == 11: 
        cnn = CNN9()
    elif ARGS.cnn_setup == 12: 
        cnn = CNN10()
    elif ARGS.cnn_setup == 13: 
        cnn = CNN11()
    elif ARGS.cnn_setup == 14: 
        cnn = CNN12()
    elif ARGS.cnn_setup == 15: 
        cnn = CNN13()
    elif ARGS.cnn_setup == 16: 
        cnn = CNN14()
    elif ARGS.cnn_setup == 17: 
        cnn = CNN15()
    elif ARGS.cnn_setup == 18: 
        cnn = CNN16()
    else: 
        raise(Exception("Choose existing CNN setup"))
        
    return cnn


def load_mapping(ARGS): 
    
    if ARGS.mapping_setup == -1 or ARGS.mapping_setup == 7: 
        mapping = LargeMapping1(ARGS)
    elif ARGS.mapping_setup == -2: 
        mapping = LargeMapping2(ARGS)
    elif ARGS.mapping_setup == -5: 
        mapping = LargeMapping5(ARGS)
    
    elif ARGS.mapping_setup == 1: 
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
    elif ARGS.mapping_setup == 6: 
        mapping = Encoder_Mapping_2()
    elif ARGS.mapping_setup == 8: 
        mapping = Encoder_Mapping_4()
    else: 
        raise(Exception("Choose existing Mapping setup"))
        
    return mapping