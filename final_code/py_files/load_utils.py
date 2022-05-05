import torch
import os
import numpy as np
import random
import warnings

from torch.utils.data import Dataset, DataLoader

from py_files.dataset import *
from py_files.cnn_models import *
from py_files.mapping_models import *
from py_files.pigan_model import *
from py_files.functions import *


def initialize_path_and_device(ARGS): 
    warnings.filterwarnings("ignore")
    
    ##### path to wich the model should be saved #####
    path = get_folder(ARGS)
    DEVICE = set_device(ARGS)
    
    with open(os.path.join(path, "ARGS.txt"), "w") as f:
        print(vars(ARGS), file=f)
        
    return path, DEVICE
    

##################################################
############# Initialize dataloader ##############
##################################################

def initialize_dataloaders(ARGS, DEVICE):
        
    root = os.path.join(os.path.abspath('..'), "Dataset", "new_original")
            
    subjects = [file.split("__")[:2] for file in  sorted(os.listdir(root))]
    subjects = np.array(sorted([list(subj) for subj in list(set(map(tuple, subjects)))]))
    
    with open("py_files/dataset_split.txt") as file:
        splits = file.readlines()
        splits = [[i for i in line.rstrip().split(", ")] for line in splits]
        
    train_subjects = np.array([subject for subject in subjects if subject[0] in splits[0]])
    val_subjects = np.array([subject for subject in subjects if subject[0] in splits[1]])
    test_subjects = np.array([subject for subject in subjects if subject[0] in splits[2]])

    # print(train_subjects)
    # idx = list(range(subjects.shape[0]))
    # split1, split2 = int(len(idx) * 0.6), int(len(idx) * 0.8)
    
    # random.seed(ARGS.seed)
    # random.shuffle(idx) # shuffles indices
    # train_idx, val_idx, test_idx = idx[:split1], idx[split1:split2], idx[split2:] # incides per data subset

    # train_subjects, val_subjects, test_subjects =  subjects[train_idx], subjects[val_idx], subjects[test_idx]

    train_ds = SirenDataset(root, train_subjects, DEVICE)
    train_dl = DataLoader(train_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=ARGS.shuffle)
    print("Train subjects:", train_ds.__len__())
    print("Train batch:", next(iter(train_dl))[1][:5])

    val_ds =  SirenDataset(root, val_subjects, DEVICE)
    val_dl = DataLoader(val_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)
    print("Val subjects:", val_ds.__len__())
    print("Val batch:", next(iter(val_dl))[1][:5])

    
    test_ds =  SirenDataset(root, test_subjects, DEVICE)
    test_dl = DataLoader(test_ds, batch_size=ARGS.batch_size, num_workers=0, shuffle=False)

    print("Test subjects:", test_ds.__len__())
    print("Test batch:", next(iter(test_dl))[1][:5])

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


def mapping_model_optim_scheduler(mapping_setup, ARGS, lr, DEVICE):
    model= load_mapping(ARGS, mapping_setup).to(DEVICE)
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
    
    if ARGS.segmentation == "binary": 
        final_activation = "sigmoid" 
    else: 
        final_activation = None
        
    models["cnn"], optims["cnn"], schedulers["cnn"] = cnn_model_optim_scheduler(ARGS, DEVICE)
    
    models["mapping"], optims["mapping"], schedulers["mapping"] = \
        mapping_model_optim_scheduler(ARGS.mapping_setup, ARGS, ARGS.mapping_lr, DEVICE)
    
    models["siren"], optims["siren"], schedulers["siren"] = \
        siren_model_optim_scheduler(ARGS, ARGS.first_omega_0, ARGS.hidden_omega_0, 
                                    ARGS.siren_lr, ARGS.siren_wd, final_activation, DEVICE)
    
    try: 
        ARGS.pcmra_mapping_setup
    except AttributeError:
        models["pcmra_mapping"], optims["pcmra_mapping"], schedulers["pcmra_mapping"] = \
            mapping_model_optim_scheduler(ARGS.mapping_setup, ARGS, ARGS.pcmra_mapping_lr, DEVICE)
    else:
        models["pcmra_mapping"], optims["pcmra_mapping"], schedulers["pcmra_mapping"] = \
            mapping_model_optim_scheduler(ARGS.pcmra_mapping_setup, ARGS, ARGS.pcmra_mapping_lr, DEVICE)  
    
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
   
    if ARGS.cnn_setup == "golden": 
        cnn = CNN_Golden()

    elif ARGS.cnn_setup == "maxpool": 
        cnn = CNN_MaxPool()

    elif ARGS.cnn_setup == "deep": 
        cnn = CNN_Deep()

    elif ARGS.cnn_setup == "batchnorm": 
        cnn = CNN_BatchNorm()

    else: 
        raise(Exception("Choose existing CNN setup"))

    return cnn


def load_mapping(ARGS, mapping_setup): 

    if mapping_setup == "golden": 
        mapping = Mapping_Golden(ARGS)

    elif mapping_setup == "golden_deep": 
        mapping = Mapping_Small(ARGS)

    elif mapping_setup == "1net": 
        mapping = Mapping_SingleNetwork(ARGS)

    elif mapping_setup == "2net": 
        mapping = Mapping_SepGammaAndBeta(ARGS) 

    elif mapping_setup == "4net": 
        mapping = Mapping_SepEachLayer(ARGS)

    else: 
        raise(Exception("Choose existing Mapping setup"))

    return mapping