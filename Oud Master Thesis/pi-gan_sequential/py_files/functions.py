import torch 
from datetime import datetime
from pathlib import Path
import os


def set_device(ARGS):
    if ARGS.device.lower() == "cpu": 
        DEVICE = torch.device("cpu")
    else: 
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('----------------------------------')
    print('Using device for training:', DEVICE)
    print('----------------------------------')

    return DEVICE 


def get_folder(ARGS): 
    now = datetime.now()
    dt = now.strftime("%Y_%m_%d_%H_%M_%S")
    
    if ARGS.name != "": 
        ARGS.name = f"_{ARGS.name}"
        
    path = os.path.join("saved_runs", f"pi_gan_{dt}{ARGS.name}")
    
    Path(f"{path}").mkdir(parents=True, exist_ok=True)   

    print(path)

    return path