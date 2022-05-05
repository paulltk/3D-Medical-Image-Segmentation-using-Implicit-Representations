import os
import numpy as np
import torch 

from torch.utils.data import Dataset, DataLoader

class SirenDataset(Dataset): 
    def __init__(self, root, subjects, DEVICE): 
        self.root = root
        self.DEVICE = DEVICE
             
        self.all_images = [image.split("__")[:3] for image in os.listdir(root) 
                           if list(image.split("__")[:2]) in subjects.tolist() 
                           and image.split("__")[3] == "pcmra.npy"]
        
        self.pcmras = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{size}__pcmra.npy")) 
                           for subj, proj, size in self.all_images]).float().to(DEVICE)

        self.masks = torch.tensor([np.load(os.path.join(self.root, f"{subj}__{proj}__{size}__mask.npy")) 
                           for subj, proj, size in self.all_images]).float().to(DEVICE)

        torch.clamp(self.masks, min=0, max=1)

            
    def __len__(self):
        return len(self.all_images)

    
    def __getitem__(self, idx):
        subj, proj, size = self.all_images[idx]
        pcmra = self.pcmras[idx]
        mask = self.masks[idx]
        
        pcmra = pcmra.permute(2, 0, 1).unsqueeze(0)
        mask = mask.permute(2, 0, 1).unsqueeze(0)
        loss_cover = torch.ones(mask.shape).to(self.DEVICE)

        return idx, subj, proj, pcmra, mask, loss_cover