import torch
import numpy as np

import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset


####################################################################################
########################### Mapping Layer Classes ##################################
####################################################################################

class ReshapeTensor(nn.Module):
    def __init__(self, size): 
        super(ReshapeTensor, self).__init__()
        self.size = size
                
    def forward(self, input):
        return input.reshape([input.shape[0]] + self.size)
        

class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)

####################################################################################
########################### Final Mapping setups ###################################
####################################################################################

########################################
########### Mapping_Golden #############
########################################

class Mapping_Golden(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_Golden, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        if ARGS.cnn_setup == "deep": 
            first_layer = 1024
        else: 
            first_layer = 6144

        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )    
        
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :self.n_gammas, :], out[:, self.n_gammas:, :]


########################################
########### Mapping_Small ##############
########################################

class Mapping_Small(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_Small, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(1024, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(1024, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
            
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :self.n_gammas, :], out[:, self.n_gammas:, :]


########################################
######## Mapping_SingleNetwork #########
########################################

class Mapping_SingleNetwork(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_SingleNetwork, self).__init__()
        
        self.dim_hidden = ARGS.dim_hidden
        
        if ARGS.cnn_setup == "deep": 
            first_layer = 1024
        else: 
            first_layer = 6144
            
        self.model = nn.Sequential(Flatten(), nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                   nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                   nn.Linear(512, 2 * self.dim_hidden), ReshapeTensor([2, self.dim_hidden]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


########################################
######## Mapping_SepGammaAndBeta #######
########################################

class Mapping_SepGammaAndBeta(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_SepGammaAndBeta, self).__init__()
        
        self.dim_hidden = ARGS.dim_hidden

        if ARGS.cnn_setup == "deep": 
            first_layer = 1024
        else: 
            first_layer = 6144

        self.gamma_model = nn.Sequential(Flatten(), nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                         nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                         nn.Linear(512, self.dim_hidden))
        
        self.beta_model = nn.Sequential(Flatten(), nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                        nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                        nn.Linear(512, self.dim_hidden))
        
    def forward(self, x):
        gamma = self.gamma_model(x)
        beta =  self.beta_model(x)
        
        return gamma, beta


########################################
######## Mapping_SepEachLayer ##########
########################################

class Mapping_SepEachLayer(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_SepEachLayer, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        if ARGS.cnn_setup == "deep": 
            first_layer = 1024
        else: 
            first_layer = 6144

        self.models = nn.ModuleList()
        for i in range(self.n_gammas):
            self.models.append(nn.Sequential(Flatten(), 
                                      nn.Linear(first_layer, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, 2 * self.dim_hidden), ReshapeTensor([2, self.dim_hidden])
                                       )
                         )
                
    def forward(self, x):
        gammas = torch.empty(0).to(x.device)
        betas = torch.empty(0).to(x.device)
        
        for model in self.models: 
            gammas = torch.cat((gammas, model(x)[:, 0, :].unsqueeze(1)), 1)
            betas = torch.cat((betas, model(x)[:, 1, :].unsqueeze(1)), 1)
        
        
        return gammas, betas