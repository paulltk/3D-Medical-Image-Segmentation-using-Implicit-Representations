import torch
import numpy as np

import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


####################################################################################
############################# CNN Layer Classes ####################################
####################################################################################

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride, padding, 
                 activation="relu", max_pool=None, layer_norm=None, batch_norm=False):
        
        super(ConvLayer, self).__init__()
        
        net = [nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding)]
        
        if activation == "relu":
            net.append(nn.ReLU())
        
        elif activation == "leakyrelu": 
            net.append(nn.LeakyReLU())
            
        if layer_norm: 
            net.append(nn.LayerNorm(layer_norm)) # add layer normalization
        
        if batch_norm: 
            net.append(nn.BatchNorm3d(out_channels)) # add batch normalization
        
        if max_pool: 
            net.append(nn.MaxPool3d(max_pool)) # add max_pooling
        
        self.model = nn.Sequential(*net)
        
    def forward(self, input):
        out = self.model(input)
        return out
    

class Flatten(nn.Module):
    """ Reshapes a 4d matrix to a 2d matrix. """
    def forward(self, input):
        return input.view(input.size(0), -1)
        

class ReshapeTensor(nn.Module):
    def __init__(self, size): 
        super(ReshapeTensor, self).__init__()
        self.size = size
                
    def forward(self, input):
        return input.reshape([input.shape[0]] + self.size)


####################################################################################
############################# Final CNN setups #####################################
####################################################################################


########################################
############# CNN_Golden ###############
######################################## 

class CNN_Golden(nn.Module):

    def __init__(self):
        
        super(CNN_Golden, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),    
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
############# CNN_Maxpool ##############
######################################## 

class CNN_Maxpool(nn.Module):

    def __init__(self):
        
        super(CNN_Maxpool, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu",
                max_pool=2),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=2),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=1, padding=2, activation="relu",
                max_pool=2), 
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
########### CNN_Extralayer #############
######################################## 

class CNN_Extralayer(nn.Module):

    def __init__(self):
        
        super(CNN_Extralayer, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu",
                layer_norm=(3, 4, 4)),    

            ConvLayer(128, 256, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(256, 256, kernel_size=5, stride=2, padding=2, activation="relu"),    
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
########### CNN_BatchNorm ##############
######################################## 

class CNN_BatchNorm(nn.Module):

    def __init__(self):
        
        super(CNN_BatchNorm, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                max_pool=(1, 2, 2), batch_norm=True),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                batch_norm=True),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),    
        )

    def forward(self, x):
        out = self.model(x)
            
        return out

####################################################################################
########################### Final Mapping setups ###################################
####################################################################################

########################################
########## Mapping_Golden #############
########################################

class Mapping_Golden(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_Golden, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024), nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024), nn.LeakyReLU(.2),
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
########### Mapping_1Network ###########
########################################

class Mapping_1Network(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_1Network, self).__init__()
        
        self.dim_hidden = ARGS.dim_hidden
        
        self.model = nn.Sequential(nn.Linear(6144, 1024), nn.LeakyReLU(.2),
                                   nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                   nn.Linear(512, 2 * self.dim_hidden), ReshapeTensor([2, self.dim_hidden]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


########################################
########### Mapping_2Networks ###########
########################################

class Mapping_2Networks(nn.Module):

    def __init__(self, ARGS):
        super(Mapping_2Networks, self).__init__()
        
        self.dim_hidden = ARGS.dim_hidden

        self.gamma_model = nn.Sequential(nn.Linear(6144, 1024), nn.LeakyReLU(.2),
                                         nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                         nn.Linear(512, self.dim_hidden))
        
        self.beta_model = nn.Sequential(nn.Linear(6144, 1024), nn.LeakyReLU(.2),
                                        nn.Linear(1024, 512), nn.LeakyReLU(.2),
                                        nn.Linear(512, self.dim_hidden))
        
    def forward(self, x):
        gamma = self.gamma_model(x)
        beta =  self.beta_model(x)
        
        return gamma, beta
















####################################################################################
############################## Earlier setups ######################################
####################################################################################


########################################
############ cnn_setup -1 ##############
######################################## 

class LargeCNN1(nn.Module):

    def __init__(self):
        
        super(LargeCNN1, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
########################################
############ cnn_setup -3 ##############
######################################## 

class LargeCNN3(nn.Module):

    def __init__(self):
        
        super(LargeCNN3, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   4,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(4,   4,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(4,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,  8,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(8,  16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out

########################################
############ cnn_setup -4 ##############
######################################## 

class LargeCNN4(nn.Module):

    def __init__(self):
        
        super(LargeCNN4, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1, 1))
        
    def forward(self, x):
        
        out = F.interpolate(x, size=(12, 32, 32), mode='trilinear')    
        
        return out
    

########################################
############ cnn_setup -5 ##############
######################################## 

class LargeCNN5(nn.Module):

    def __init__(self):
        
        super(LargeCNN5, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(64,  128,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(128, 256, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(256, 256, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup -6 ##############
######################################## 

class LargeCNN6(nn.Module):

    def __init__(self):
        
        super(LargeCNN6, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[ ]:


########################################
########## mapping_setup -1 #############
########################################

class LargeMapping1(nn.Module):

    def __init__(self, ARGS):
        super(LargeMapping1, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
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
########## mapping_setup -2 #############
########################################

class LargeMapping2(nn.Module):

    def __init__(self, ARGS):
        super(LargeMapping2, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(12288, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(12288, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
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
########## mapping_setup -5 #############
########################################

class LargeMapping5(nn.Module):

    def __init__(self, ARGS):
        super(LargeMapping5, self).__init__()
        
        self.n_gammas = ARGS.siren_hidden_layers + 1
        self.dim_hidden = ARGS.dim_hidden
        
        self.gammas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(12288, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, self.dim_hidden), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(self.n_gammas):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(12288, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 512), 
                                      nn.LeakyReLU(.2),
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


# ## Combi 1

# In[ ]:


class CNN1(nn.Module):

    def __init__(self):
        
        super(CNN1, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(2, 4, 4)),
            
            Flatten(),
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[ ]:


class Mapping1(nn.Module):

    def __init__(self):
        super(Mapping1, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features=4096, out_features=2048, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(in_features=2048, out_features=1024, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(1024, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[ ]:


class Mapping2(nn.Module):

    def __init__(self):
        super(Mapping2, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(in_features=4096, out_features=2048, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(in_features=2048, out_features=1024, bias=True),
                                   nn.LeakyReLU(.2),
                                   
                                   nn.Linear(1024, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
                
        return out[:, :4, :], out[:, 4:, :]


# ## Combi 2 

# In[ ]:


class CNN2(nn.Module):

    def __init__(self):
        
        super(CNN2, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(12, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(6, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", layer_norm=(3, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", layer_norm=(2, 4, 4)),
            
            Flatten(),
            
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.LeakyReLU(.2),

            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.LeakyReLU(.2),
         )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[ ]:


class Mapping3(nn.Module):

    def __init__(self):
        super(Mapping3, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]


# In[ ]:


class Mapping4(nn.Module):

    def __init__(self):
        super(Mapping4, self).__init__()
        
        self.model = nn.Sequential(nn.Linear(1024, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, :4, :], out[:, 4:, :]


# ## Encoder

# In[ ]:


class Encoder(nn.Module):

    def __init__(self):
        
        super(Encoder, self).__init__()
        
        self.model = nn.Sequential(            
            ConvLayer(1,    16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      batch_norm=True, max_pool=(1, 2, 2)),
            
            ConvLayer(16,   32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,   32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(32,   64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,   64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(64,   128,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1, activation="relu"), 
            
            ConvLayer(128,   512,  kernel_size=(3, 4, 4), stride=1, padding=0)
            )

    def forward(self, x):
        out = self.model(x)
            
        return out

class Encoder_1(nn.Module):

    def __init__(self):
        
        super(Encoder_1, self).__init__()
        
        self.model = nn.Sequential(            
            ConvLayer(1,    16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      batch_norm=True, max_pool=(1, 2, 2)),
            
            ConvLayer(16,   32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,   32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(32,   64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,   64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(64,   128,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1, activation="relu",
                      batch_norm=True),
            
            ConvLayer(128,   512,  kernel_size=(3, 4, 4), stride=1, padding=0)
            )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

class Encoder_2(nn.Module):

    def __init__(self):
        
        super(Encoder_2, self).__init__()
        
        self.model = nn.Sequential(            
            ConvLayer(1,    16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      batch_norm=True, max_pool=(1, 2, 2)),
            
            ConvLayer(16,   32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,   32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(32,   64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,   64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      batch_norm=True),
            
            ConvLayer(64,   128,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128,  128,  kernel_size=3, stride=2, padding=1), 
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

class Encoder_Mapping(nn.Module):

    def __init__(self):
        super(Encoder_Mapping, self).__init__()
        
        self.model = nn.Sequential(Flatten(), 
                                   nn.Linear(512, 512),
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 2048), 
                                   ReshapeTensor([8, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, :4, :], out[:, 4:, :]
    
class Encoder_Mapping_1(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_1, self).__init__()
        
        self.model = nn.Sequential(Flatten(), 
                                   nn.Linear(512, 512),
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   nn.LeakyReLU(.2),
                                   nn.Linear(512, 512), 
                                   ReshapeTensor([2, 256]))
        
    def forward(self, x):
        out = self.model(x)  
        
        return out[:, 0, :], out[:, 1, :]
    


# # Currently used 

# ### CNN

# #### Output 128, 3, 4, 4

# In[ ]:


########################################
############ cnn_setup 5 ###############
########################################

class CNN3(nn.Module):

    def __init__(self):
        
        super(CNN3, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


########################################
############ cnn_setup 13 ##############
########################################

class CNN11(nn.Module):

    def __init__(self):
        
        super(CNN11, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup 6 ###############
########################################
class CNN4(nn.Module):

    def __init__(self):
        
        super(CNN4, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out

    
########################################
############ cnn_setup 7 ###############
########################################    
class CNN5(nn.Module):

    def __init__(self):
        
        super(CNN5, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
########################################
############ cnn_setup 8 ###############
######################################## 

class CNN6(nn.Module):

    def __init__(self):
        
        super(CNN6, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=3, stride=1, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
########################################
############ cnn_setup 9 ###############
######################################## 

class CNN7(nn.Module):

    def __init__(self):
        
        super(CNN7, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(16,  16,  kernel_size=3, stride=1, padding=1, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(32,  32,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(64,  64,  kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=3, stride=1, padding=1, activation="relu"),
            ConvLayer(128, 128, kernel_size=3, stride=2, padding=1, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


    
########################################
############ cnn_setup 10 ##############
######################################## 

class CNN8(nn.Module):

    def __init__(self):
        
        super(CNN8, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup 14 ##############
######################################## 

class CNN12(nn.Module):

    def __init__(self):
        
        super(CNN12, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
    
########################################
############ cnn_setup 15 ##############
######################################## 

class CNN13(nn.Module):

    def __init__(self):
        
        super(CNN13, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu"),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
########################################
############ cnn_setup 16 ##############
######################################## 

class CNN14(nn.Module):

    def __init__(self):
        
        super(CNN14, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

########################################
############ cnn_setup 17 ##############
######################################## 

class CNN15(nn.Module):

    def __init__(self):
        
        super(CNN15, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(2, 2, 2), layer_norm=(24, 32, 32)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    
    
########################################
############ cnn_setup 18 ##############
######################################## 

class CNN16(nn.Module):

    def __init__(self):
        
        super(CNN16, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=(1, 2, 2), padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# In[ ]:


########################################
############ cnn_setup -2 ##############
######################################## 

class LargeCNN(nn.Module):

    def __init__(self):
        
        super(LargeCNN, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   8,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(8,   8,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 128, 128)),
            
            ConvLayer(8,   16,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu"),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# #### Output 512, 1, 1, 1

# In[ ]:


########################################
############ cnn_setup 11 ##############
######################################## 

class CNN9(nn.Module):

    def __init__(self):
        
        super(CNN9, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
            ConvLayer(128, 512, kernel_size=(3, 4, 4), stride=1, padding=0, activation="relu", 
                      layer_norm=(1, 1, 1)),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out
    

    
    
########################################
############ cnn_setup 12 ##############
######################################## 

class CNN10(nn.Module):

    def __init__(self):
        
        super(CNN10, self).__init__()
        
        self.model = nn.Sequential(
            ConvLayer(1,   16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 64, 64)),
            ConvLayer(16,  16,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      max_pool=(1, 2, 2), layer_norm=(24, 64, 64)),
            
            ConvLayer(16,  32,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(24, 32, 32)),
            ConvLayer(32,  32,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            
            ConvLayer(32,  64,  kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(12, 16, 16)),
            ConvLayer(64,  64,  kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            
            ConvLayer(64,  128, kernel_size=5, stride=1, padding=2, activation="relu", 
                      layer_norm=(6, 8, 8)),
            ConvLayer(128, 128, kernel_size=5, stride=2, padding=2, activation="relu", 
                      layer_norm=(3, 4, 4)),
            
            ConvLayer(128, 512, kernel_size=(3, 4, 4), stride=1, padding=0,  activation="relu"),
            
        )

    def forward(self, x):
        out = self.model(x)
            
        return out


# ### Mappings

# #### Input 512, 1, 1, 1

# In[ ]:


########################################
########## mapping_setup 6 #############
########################################

class Encoder_Mapping_2(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_2, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(512, 512),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(512, 512),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
             
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]

    


# #### Input 128, 3, 4, 4

# In[ ]:


########################################
########## mapping_setup 7 #############
########################################

class Encoder_Mapping_3(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_3, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
            
        
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]
    
    
########################################
########## mapping_setup 8 #############
########################################

class Encoder_Mapping_4(nn.Module):

    def __init__(self):
        super(Encoder_Mapping_4, self).__init__()
        
        self.gammas = nn.ModuleList()
        for i in range(4):
            self.gammas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
        
        
        self.betas = nn.ModuleList()
        for i in range(4):
            self.betas.append(nn.Sequential(Flatten(), 
                                      nn.Linear(6144, 2048),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(2048, 1024),
                                      nn.LeakyReLU(.2),
                                      nn.Linear(1024, 512), 
                                      nn.LeakyReLU(.2),
                                      nn.Linear(512, 256), 
                                       )
                         )
            
        
    def forward(self, x):
        out = torch.empty(0).to(x.device)
        
        for gamma in self.gammas: 
            out = torch.cat((out, gamma(x).unsqueeze(1)), 1)
        
        for beta in self.betas: 
            out = torch.cat((out, beta(x).unsqueeze(1)), 1)
        
        return out[:, :4, :], out[:, 4:, :]


# In[ ]:


# mapping = Encoder_Mapping_4().cuda()
# # input_shape = (24, 512, 1, 1, 1)
# input_shape = (24, 128, 3, 4, 4)
# inp = torch.randn(input_shape).cuda()

# summary(mapping, input_size=input_shape, depth=3)


# In[ ]:


print("Imported CNN and Mapping functions.")

