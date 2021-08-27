import warnings
import time

from py_files.args import *

from py_files.functions import *

from py_files.dataset import *

from py_files.cnn_models import *
from py_files.mapping_models import *
from py_files.pigan_model import *

from py_files.load_utils import *
from py_files.data_utils import *
from py_files.plot_utils import *
from py_files.loss_utils import *
from py_files.train_utils import *
from py_files.save_utils import *

training_setups = [
               ("reconstruction", None, "golden", "1net", 100), 
               ("reconstruction", None, "golden", "2net", 100), 
               ("reconstruction", None, "golden", "4net", 100), 
               ("reconstruction", None, "golden", "golden", 100), 
               ("segmentation", "binary", "golden", "1net", 100), 
               ("segmentation", "binary", "golden", "2net", 100), 
               ("segmentation", "binary", "golden", "4net", 100), 
               ("segmentation", "binary", "golden", "golden", 100), 
               ]

for i in range(1):
    for training_setup, segmentation, cnn_setup, mapping_setup, first_omega_0 in training_setups: 
        ARGS = init_ARGS()
        
        ARGS.training_setup = training_setup
        ARGS.segmentation = segmentation

        ARGS.cnn_setup = mapping_setup
        ARGS.mapping_setup = mapping_setup
        ARGS.first_omega_0 = first_omega_0
        
        print(f"Starting training {ARGS.name}.")

        print(vars(ARGS))
            
        complete_training(ARGS)  
                
        torch.cuda.empty_cache()  
