import os 
import ast


class init_ARGS(object): 
    def __init__(self): 
        self.device = "GPU"
        self.print_models = False
        self.name = ""

        # Pretrained
        self.pretrained = None
        self.pretrained_best_dataset = "train"
        self.pretrained_best_loss = "mask"
        self.pretrained_models = None
        self.pretrained_lr_reset = None

        # Dataset
        self.dataset = "new"
        self.seed = 34
        self.n_coords_sample = 5000 # coords sampling
        self.norm_min_max = [0, 1]

        # Transformations
        self.rotate = True
        self.translate = True
        self.flip = True
        self.crop = True
        self.stretch = True

        self.translate_max_pixels = 20
        self.stretch_factor = 1.2

        # Train variables
        self.pcmra_epochs = 5000
        self.mask_epochs = 5000

        self.batch_size = 24
        self.eval_every = 50

        self.shuffle = True
        
        # Optim scheduler
        self.min_lr = 1e-5
        self.patience = 200
        
        # CNN setup 
        self.cnn_setup = -1
        
        self.pcmra_train_cnn = True
        self.mask_train_cnn = True
        
        self.cnn_lr = 1e-4
        self.cnn_wd = 0
        
        # Mapping setup
        self.mapping_setup = -1
        
        self.mapping_lr = 1e-4
        self.pcmra_mapping_lr = 1e-4
        
        # SIREN setup 
        self.dim_hidden = 256
        self.siren_hidden_layers = 3
        
        self.first_omega_0 = 30.
        self.hidden_omega_0 = 30.
        
        self.siren_lr = 1e-4
        self.siren_wd = 0
        
        self.mask_siren_final_activation = "sigmoid"
        
        # PCMRA SIREN setup

        self.pcmra_first_omega_0 = 30.
        self.pcmra_hidden_omega_0 = 30.
        
        self.pcmra_siren_lr = 1e-4
        self.pcmra_siren_wd = 0
    
        # switch to sdf loss
        self.sdf = False
        self.sdf_split = None

        # SDF loss lambdas
        self.lambda_sdf = 3e2
        self.lambda_inter = 5e1
        self.lambda_normal = 1e1
        self.lambda_grad = 5e0

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
