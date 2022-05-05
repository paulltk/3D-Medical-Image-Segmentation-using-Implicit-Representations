import os 
import ast


class init_ARGS(object): 
    def __init__(self, silent=False): 
        self.device = "GPU"
        self.print_models = False
        self.save_models = True
        self.name = ""

        # training
        self.training_setup = "segmentation" # reconstruction, segmentation, combined, consecutively
        self.train_encoder_seg = True

        # Pretrained
        self.pretrained = None
        self.pretrained_best_dataset = "train"
        self.pretrained_best_loss = "mask"
        self.pretrained_models = None
        self.pretrained_lr_reset = None

        # Dataset
        self.seed = 34
        self.n_coords_sample = 5000 # coords sampling
        self.norm_min_max = [0, 1]

        # Transformations
        self.rotate = True
        self.translate = True
        self.crop = True
        self.stretch = True
        self.flip = False

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
        self.patience = 50
        
        # Model setups 
        self.cnn_setup = "golden"
        self.mapping_setup = "golden"
        
        # SIREN setup 
        self.dim_hidden = 256
        self.siren_hidden_layers = 3
        
        # Omega hyper-parameter
        self.first_omega_0 = 30.
        self.hidden_omega_0 = 30.
        self.pcmra_first_omega_0 = 30.
        self.pcmra_hidden_omega_0 = 30.

        # Segmentation type
        self.segmentation = "binary"
        
        # Learning rates 
        self.cnn_lr = 1e-4
        self.mapping_lr = 1e-4
        self.pcmra_mapping_lr = 1e-4        
        self.siren_lr = 1e-4
        self.pcmra_siren_lr = 1e-4

        # Weight decays 
        self.cnn_wd = 0
        self.siren_wd = 0
        self.pcmra_siren_wd = 0

        # SDF loss lambdas
        self.lambda_sdf = 3e2
        self.lambda_inter = 1e2
        self.lambda_normal = 1e1
        self.lambda_grad = 5e0

        # SDF surface/non-surface split
        self.sdf_split = 0.5

        # combined segmentation loss lambda
        self.seg_loss_lambda = 2e-1
        
        if not silent:
            print("WARNING: ARGS class initialized.")

    def set_args(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
          
        
def load_args(run, print_changed=True, silent=False):
    run_path = os.path.join("saved_runs", run, "ARGS.txt")

    with  open(run_path, "r") as f:
        contents = f.read()
        args_dict = ast.literal_eval(contents)
    
    ARGS = init_ARGS(silent)
    
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
