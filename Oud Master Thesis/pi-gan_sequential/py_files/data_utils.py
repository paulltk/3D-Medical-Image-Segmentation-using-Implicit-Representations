import random
import torch
import numpy as np
import math
import numbers


import torch.nn.functional as F

from kornia.augmentation.augmentation3d import *
from kornia.geometry.transform import *


##################################################
############## TRANSFORM IMAGE ###################
##################################################


def get_random_shift(ARGS):
        max_t = (4, ARGS.translate_max_pixels, ARGS.translate_max_pixels)
        
        shifts = (random.randint(-max_t[0], max_t[0]), 
                  random.randint(-max_t[1], max_t[1]), 
                  random.randint(-max_t[2], max_t[2]))
        return shifts
     
def translate_image(image, shifts):
    
    image = torch.roll(image, shifts=shifts, dims=(0, 1, 2))

    for axis, shift in enumerate(shifts):
        idx = [[None, None], [None, None], [None, None]]

        if shift > 0: 
            idx[axis][1] = shift
        elif shift < 0: 
            idx[axis][0] = shift
        else:
            continue

        image[idx[0][0]:idx[0][1], idx[1][0]:idx[1][1], idx[2][0]:idx[2][1]] = 0

    return image


def translate_batch(batch, ARGS): 
    idx, subj, proj, pcmras, masks, loss_covers = batch
    
    new_pcmras = new_masks = new_loss_covers = torch.empty((0)).to(pcmras.device)
    
    for pcmra, mask, loss_cover in zip(pcmras, masks, loss_covers): 
        pcmra, mask, loss_cover = pcmra.squeeze(), mask.squeeze(), loss_cover.squeeze()
        
        shifts = get_random_shift(ARGS)
        pcmra = translate_image(pcmra, shifts).unsqueeze(0).unsqueeze(0)
        mask = translate_image(mask, shifts).unsqueeze(0).unsqueeze(0)
        loss_cover = translate_image(loss_cover, shifts).unsqueeze(0).unsqueeze(0)

        new_pcmras = torch.cat((new_pcmras, pcmra), 0)
        new_masks = torch.cat((new_masks, mask), 0)
        new_loss_covers = torch.cat((new_loss_covers, loss_cover), 0)
    
    return idx, subj, proj, new_pcmras, new_masks, new_loss_covers
    

def flip_batch(batch): 
    d_flip, h_flip, v_flip = RandomDepthicalFlip3D(), RandomHorizontalFlip3D(), RandomVerticalFlip3D()
    
    idx, subj, proj, pcmras, masks, loss_covers = batch
    pcmra_masks = torch.cat((pcmras, masks, loss_covers), 1)
    
    pcmra_masks = d_flip(pcmra_masks)
    pcmra_masks = h_flip(pcmra_masks)
    pcmra_masks = v_flip(pcmra_masks)
    
    pcmras, masks, loss_covers = pcmra_masks.split(1, dim=1)

    return idx, subj, proj, pcmras, masks, loss_covers
    

def rotate_batch(batch):
    rotate = RandomRotation3D((10., 15., 15.), p=1.0)
    
    idx, subj, proj, pcmras, masks, loss_covers = batch
    pcmra_masks = torch.cat((pcmras, masks, loss_covers), 1)
    
    pcmra_masks = rotate(pcmra_masks)
    pcmras, masks, loss_covers = pcmra_masks.split(1, dim=1)
    
    return idx, subj, proj, pcmras, masks, loss_covers


def crop_batch(batch, stretch=True, stretch_factor=1.2):
    idx, subj, proj, pcmras, masks, loss_covers = batch
    
    orig_shape = pcmras.shape[2:]
    
    crop_sample = RandomCrop3D(orig_shape, p=1.)
    
    rand = random.uniform
    inc = stretch_factor
    if stretch:
        resize = [rand(1., inc), rand(1., inc), rand(1., inc)]
    else: 
        resize = [rand(1., inc)] * 3
    
    size = tuple([int(i * j) for i, j in zip(orig_shape, resize)])
    
    pcmras = F.interpolate(pcmras, size=size, mode="trilinear")
    masks = F.interpolate(masks, size=size, mode="trilinear")
    loss_covers = F.interpolate(loss_covers, size=size, mode="trilinear")
    
    pcmra_masks = torch.cat((pcmras, masks, loss_covers), 1)
    pcmra_masks = crop_sample(pcmra_masks)

    pcmras, masks, loss_covers = pcmra_masks.split(1, dim=1)

    return idx, subj, proj, pcmras, masks, loss_covers


def transform_batch(batch, ARGS):
    """ Combine all previously defined transformation
    functions. """
        
    if ARGS.flip: 
        batch = flip_batch(batch)
    if ARGS.translate: 
        batch = translate_batch(batch, ARGS)
    if ARGS.crop: 
        batch = crop_batch(batch, ARGS.stretch, ARGS.stretch_factor)
    if ARGS.rotate: 
        batch = rotate_batch(batch)
    
    idx, subj, proj, pcmras, masks, loss_covers = batch
    
    masks = torch.round(masks)
    loss_covers = torch.floor(torch.round(loss_covers*10) / 10)
    
    return idx, subj, proj, pcmras, masks, loss_covers


##################################################
##### Functions to extract surface and norm ######
##################################################

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
            self.kernel_radius = kernel_size // 2
        else:
            self.kernel_radius = kernel_size[0] // 2

        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim


        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) *                       torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.kernel_radius)
    
    
def gradient3d(data, normalize=False, s=2):
    data_padded = torch.nn.functional.pad(data, (1,1,1,1,1,1,0,0,0,0))

    grad_x = (data_padded[:,:, s:,   1:-1, 1:-1] - data_padded[:,:, 0:-s, 1:-1, 1:-1]) / s
    grad_y = (data_padded[:,:, 1:-1, s:,   1:-1] - data_padded[:,:, 1:-1, 0:-s, 1:-1]) / s
    grad_z = (data_padded[:,:, 1:-1, 1:-1, s:  ] - data_padded[:,:, 1:-1, 1:-1 ,0:-s]) / s
    
    grad = torch.cat([grad_x,grad_y,grad_z], dim=1)
    grad_magn = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    if normalize:
        eps=1e-8
        grad = grad/(grad_magn + eps)
    
    return grad, grad_magn


def initialize_blurring_layer(sigma, DEVICE):
    # Initialize the blurring layer
    size = math.ceil(3*sigma)
    return GaussianSmoothing(1, [size,size,size], sigma, dim=3).to(DEVICE)


def get_surface_and_norm(batch, blur_layer, DEVICE): 

    mask = batch[-2]
    
    # get the surface
    maxpool_layer = torch.nn.MaxPool3d(3,stride=1,padding=1).to(DEVICE)
    
    mask_eroded = 1 - maxpool_layer(1 - mask)
    surface = (mask - mask_eroded).type(torch.float32)

    # get the norm
    masks_blurred = blur_layer(mask)     

    grad, grad_magn = gradient3d(masks_blurred, normalize=True, s=2)
    norm = grad * surface * -1
    
    return surface, norm

##################################################
###### Functions that return a siren batch #######
##################################################


def prod(val) :  
    res = 1 
    for ele in val:  
        res *= ele  
    return res 

    
def get_coords(*sidelengths):
    tensors = []

    for sidelen in sidelengths:
        tensors.append(torch.linspace(-1, 1, steps=sidelen))

    tensors = tuple(tensors)
    coords = torch.stack(torch.meshgrid(*tensors), dim=-1)

    return coords.reshape(-1, len(sidelengths))


def reshape_arrays(*arrays): 
    return [array.view(array.shape[0], array.shape[1], -1).permute(0, 2, 1) for array in arrays]


def get_siren_batch(batch, blur_layer, n, ARGS): 
    
    sdf_split = ARGS.sdf_split

    idx, subj, proj, pcmras, masks, loss_covers = batch
    subjects = []
 
    # initialize a coords matrix
    coords = get_coords(*pcmras.shape[2:]).to(pcmras.device)

    # reshape all matrixes 
    pcmra_array, mask_array, loss_cover_array = reshape_arrays(pcmras, masks, loss_covers)
    
    coords_array = coords.unsqueeze(0).repeat(pcmras.shape[0], 1, 1)
    
    if ARGS.sdf: 
        # get the surface and norm of the mask
        surface_n, random_n = int(n*sdf_split), n - int(n*sdf_split)
        surfaces, norms = get_surface_and_norm(batch, blur_layer, masks.device)        
        surface_array, norm_array = reshape_arrays(surfaces, norms)
        
    elif not ARGS.sdf: 
        surface_array = norm_array = torch.tensor([]).repeat(pcmras.shape[0], 1)

    if n != -1:
        # select n coords and their corresponding values
        for pcmra, mask, loss_cover, surface, norm in zip(pcmra_array, mask_array, loss_cover_array, surface_array, norm_array):

            if ARGS.sdf:
                # select n * sfd_split points that lie on the surface
                surface_idx = (surface != 0).nonzero()[:, 0].flatten().cpu().numpy()
                surface_idx = np.random.choice(surface_idx, surface_n)

                # select n random coords that have a non zero loss_cover
                random_idx = (loss_cover != 0).nonzero()[:, 0].cpu().numpy()
                random_idx = np.random.choice(random_idx, random_n)

                idx = np.concatenate((surface_idx, random_idx))
        
                subject = [coords[idx, :].unsqueeze(0), pcmra[idx, :].unsqueeze(0), mask[idx, :].unsqueeze(0),
                        surface[idx, :].unsqueeze(0), norm[idx, :].unsqueeze(0)]

            elif not ARGS.sdf: 
                # select n random coords that have a non zero loss_cover
                idx = (loss_cover != 0).nonzero()[:, 0].cpu().numpy()
                idx = np.random.choice(idx, n)
                
                subject = [coords[idx, :].unsqueeze(0), pcmra[idx, :].unsqueeze(0), mask[idx, :].unsqueeze(0),
                        torch.tensor([]).unsqueeze(0), torch.tensor([]).unsqueeze(0)]

            subjects.append(subject)

        coords_array = torch.cat([subj[0] for subj in subjects], 0)
        pcmra_array = torch.cat([subj[1] for subj in subjects], 0)
        mask_array = torch.cat([subj[2] for subj in subjects], 0)
        surface_array = torch.cat([subj[3] for subj in subjects], 0)
        norm_array = torch.cat([subj[4] for subj in subjects], 0)
    
    return idx, subj, proj, pcmras, coords_array, pcmra_array, mask_array, surface_array, norm_array