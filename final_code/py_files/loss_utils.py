import torch
import torch.nn.functional as F
import math
import numpy as np

def calc_dice_loss(pred, target):
    
    smooth = 0.

    pred = torch.round(pred)

    pflat, tflat = pred.flatten(), target.flatten()
    
    intersection = (pflat * tflat).sum()

    A_sum = torch.sum(pflat * pflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


# calculate gradient, needed for sdf loss
def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def sdf_criterion(model_in, model_out, surface_array, norm_array, ARGS):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = surface_array
    gt_normals = norm_array

    coords = model_in
    pred_sdf = model_out

    grad = gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf == 1, pred_sdf, torch.zeros_like(pred_sdf)) # surface values should be zero
    inter_constraint = torch.where(gt_sdf == 1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf))) # absolute values off surface should be high
    normal_constraint = torch.where(gt_sdf == 1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad[..., :1])) # norm of surface points should equal gradient
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1) # gradient should be 1

    return (torch.abs(sdf_constraint).mean() * ARGS.lambda_sdf + 
            inter_constraint.mean() * ARGS.lambda_inter + 
            normal_constraint.mean() * ARGS.lambda_normal + 
            grad_constraint.mean() * ARGS.lambda_grad)