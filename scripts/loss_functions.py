import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import scipy.special
from typing import Any, Tuple


def AngularDistanceLoss(pred: Tensor, truth: Tensor, eps: float = 1e-7, reduction: str = "mean") -> Tensor:
    """Angular distance loss function for direction reconstruction."""
    # Normalize pred and truth to unit vectors
    pred = F.normalize(pred, p=2, dim=1)
    truth = F.normalize(truth, p=2, dim=1)
    
    # Clamp prevents invalid input to arccos
    cos_sim = F.cosine_similarity(pred, truth)
    angle = torch.acos(torch.clamp(cos_sim, min=-1.0 + eps, max=1.0 - eps))
    loss = angle / np.pi
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def DirectionalMSELoss(pred: Tensor, truth: Tensor) -> Tensor:
    """
    Simple MSE loss on directional vector components (x, y, z).
    This should converge to dataset mean if no learning signal exists.
    """
    return F.mse_loss(pred, truth)


def GaussianNLLLoss(mu: Tensor, var: Tensor, target: Tensor) -> Tensor:
    """
    Gaussian Negative Log Likelihood Loss for energy reconstruction with uncertainty.
    """
    # Ensure var is positive and stable
    var = F.softplus(var) + 1e-6
    
    # NLL for each sample
    nll = 0.5 * ((target - mu)**2 / var + torch.log(var))
    
    # Return average over batch
    return torch.mean(nll)

def VonMisesFisherLoss(n_pred, n_true, eps = 1e-8):
    """  von Mises-Fisher Loss: n_true is unit vector ! """
    kappa = torch.norm(n_pred, dim=1)        
    logC  = -kappa + torch.log( ( kappa+eps )/( 1-torch.exp(-2*kappa)+2*eps ) )
    return -( (n_true*n_pred).sum(dim=1) + logC ).mean() 