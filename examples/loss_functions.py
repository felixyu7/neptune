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


class LogCMK(torch.autograd.Function):
    """
    Log normalization constant for von Mises-Fisher distribution.
    From: https://github.com/mryab/vmf_loss/blob/master/losses.py
    """

    @staticmethod
    def forward(ctx: Any, m: int, kappa: Tensor) -> Tensor:
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        
        # Use Bessel function iv
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        
        # Prevent log(0) issues
        iv = torch.clamp(iv, min=1e-30)
        return (
            (m / 2.0 - 1) * torch.log(kappa.clamp(min=1e-6))
            - torch.log(iv) 
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa_cpu = kappa.double().cpu().numpy()
        
        # Calculate ratio of Bessel functions
        ratio = (scipy.special.iv(m / 2.0, kappa_cpu)) / (scipy.special.iv(m / 2.0 - 1, kappa_cpu))
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        grads = -torch.from_numpy(ratio).to(grad_output.device).type(dtype)
        
        return None, grad_output * grads


def VonMisesFisherLoss(pred: Tensor, truth: Tensor, kappa_switch: float = 100., eps: float = 1e-6) -> Tensor:
    """
    Von Mises-Fisher loss function for directional data.
    """
    m = truth.size(1)  # Dimension (3 for 3D direction)
    kappa = torch.norm(pred, dim=1).clamp(min=eps)
    dotprod = torch.sum(F.normalize(pred, dim=1) * truth, dim=1)
    
    kappa_switch_tensor = torch.tensor([kappa_switch], device=kappa.device, dtype=kappa.dtype)
    mask_exact = kappa < kappa_switch_tensor
    mask_approx = ~mask_exact
    log_cmk_val = torch.zeros_like(kappa)
    
    # Exact computation for small kappa
    if mask_exact.any():
        log_cmk_val[mask_exact] = LogCMK.apply(m, kappa[mask_exact])
    
    # Approximate computation for large kappa
    if mask_approx.any():
        log_cmk_val[mask_approx] = kappa[mask_approx] - (m / 2 - 1) * torch.log(kappa[mask_approx]) - (m / 2) * np.log(2 * np.pi) - torch.log(torch.tensor(m / 2 - 1).to(kappa.device))
    
    loss = -kappa * dotprod + log_cmk_val
    return torch.mean(loss)