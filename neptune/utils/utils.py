import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Union, Any, Tuple
import scipy.special

def LogCoshLoss(pred: Tensor, truth: Tensor) -> Tensor:
    """LogCosh loss function. approximated for easier to compute gradients"""
    x = pred - truth
    return (x + torch.nn.functional.softplus(-2.0 * x) - np.log(2.0)).mean()

def AngularDistanceLoss(pred: Tensor, truth: Tensor, weights: Optional[Tensor]=None, eps: float=1e-7, reduction: str="mean") -> Tensor:
    """Angular distance loss function"""
    # normalize pred and truth to unit vectors
    pred = F.normalize(pred, p=2, dim=1)
    truth = F.normalize(truth, p=2, dim=1)
    # clamp prevents invalid input to arccos
    cos_sim = F.cosine_similarity(pred, truth)
    angle = torch.acos(torch.clamp(cos_sim, min=-1.0 + eps, max=1.0 - eps))
    loss = angle / np.pi

    if weights is not None:
        loss = loss * weights
    
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
    Calculate the Gaussian Negative Log Likelihood Loss.
    mu, var, target: all shape [batch_size] (or broadcastable).
    var is assumed to be strictly > 0.
    """
    # Ensure var is positive and stable
    var = F.softplus(var) + 1e-6 # Apply softplus and add epsilon for stability
    # NLL for each sample
    nll = 0.5 * ((target - mu)**2 / var + torch.log(var))
    # Return average over batch
    return torch.mean(nll)

# VMF loss function based on implementation from graphnet (https://github.com/graphnet-team/graphnet) #
    
class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin
    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        # Use Bessel function iv (scipy.special.iv)
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        # Prevent log(0) issues for iv
        iv = torch.clamp(iv, min=1e-30) # Clamp iv to avoid log(0)
        return (
            (m / 2.0 - 1) * torch.log(kappa.clamp(min=1e-6)) # Clamp kappa to avoid log(0)
            - torch.log(iv) 
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tuple[None, Tensor]: # Added Tuple return type hint
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa_cpu = kappa.double().cpu().numpy()
        # Calculate ratio of Bessel functions
        ratio = (scipy.special.iv(m / 2.0, kappa_cpu)) / (scipy.special.iv(m / 2.0 - 1, kappa_cpu))
        # Handle potential NaN/inf if the denominator Bessel function is zero
        ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        grads = -torch.from_numpy(ratio).to(grad_output.device).type(dtype)
        
        return (
            None, # Gradient for m is None
            grad_output * grads,
        )
    
def log_cmk_exact(m: int, kappa: Tensor) -> Tensor:
    """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
    return LogCMK.apply(m, kappa)

def log_cmk_approx(m: int, kappa: Tensor) -> Tensor:
    """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

    [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
    Uses approximation for large kappa.
    """
    v = m / 2.0 - 0.5
    a = torch.sqrt((v + 1) ** 2 + kappa**2)
    b = v - 1
    # Clamp arguments to log to avoid issues
    log_term = torch.log(b + a).clamp(max=85) # Clamp to avoid overflow in exp later if needed
    return -a + b * log_term
    
def VonMisesFisherLoss(pred: Tensor, truth: Tensor, kappa_switch: float = 100., reg_factor: float = 0.0, eps: float = 1e-6) -> Tensor:
    """
    Von Mises-Fisher loss function.
    Args:
        pred: Predicted output tensor [B, D].
        truth: True output tensor [B, D].
        kappa_switch: Value of kappa at which to switch from exact to approximate log_cmk.
        reg_factor: Optional regularization factor for kappa.
        eps: Small epsilon for numerical stability (used for normalizing pred if needed).
    Returns:
        Mean loss value.
    """
    m = truth.size(1) # Dimension (e.g., 3 for 3D direction)
    kappa = torch.norm(pred, dim=1).clamp(min=eps)
    dotprod = torch.sum(F.normalize(pred, dim=1) * truth, dim=1)
    
    kappa_switch_tensor = torch.tensor([kappa_switch], device=kappa.device, dtype=kappa.dtype)
    mask_exact = kappa < kappa_switch_tensor
    mask_approx = ~mask_exact

    log_cmk_val = torch.zeros_like(kappa)

    # Calculate exact log C_m(k) for kappa < kappa_switch
    if torch.any(mask_exact):
        log_cmk_val[mask_exact] = log_cmk_exact(m, kappa[mask_exact])

    # Calculate approximate log C_m(k) for kappa >= kappa_switch
    if torch.any(mask_approx):
         # Ensure continuity at kappa_switch
        offset = log_cmk_approx(m, kappa_switch_tensor) - log_cmk_exact(m, kappa_switch_tensor)
        log_cmk_val[mask_approx] = log_cmk_approx(m, kappa[mask_approx]) - offset
    
    # Loss = -log C_m(k) - kappa * <pred_dir, true_dir> + reg * kappa
    elements = -log_cmk_val - (kappa * dotprod) + (reg_factor * kappa)
    return elements.mean()

def CombinedAngularVMFDistanceLoss(pred: Tensor, truth: Tensor, angular_weight: float = 0.5, vmf_kappa_switch: float = 100., vmf_reg_factor: float = 0.0, vmf_eps: float = 1e-6, angular_eps: float = 1e-7) -> Tensor:
    """
    Combined loss function weighting Angular Distance and Von Mises-Fisher loss.
    Args:
        pred: Predicted output tensor [B, D].
        truth: True output tensor [B, D].
        angular_weight: Weight factor for the AngularDistanceLoss component (0 to 1).
                        The VMF loss component will have weight (1 - angular_weight).
        vmf_kappa_switch: kappa value to switch vMF calculation (passed to VonMisesFisherLoss).
        vmf_reg_factor: kappa regularization factor (passed to VonMisesFisherLoss).
        vmf_eps: Epsilon for VMF stability (passed to VonMisesFisherLoss).
        angular_eps: Epsilon for AngularDistanceLoss stability.
    Returns:
        Mean combined loss value.
    """
    if not (0.0 <= angular_weight <= 1.0):
        raise ValueError(f"angular_weight must be between 0 and 1, got {angular_weight}")

    # Angular distance loss (normalizes pred internally)
    loss_ang = AngularDistanceLoss(pred, truth, eps=angular_eps, reduction="mean")

    # Von Mises-Fisher loss
    loss_vmf = VonMisesFisherLoss(pred, truth, kappa_switch=vmf_kappa_switch, reg_factor=vmf_reg_factor, eps=vmf_eps)

    # Combine losses
    combined_loss = angular_weight * loss_ang + (1.0 - angular_weight) * loss_vmf
    
    return combined_loss

def CombinedAngleEnergyLoss(pred: Tensor, truth: Tensor) -> Tensor:
    """Combined loss function for both angular and energy reco"""
    # Assumes truth/pred are shape [B, 4] (log_energy, dir_x, dir_y, dir_z)
    energy_pred = pred[:, 0]
    energy_truth = truth[:, 0]
    angles_pred = pred[:, 1:]
    angles_truth = truth[:, 1:]
    
    energy_loss = LogCoshLoss(energy_pred, energy_truth)
    angle_loss = AngularDistanceLoss(angles_pred, angles_truth)
    
    # 0.5 weighting on the energy loss since it tends to be larger
    loss = angle_loss + 0.5 * energy_loss
    return loss

def CrossEntropyLoss(pred: Tensor, truth: Tensor) -> Tensor:
    return F.cross_entropy(pred, truth)

def BinaryCrossEntropyLoss(pred: Tensor, truth: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(pred, truth)

def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the unit vector of the vector."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot_product = np.dot(v1_u, v2_u)
    # clip prevents invalid input to arccos
    clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
    return np.arccos(clipped_dot_product)

def farthest_point_sampling(points: Tensor, n_samples: int) -> Tensor:
    """
    Efficiently select a subset of points using Farthest Point Sampling
    Args:
        points: Tensor of shape [N, D] where N is the number of points and D is the dimension
        n_samples: Number of points to sample
    Returns:
        indices: Tensor of shape [n_samples] containing indices of selected points
    """
    device = points.device
    N, D = points.shape
    
    if N == 0:
        return torch.tensor([], dtype=torch.long, device=device)
    if n_samples <= 0:
        return torch.tensor([], dtype=torch.long, device=device)
    if n_samples > N:
        # If requesting more samples than available points, return all indices shuffled
        return torch.randperm(N, device=device)
        
    selected_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    # Initialize with a random point 
    first_idx = torch.randint(0, N, (1,), device=device)
    selected_indices[0] = first_idx
    
    # Initialize min_distances tensor
    min_distances = torch.full((N,), float('inf'), device=device, dtype=points.dtype)
    
    # Calculate initial distances from the first point
    first_point = points[first_idx]
    distances = torch.sum((points - first_point)**2, dim=1)
    min_distances = torch.minimum(min_distances, distances)
    
    # Iteratively select points
    for i in range(1, n_samples):
        # Select the point with the largest minimum distance
        next_idx = torch.argmax(min_distances)
        selected_indices[i] = next_idx
        
        # Skip update on the last iteration for efficiency
        if i < n_samples - 1:
            # Compute distances from this new point to all other points
            new_point = points[next_idx]
            new_distances = torch.sum((points - new_point)**2, dim=1)
            # Update min_distances 
            min_distances = torch.minimum(min_distances, new_distances)
    
    return selected_indices


# Removed downstream_task_loss function as logic is now in Neptune._get_loss_function
# def downstream_task_loss(preds: Tensor, labels: Tensor, task: str='angular_reco', current_epoch: int=0) -> Tensor:
#     """
#     Compute the appropriate loss function based on the downstream task
#     Args:
#         preds: Model predictions
#         labels: Ground truth labels
#         task: The task type (angular_reco, energy_reco, combined_reco,
#                            vmf_reco, gaussian_energy_reco)
#         current_epoch: Current training epoch (unused but kept for compatibility)
#     Returns:
#         loss: The computed loss value
#     """
#     if task == 'angular_reco':
#         # Angular reconstruction task (direction only)
#         # Assumes preds/labels are shape [B, 3]
#         return AngularDistanceLoss(preds, labels)
#     elif task == 'energy_reco':
#         # Energy reconstruction task
#         # Assumes preds/labels are shape [B, 1] or [B]
#         if preds.dim() > 1 and preds.shape[1] == 1:
#             preds = preds.squeeze(1)
#         if labels.dim() > 1 and labels.shape[1] == 1:
#             labels = labels.squeeze(1)
#         return LogCoshLoss(preds, labels)
#     elif task == 'combined_reco':
#         # Combined energy and direction reconstruction
#         # Assumes preds/labels are shape [B, 4]
#         return CombinedAngleEnergyLoss(preds, labels)
#     elif task == 'vmf_reco':
#         # Von Mises-Fisher loss for angular reconstruction
#         # Assumes preds are unnormalized [B, 3], labels are unit vectors [B, 3]
#         return VonMisesFisherLoss(preds, labels)
#     elif task == 'gaussian_energy_reco':
#         # Gaussian NLL loss for energy reconstruction
#         # Assumes preds are [B, 2] (mu, var), labels are [B, 1] or [B]
#         mu = preds[:, 0]
#         var = preds[:, 1]
#         if labels.dim() > 1 and labels.shape[1] == 1:
#             labels = labels.squeeze(1)
#         return GaussianNLLLoss(mu, var, labels)
#     else:
#         raise ValueError(f"Unknown task type: {task}") 