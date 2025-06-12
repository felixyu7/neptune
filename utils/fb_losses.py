import torch
import torch.nn as nn
import numpy as np
from sphere.distribution import fb8

# Batch-aware helper functions to build orientation matrices
def create_matrix_H_torch(theta, phi):
    b = theta.shape[0]
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
    H = torch.zeros(b, 3, 3, dtype=theta.dtype, device=theta.device)
    H[:, 0, 0], H[:, 0, 1] = cos_theta, -sin_theta
    H[:, 1, 0], H[:, 1, 1], H[:, 1, 2] = sin_theta * cos_phi, cos_theta * cos_phi, -sin_phi
    H[:, 2, 0], H[:, 2, 1], H[:, 2, 2] = sin_theta * sin_phi, cos_theta * sin_phi, cos_phi
    return H

def create_matrix_K_torch(psi):
    b = psi.shape[0]
    cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)
    K = torch.eye(3, dtype=psi.dtype, device=psi.device).unsqueeze(0).repeat(b, 1, 1)
    K[:, 1, 1], K[:, 1, 2] = cos_psi, -sin_psi
    K[:, 2, 1], K[:, 2, 2] = sin_psi, cos_psi
    return K

def create_matrix_Gamma_torch(theta, phi, psi):
    H = create_matrix_H_torch(theta, phi)
    K = create_matrix_K_torch(psi)
    return torch.bmm(H, K)

def spherical_coordinates_to_nu_torch(alpha, rho):
    gamma_nu = create_matrix_Gamma_torch(alpha, rho, torch.zeros_like(alpha))
    return gamma_nu[:, :, 0]

# `sphere` library's series expansion with analytical gradients
class LogNormalizeSphereSeries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kappa, beta, eta, alpha, rho):
        batch_size = kappa.shape[0]
        log_c_values = []
        instances = []
        for i in range(batch_size):
            instance = fb8(
                theta=0, phi=0, psi=0,
                kappa=kappa[i].item(), beta=beta[i].item(), eta=eta[i].item(),
                alpha=alpha[i].item(), rho=rho[i].item()
            )
            log_c = instance.log_normalize()
            instances.append(instance)
            log_c_values.append(log_c)
        
        ctx.instances = instances
        return torch.tensor(log_c_values, dtype=kappa.dtype, device=kappa.device)

    @staticmethod
    def backward(ctx, grad_output):
        grads = []
        for instance in ctx.instances:
            grad_array = instance._grad_log_normalize()
            # grad_array is [grad_kappa, grad_beta, grad_eta, grad_alpha, grad_rho]
            grad_tuple = (
                grad_array[0],  # grad_kappa
                grad_array[1],  # grad_beta
                grad_array[2],  # grad_eta
                grad_array[3],  # grad_alpha
                grad_array[4]   # grad_rho
            )
            grads.append(grad_tuple)
        
        # Transpose and convert to tensors; each has shape (batch,)
        grad_tensors = [
            torch.tensor(list(g), dtype=grad_output.dtype, device=grad_output.device)
            for g in zip(*grads)
        ]

        # Element-wise multiply (no broadcasting to (B,B))
        return tuple(g * grad_output for g in grad_tensors)


class FB8NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta, phi, psi, kappa, beta, eta, alpha, rho):
        params = [theta, phi, psi, kappa, beta, eta, alpha, rho]
        theta, phi, psi, kappa, beta, eta, alpha, rho = [p.squeeze() if p.dim() > 1 else p for p in params]

        Gamma = create_matrix_Gamma_torch(theta, phi, psi)
        # Per sphere.spherical_coordinates_to_gammas, use columns.
        gamma1, gamma2, gamma3 = Gamma[:, :, 0], Gamma[:, :, 1], Gamma[:, :, 2]
        
        nu = spherical_coordinates_to_nu_torch(alpha, rho)
        
        logC = LogNormalizeSphereSeries.apply(kappa, beta, eta, alpha, rho)

        g1x = torch.einsum('bi,bi->b', x, gamma1)
        g2x = torch.einsum('bi,bi->b', x, gamma2)
        g3x = torch.einsum('bi,bi->b', x, gamma3)
        
        ngx = torch.einsum('bi,bi->b', nu, torch.stack([g1x, g2x, g3x], dim=1))
        logp = kappa * ngx + beta * (g2x**2 - eta * g3x**2) - logC
        
        return -logp

class FB5NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta, phi, psi, kappa, beta):
        params = [theta, phi, psi, kappa, beta]
        theta, phi, psi, kappa, beta = [p.squeeze() if p.dim() > 1 else p for p in params]

        Gamma = create_matrix_Gamma_torch(theta, phi, psi)
        # Per sphere.spherical_coordinates_to_gammas, use columns.
        gamma1, gamma2, gamma3 = Gamma[:, :, 0], Gamma[:, :, 1], Gamma[:, :, 2]

        # For FB5, eta is 1, alpha and rho are 0
        eta = torch.ones_like(kappa)
        alpha = torch.zeros_like(kappa)
        rho = torch.zeros_like(kappa)
        
        logC = LogNormalizeSphereSeries.apply(kappa, beta, eta, alpha, rho)

        g1x = torch.einsum('bi,bi->b', x, gamma1)
        g2x = torch.einsum('bi,bi->b', x, gamma2)
        g3x = torch.einsum('bi,bi->b', x, gamma3)
        
        logp = kappa * g1x + beta * (g2x**2 - g3x**2) - logC
        
        return -logp.mean()

class FB6NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta, phi, psi, kappa, beta, eta):
        params = [theta, phi, psi, kappa, beta, eta]
        theta, phi, psi, kappa, beta, eta = [p.squeeze() if p.dim() > 1 else p for p in params]

        Gamma = create_matrix_Gamma_torch(theta, phi, psi)
        # Per sphere.spherical_coordinates_to_gammas, use columns.
        gamma1, gamma2, gamma3 = Gamma[:, :, 0], Gamma[:, :, 1], Gamma[:, :, 2]

        # For FB6, alpha and rho are 0
        alpha = torch.zeros_like(kappa)
        rho = torch.zeros_like(kappa)
        
        logC = LogNormalizeSphereSeries.apply(kappa, beta, eta, alpha, rho)

        g1x = torch.einsum('bi,bi->b', x, gamma1)
        g2x = torch.einsum('bi,bi->b', x, gamma2)
        g3x = torch.einsum('bi,bi->b', x, gamma3)
        
        # For FB6, nu is [1,0,0], so ngx simplifies to g1x
        logp = kappa * g1x + beta * (g2x**2 - eta * g3x**2) - logC
        
        return -logp.mean()