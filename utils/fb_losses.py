import torch
import torch.nn as nn
import numpy as np
from sphere.distribution.distribution import fb8, FB8Distribution
from sphere.distribution.saddle import spa
import math
from typing import Tuple

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
        
        return -logp.mean()

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

# -------------------------------------------------------------
#  Efficient FB8 Negative Log-Likelihood
#  Method from: "Maximum likelihood estimation of the Fisher-Bingham
#  distribution via efficient calculation of its normalizing constant"
#  by Chen et al. (2020)
# -------------------------------------------------------------

def fb8_to_Bv(theta: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor,
              kappa: torch.Tensor, beta: torch.Tensor, eta: torch.Tensor,
              alpha: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert FB8 parameterisation to (B, v) with tr(B)=0."""
    Gamma = create_matrix_Gamma_torch(theta, phi, psi)
    nu = spherical_coordinates_to_nu_torch(alpha, rho)
    
    v = kappa.unsqueeze(-1) * torch.bmm(Gamma, nu.unsqueeze(-1)).squeeze(-1)

    gamma2 = Gamma[..., :, 1]
    gamma3 = Gamma[..., :, 2]
    g2g2t = torch.bmm(gamma2.unsqueeze(-1), gamma2.unsqueeze(-2))
    g3g3t = torch.bmm(gamma3.unsqueeze(-1), gamma3.unsqueeze(-2))
    B = beta.unsqueeze(-1).unsqueeze(-1) * (g2g2t - eta.unsqueeze(-1).unsqueeze(-1) * g3g3t)

    tr = B.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    B_traceless = B - tr[..., None] * torch.eye(3, device=B.device, dtype=B.dtype) / 3.0
    
    return B_traceless, v

def _cet_fft(Lambda: torch.Tensor, gamma: torch.Tensor, N: int = 1024,
             c: float = None,
             omega_d: float = 0.5,
             omega_u: float = 1.5
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return log(C), d(logC)/dΛ, d(logC)/dγ for each batch row, using fp64 for stability.
    This version uses adaptive c parameter based on gamma magnitude to handle high-kappa regimes.
    """
    original_dtype = Lambda.dtype
    Lambda_64 = Lambda.to(torch.float32)
    gamma_64 = gamma.to(torch.float32)
    
    B, p = Lambda_64.shape
    assert p == 3, "Only S² supported (p=3)"

    # Adaptive c parameter based on gamma magnitude
    # For high gamma values, c needs to scale to avoid numerical issues
    if c is None:
        max_gamma = gamma_64.abs().max()
        max_lambda = Lambda_64.abs().max()
        effective_scale = torch.max(max_gamma, max_lambda)

        if effective_scale <= 10:
            c = 25.0
        elif effective_scale <= 80:
            c = 0.5 * effective_scale
        elif effective_scale <= 250:
            c = 0.6 * effective_scale
        else:
            c = 0.7 * effective_scale

    # Shift eigenvalues for numerical stability
    Λ_max, Λ_max_indices = Lambda_64.max(dim=1, keepdim=True)
    Λ_shift  = Lambda_64 - Λ_max
    
    # 1. Implement Adaptive 'd'
    # The theory from Chen et al. (2020) requires 'd' to be chosen based on the
    # eigenvalues (Λ) and the contour location (c). This re-introduces adaptivity.
    # d must be <= min(c + Λ_shift_i).
    if effective_scale <= 80:
        d_factor = 0.25
    else:
        d_factor = 0.1
    min_lambda_shift, _ = torch.min(Λ_shift, dim=1, keepdim=True)
    d = (d_factor * (c + min_lambda_shift)).clamp_min(1e-9) # Ensure d is positive

    # 2. Update h, p1, p2 formulas based on Chen et al. (2020)
    # These are now tensors of shape (B, 1) because 'd' is adaptive.
    h  = torch.sqrt(2 * math.pi * d * (omega_d + omega_u) / (omega_d**2 * N))
    p1 = torch.sqrt(N * h / omega_d)
    p2 = torch.sqrt(omega_d * N * h / 4.0)

    # Set up the integration grid and window function.
    # 'nh' and 'log_w' will now be shape (B, 2N+2) due to broadcasting.
    ns = torch.arange(-N - 1, N + 1, device=Lambda_64.device, dtype=torch.float64)
    nh = ns.unsqueeze(0) * h
    log_w = torch.log(0.5 * torch.special.erfc(nh.abs() / p1 - p2))
    
    # Denominator in the integrand's exponent. This broadcasts correctly.
    # We use the adaptive 'c' for all samples in the batch for stability.
    denom = -Λ_shift.unsqueeze(-1) + 1j * nh.unsqueeze(1) + c
    
    # Calculate the log-magnitude and phase of the complex term A_n
    g2_4d = (gamma_64.unsqueeze(-1) ** 2) / (4.0 * denom)
    log_abs_A = (g2_4d.real - 0.5 * torch.log(denom.abs())).sum(dim=1)
    arg_A = (g2_4d.imag - 0.5 * denom.angle()).sum(dim=1)
    
    # Calculate the log of the full complex integrand z_n
    log_z_real = log_w + log_abs_A
    log_z_imag = nh + arg_A
    
    # Use the log-sum-exp trick to stably sum the integrands
    max_log_z = log_z_real.max(dim=-1, keepdim=True)[0]
    exp_term = torch.exp(log_z_real - max_log_z)
    cos_log_z_imag = torch.cos(log_z_imag)
    sin_log_z_imag = torch.sin(log_z_imag)

    S_R = (exp_term * cos_log_z_imag).sum(dim=-1)
    S_I = (exp_term * sin_log_z_imag).sum(dim=-1)
    S_abs = torch.hypot(S_R, S_I).clamp_min(1e-99)

    log_k = 0.5 * math.log(math.pi) + c + torch.log(h.squeeze(-1))
    logC = log_k + torch.log(S_abs) + max_log_z.squeeze(-1) + Λ_max.squeeze(-1)
    
    # --- Gradients ---
    dlogA_dΛ = -g2_4d / denom - 0.5 / denom
    dlogA_dγ = gamma_64.unsqueeze(-1) / (2 * denom)

    re_z_scaled = exp_term * cos_log_z_imag
    im_z_scaled = exp_term * sin_log_z_imag
    
    re_dlogA_dΛ = dlogA_dΛ.real
    im_dlogA_dΛ = dlogA_dΛ.imag
    re_dlogA_dγ = dlogA_dγ.real
    im_dlogA_dγ = dlogA_dγ.imag
    
    num_dΛ_real = (re_z_scaled.unsqueeze(1) * re_dlogA_dΛ - im_z_scaled.unsqueeze(1) * im_dlogA_dΛ).sum(dim=-1)
    num_dΛ_imag = (im_z_scaled.unsqueeze(1) * re_dlogA_dΛ + re_z_scaled.unsqueeze(1) * im_dlogA_dΛ).sum(dim=-1)

    S_abs2 = S_abs.unsqueeze(-1)**2
    dlogC_dΛ_shifted = (S_R.unsqueeze(-1) * num_dΛ_real + S_I.unsqueeze(-1) * num_dΛ_imag) / S_abs2
    
    dΛ_max_dΛ = torch.nn.functional.one_hot(Λ_max_indices.squeeze(-1), num_classes=p).to(dlogC_dΛ_shifted.dtype)
    grad_sum = dlogC_dΛ_shifted.sum(dim=1, keepdim=True)
    dlogC_dΛ = dlogC_dΛ_shifted + dΛ_max_dΛ * (1 - grad_sum)
    
    num_dγ_real = (re_z_scaled.unsqueeze(1) * re_dlogA_dγ - im_z_scaled.unsqueeze(1) * im_dlogA_dγ).sum(dim=-1)
    num_dγ_imag = (im_z_scaled.unsqueeze(1) * re_dlogA_dγ + re_z_scaled.unsqueeze(1) * im_dlogA_dγ).sum(dim=-1)
    
    dlogC_dγ = (S_R.unsqueeze(-1) * num_dγ_real + S_I.unsqueeze(-1) * num_dγ_imag) / S_abs2
    
    return logC.to(original_dtype), dlogC_dΛ.to(original_dtype), dlogC_dγ.to(original_dtype)

class _FBNormaliserFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        Λ, O = torch.linalg.eigh(B)
        γ = (O.mT @ v.unsqueeze(-1)).squeeze(-1)
        logC, dlogC_dΛ, dlogC_dγ = _cet_fft(Λ, γ)
        ctx.save_for_backward(O, dlogC_dΛ, dlogC_dγ)
        return logC

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        O, dlogC_dΛ, dlogC_dγ = ctx.saved_tensors
        
        dlogC_dB = O @ torch.diag_embed(dlogC_dΛ) @ O.mT
        dlogC_dv = O @ dlogC_dγ.unsqueeze(-1)
        
        grad_B = grad_out.unsqueeze(-1).unsqueeze(-1) * dlogC_dB
        grad_v = grad_out.unsqueeze(-1) * dlogC_dv.squeeze(-1)
        return grad_B, grad_v

class EfficientFB8NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, theta, phi, psi, kappa, beta, eta, alpha, rho):
        params = [theta, phi, psi, kappa, beta, eta, alpha, rho]
        theta, phi, psi, kappa, beta, eta, alpha, rho = [p.squeeze() if p.dim() > 1 else p for p in params]
        
        B, v = fb8_to_Bv(theta, phi, psi, kappa, beta, eta, alpha, rho)
        logC = _FBNormaliserFn.apply(B, v)
        quad = (x.unsqueeze(-2) @ B @ x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        lin  = (v * x).sum(dim=-1)
        nll  = -quad - lin + logC
        return nll.mean()
 
# # -------------------------------------------------------------
# #  Efficient FB8 Negative Log-Likelihood
# #  Method from: "Maximum likelihood estimation of the Fisher-Bingham
# #  distribution via efficient calculation of its normalizing constant"
# #  by Chen et al. (2020)
# # -------------------------------------------------------------

# def fb8_to_Bv(theta: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor,
#               kappa: torch.Tensor, beta: torch.Tensor, eta: torch.Tensor,
#               alpha: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#     """Convert FB8 parameterisation to (B, v) with tr(B)=0."""
#     Gamma = create_matrix_Gamma_torch(theta, phi, psi)
#     nu = spherical_coordinates_to_nu_torch(alpha, rho)
    
#     v = kappa.unsqueeze(-1) * torch.bmm(Gamma, nu.unsqueeze(-1)).squeeze(-1)

#     gamma2 = Gamma[..., :, 1]
#     gamma3 = Gamma[..., :, 2]
#     g2g2t = torch.bmm(gamma2.unsqueeze(-1), gamma2.unsqueeze(-2))
#     g3g3t = torch.bmm(gamma3.unsqueeze(-1), gamma3.unsqueeze(-2))
#     B = beta.unsqueeze(-1).unsqueeze(-1) * (g2g2t - eta.unsqueeze(-1).unsqueeze(-1) * g3g3t)

#     tr = B.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
#     B_traceless = B - tr[..., None] * torch.eye(3, device=B.device, dtype=B.dtype) / 3.0
    
#     return B_traceless, v

# def _cet_fft(Lambda: torch.Tensor, gamma: torch.Tensor, N: int = 400,
#              w_d: float = 0.5, r: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Return log(C), d(logC)/dΛ, d(logC)/dγ for each batch row, using fp64 for stability."""
#     original_dtype = Lambda.dtype
#     Lambda_64 = Lambda.to(torch.float64)
#     gamma_64 = gamma.to(torch.float64)
    
#     B, p = Lambda_64.shape
#     assert p == 3, "Only S² supported (p=3)"

#     Λ_max, Λ_max_indices = Lambda_64.max(dim=1, keepdim=True)
#     Λ_shift  = Lambda_64 - Λ_max
    
#     c  = (15 * math.pi) / (r ** 2 * (1 + r) * w_d)
#     d  = c / 2.0
#     h  = math.sqrt(2 * math.pi * d * (1 + r) / (w_d * N))
#     p1 = math.sqrt(N * h / w_d)
#     p2 = math.sqrt(w_d * N * h / 4.0)

#     ns = torch.arange(-N - 1, N + 1, device=Lambda_64.device, dtype=torch.float64)
#     nh = ns * h
#     w = 0.5 * torch.special.erfc(nh.abs() / p1 - p2)
#     denom = -Λ_shift[:, :, None] + 1j * nh + c

#     exp_fac = torch.exp((gamma_64[:, :, None] ** 2) / (4.0 * denom))
#     sqrt_fac = denom.sqrt()
#     A = (exp_fac / sqrt_fac).prod(dim=1)

#     dA_dΛ = A[:, None, :] * ((-gamma_64[:, :, None]**2) / (4 * denom**2) - 1 / (2 * denom))
#     dA_dγ = A[:, None, :] * (gamma_64[:, :, None] / (2 * denom))

#     common = w * torch.exp(1j * nh)
#     S_complex  = (common * A).sum(dim=-1)
#     SL_complex = (common * dA_dΛ).sum(dim=-1)
#     SG_complex = (common * dA_dγ).sum(dim=-1)

#     log_const = 0.5 * math.log(math.pi) + c + math.log(h)
#     logC = log_const + torch.log(S_complex.real) + Λ_max.squeeze(-1)
    
#     dlogC_dΛ_shifted = (1 / S_complex.real.unsqueeze(-1)) * SL_complex.real
    
#     dΛ_max_dΛ = torch.nn.functional.one_hot(Λ_max_indices.squeeze(-1), num_classes=p).to(dlogC_dΛ_shifted.dtype)
#     grad_sum = dlogC_dΛ_shifted.sum(dim=1, keepdim=True)
#     dlogC_dΛ = dlogC_dΛ_shifted + dΛ_max_dΛ * (1 - grad_sum)

#     dlogC_dγ = (1 / S_complex.real.unsqueeze(-1)) * SG_complex.real

#     return logC.to(original_dtype), dlogC_dΛ.to(original_dtype), dlogC_dγ.to(original_dtype)

# class _FBNormaliserFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, B: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         Λ, O = torch.linalg.eigh(B)
#         γ = (O.mT @ v.unsqueeze(-1)).squeeze(-1)
#         logC, dlogC_dΛ, dlogC_dγ = _cet_fft(Λ, γ)
#         ctx.save_for_backward(O, dlogC_dΛ, dlogC_dγ)
#         return logC

#     @staticmethod
#     def backward(ctx, grad_out: torch.Tensor):
#         O, dlogC_dΛ, dlogC_dγ = ctx.saved_tensors
        
#         dlogC_dB = O @ torch.diag_embed(dlogC_dΛ) @ O.mT
#         dlogC_dv = O @ dlogC_dγ.unsqueeze(-1)
        
#         grad_B = grad_out.unsqueeze(-1).unsqueeze(-1) * dlogC_dB
#         grad_v = grad_out.unsqueeze(-1) * dlogC_dv.squeeze(-1)
#         return grad_B, grad_v

# class EfficientFB8NLLLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, theta, phi, psi, kappa, beta, eta, alpha, rho):
#         params = [theta, phi, psi, kappa, beta, eta, alpha, rho]
#         theta, phi, psi, kappa, beta, eta, alpha, rho = [p.squeeze() if p.dim() > 1 else p for p in params]
        
#         B, v = fb8_to_Bv(theta, phi, psi, kappa, beta, eta, alpha, rho)
#         logC = _FBNormaliserFn.apply(B, v)
#         quad = (x.unsqueeze(-2) @ B @ x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
#         lin  = (v * x).sum(dim=-1)
#         nll  = -quad - lin + logC
#         return nll.mean()

def logmap_sphere(p1: torch.Tensor, p2: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Computes the logarithmic map from p1 to p2 on the S^2 sphere.
    This gives the tangent vector at p1 that points towards p2.
    """
    dot_product = torch.sum(p1 * p2, dim=-1, keepdim=True)
    # Clamp to prevent acos from producing NaNs due to floating point errors
    dot_product = torch.clamp(dot_product, -1.0 + epsilon, 1.0 - epsilon)
    
    angle = torch.acos(dot_product)
    # Project p2 onto the tangent plane at p1
    v = p2 - dot_product * p1
    # Normalize the direction vector
    v = v / torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(epsilon)
    
    # The logmap is the direction vector scaled by the angle (geodesic distance)
    return angle * v

class FB8ScoreMatchingLoss(nn.Module):
    """
    Computes the loss for an FB8 distribution on the S^2 sphere using a
    geometrically-correct Riemannian Denoising Score Matching objective.
    This version uses the Varadhan approximation for the target score.
    """
    def __init__(self, sigma: float = 0.05):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor,
                theta: torch.Tensor, phi: torch.Tensor, psi: torch.Tensor,
                kappa: torch.Tensor, beta: torch.Tensor, eta: torch.Tensor,
                alpha: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Riemannian Denoising Score Matching loss.
        """
        params = [theta, phi, psi, kappa, beta, eta, alpha, rho]
        params_squeezed = [p.squeeze() if p.dim() > 1 else p for p in params]
        B, v = fb8_to_Bv(*params_squeezed)

        # 1. Add noise and project back to the sphere
        noise = torch.randn_like(x) * self.sigma
        x_noisy = x + noise
        x_noisy = x_noisy / torch.norm(x_noisy, p=2, dim=-1, keepdim=True)
        
        # 2. Calculate the model's score (projected Euclidean gradient)
        x_noisy_col = x_noisy.unsqueeze(-1)
        Bx_noisy = torch.bmm(B, x_noisy_col).squeeze(-1)
        score_model_ambient = 2 * Bx_noisy + v
        projection_term_model = torch.sum(score_model_ambient * x_noisy, dim=-1, keepdim=True)
        score_model_riemannian = score_model_ambient - projection_term_model * x_noisy

        # --- START OF VARADHAN APPROXIMATION ---
        
        # 3. Calculate the target score using the Varadhan approximation
        # s_target = logmap_x_noisy(x) / sigma^2
        # The logmap already returns a tangent vector, so no projection is needed.
        score_target_riemannian = logmap_sphere(x_noisy, x) / (self.sigma**2)

        # 4. Calculate the loss. Note: According to the paper's Table 2, this
        #    formulation of the DSM loss does not use the sigma^2 weighting.
        loss_per_sample = torch.sum((score_model_riemannian - score_target_riemannian)**2, dim=-1)
        
        # --- END OF VARADHAN APPROXIMATION ---
        
        return loss_per_sample.mean()