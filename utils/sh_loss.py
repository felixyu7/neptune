"""
Spherical Harmonic Loss Implementation

This module implements a novel loss function for directional uncertainty quantification
on the 2D sphere using spherical harmonic expansions. The approach enables neural networks
to predict rich, non-parametric probability distributions over spherical directions.

Key components:
- SHLoss: Main loss module that computes negative log-likelihood
- SHComponents: Helper class for spherical harmonic operations
- Utility functions for coordinate conversions and interpolation

References:
- See concept.md for theoretical background
- Uses torch-harmonics library for differentiable SH transforms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_harmonics as harmonics
from typing import Tuple
import math


def cartesian_to_spherical(xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert Cartesian coordinates to spherical coordinates (theta, phi).
    
    Args:
        xyz: Tensor of shape (..., 3) containing Cartesian coordinates
        
    Returns:
        theta: Polar angle [0, π] (colatitude)
        phi: Azimuthal angle [0, 2π] (longitude)
    """
    # Use F.normalize for a more concise and potentially faster implementation
    xyz_normalized = F.normalize(xyz, p=2, dim=-1)
    x, y, z = xyz_normalized[..., 0], xyz_normalized[..., 1], xyz_normalized[..., 2]
    
    # Compute spherical coordinates
    theta = torch.acos(torch.clamp(z, -1.0, 1.0))
    phi = torch.atan2(y, x) % (2 * math.pi)  # Ensure phi is in [0, 2π]
    
    return theta, phi


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        theta: Polar angle [0, π]
        phi: Azimuthal angle [0, 2π]
        
    Returns:
        xyz: Tensor of shape (..., 3) containing Cartesian coordinates
    """
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)


class SHComponents(nn.Module):
    """
    Helper class that manages spherical harmonic transforms and quadrature integration.
    """
    
    def __init__(self, l_max: int, n_theta: int, n_lambda: int, grid: str = "legendre-gauss"):
        """
        Initialize spherical harmonic components.
        
        Args:
            l_max: Maximum spherical harmonic degree
            n_theta: Number of theta (colatitude) grid points
            n_lambda: Number of lambda (longitude) grid points  
            grid: Grid type for quadrature ("legendre-gauss" or "equiangular")
        """
        super().__init__()
        
        self.l_max = l_max
        self.n_theta = n_theta
        self.n_lambda = n_lambda
        self.grid = grid
        
        # For real spherical harmonics, the number of coefficients is (l_max + 1)^2
        self.n_coeffs = (l_max + 1)**2
        
        # Initialize inverse SH transform (coefficients -> grid)
        self.inverse_sht = harmonics.InverseRealSHT(n_theta, n_lambda, grid=grid)
        
        # Get quadrature weights for integration
        if grid == "legendre-gauss":
            from torch_harmonics.quadrature import legendre_gauss_weights
            theta_coords, theta_weights = legendre_gauss_weights(n_theta, a=0.0, b=math.pi)
        elif grid == "equiangular":
            from torch_harmonics.quadrature import clenshaw_curtiss_weights
            cost, theta_weights = clenshaw_curtiss_weights(n_theta, a=-1.0, b=1.0)
            theta_coords = np.flip(np.arccos(cost)).copy()
        else:
            raise ValueError(f"Unsupported grid type: {grid}")

        lambda_coords = np.linspace(0, 2 * math.pi, n_lambda, endpoint=False)
        lambda_weights = (2 * math.pi / n_lambda) * np.ones(n_lambda)
        
        # Register as buffers (not parameters) - use double precision for stability
        self.register_buffer('theta_coords', torch.tensor(theta_coords, dtype=torch.float64))
        self.register_buffer('lambda_coords', torch.tensor(lambda_coords, dtype=torch.float64))
        self.register_buffer('theta_weights', torch.tensor(theta_weights, dtype=torch.float64))
        self.register_buffer('lambda_weights', torch.tensor(lambda_weights, dtype=torch.float64))
        
        # Create 2D weight tensor for integration
        theta_2d, lambda_2d = torch.meshgrid(self.theta_coords, self.lambda_coords, indexing='ij')
        integration_weights = (self.theta_weights[:, None] * self.lambda_weights[None, :] * 
                             torch.sin(theta_2d))
        self.register_buffer('integration_weights', integration_weights)
        
        # Create coordinate grids for interpolation
        self.register_buffer('theta_grid', theta_2d)
        self.register_buffer('lambda_grid', lambda_2d)
    
    def coefficients_to_grid(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Convert SH coefficients to function values on the spherical grid.
        
        Args:
            coefficients: Tensor of shape (..., n_coeffs) containing SH coefficients
            
        Returns:
            grid_values: Tensor of shape (..., n_theta, n_lambda) containing function values
        """
        batch_shape = coefficients.shape[:-1]
        lmax = self.inverse_sht.lmax
        mmax = self.inverse_sht.mmax
        
        coeff_tensor = torch.zeros(*batch_shape, lmax, mmax, 
                                  dtype=torch.complex128, device=coefficients.device)
        
        # Map from flat coefficient array to complex tensor for torch-harmonics
        # The format is (l, m) indexing for real spherical harmonics.
        # m=0 terms are real, m>0 terms are complex (cos and sin parts).
        coeff_idx = 0
        for l in range(self.l_max + 1):
            if l >= lmax: continue
            
            # m = 0 term (real)
            if coeff_idx < self.n_coeffs:
                coeff_tensor[..., l, 0] = coefficients[..., coeff_idx].to(torch.complex128)
                coeff_idx += 1
            
            # m > 0 terms (complex)
            for m in range(1, l + 1):
                if m >= mmax: continue
                
                if coeff_idx + 1 < self.n_coeffs:
                    real_part = coefficients[..., coeff_idx]
                    imag_part = coefficients[..., coeff_idx + 1]
                    coeff_tensor[..., l, m] = torch.complex(real_part, imag_part)
                    coeff_idx += 2
        
        grid_values = self.inverse_sht(coeff_tensor)
        return torch.real(grid_values)
    
    def integrate_over_sphere(self, values: torch.Tensor) -> torch.Tensor:
        """
        Integrate function values over the sphere using quadrature.
        
        Args:
            values: Tensor of shape (..., n_theta, n_lambda) containing function values
            
        Returns:
            integral: Tensor of shape (...) containing integrated values
        """
        weighted_values = values * self.integration_weights
        integral = torch.sum(weighted_values, dim=(-2, -1))
        return integral
    
    def interpolate_at_point(self, grid_values: torch.Tensor, theta: torch.Tensor,
                           phi: torch.Tensor) -> torch.Tensor:
        """
        Interpolate grid values at specific spherical coordinates using F.grid_sample.
        
        Args:
            grid_values: Tensor of shape (..., n_theta, n_lambda)
            theta: Polar angle [0, π]
            phi: Azimuthal angle [0, 2π]
            
        Returns:
            interpolated: Tensor of shape (...) containing interpolated values
        """
        batch_shape = grid_values.shape[:-2]
        grid_values_unsqueezed = grid_values.unsqueeze(-3)
        
        # Pad the grid for circular interpolation in the phi (longitude) dimension
        grid_padded = F.pad(grid_values_unsqueezed.float(), (0, 1, 0, 0), mode='circular')

        # Normalize coordinates to [-1, 1] for grid_sample
        y_norm = (theta / math.pi) * 2 - 1
        x_norm = (phi / math.pi) - 1
        
        # Create sampling grid
        sampling_grid = torch.stack([x_norm, y_norm], dim=-1)
        while len(sampling_grid.shape) < 4:
            sampling_grid = sampling_grid.unsqueeze(1)
            
        # Perform interpolation
        interpolated = F.grid_sample(
            grid_padded,
            sampling_grid.float(),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        result = interpolated.squeeze()
        if batch_shape:
            result = result.view(*batch_shape)
            
        return result


class SHLoss(nn.Module):
    """
    Spherical Harmonic Loss Module
    """
    
    def __init__(self, l_max: int = 4, n_theta: int = 64, n_lambda: int = 128, 
                 grid: str = "legendre-gauss", eps: float = 1e-8):
        """
        Initialize the SH Loss module.
        
        Args:
            l_max: Maximum spherical harmonic degree
            n_theta: Number of theta grid points for integration
            n_lambda: Number of lambda grid points for integration
            grid: Grid type for quadrature
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        
        self.l_max = l_max
        self.eps = eps
        self.sh_components = SHComponents(l_max, n_theta, n_lambda, grid)
        
    def forward(self, sh_coefficients: torch.Tensor, target_directions: torch.Tensor) -> torch.Tensor:
        """
        Compute the spherical harmonic loss.
        
        Args:
            sh_coefficients: Tensor of shape (batch_size, n_coeffs)
            target_directions: Tensor of shape (batch_size, 3)
            
        Returns:
            loss: Tensor of shape (batch_size,) containing NLL
        """
        # Use float64 for stability
        sh_coefficients_double = sh_coefficients.to(torch.float64)
        target_directions_double = target_directions.to(torch.float64)

        # Step 1: Convert SH coefficients to unnormalized log-density f(y) on grid
        f_values = self.sh_components.coefficients_to_grid(sh_coefficients_double)
        
        # Step 2: Compute normalization constant Z = ∫ exp(f(y)) dy
        exp_f_values = torch.exp(f_values)
        Z = self.sh_components.integrate_over_sphere(exp_f_values)
        
        # Step 3: Convert target directions to spherical coordinates
        theta_target, phi_target = cartesian_to_spherical(target_directions_double)
        
        # Step 4: Evaluate f(y_target) at target directions via interpolation
        f_target = self.sh_components.interpolate_at_point(f_values, theta_target, phi_target)
        
        # Step 5: Compute log probability log(P(y_target)) = f(y_target) - log(Z)
        log_prob = f_target - torch.log(Z + self.eps)
        
        # Step 6: Return negative log-likelihood loss
        return -log_prob
    
    def predict_distribution(self, sh_coefficients: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the probability distribution from SH coefficients.
        
        Args:
            sh_coefficients: Tensor of shape (batch_size, n_coeffs)
            
        Returns:
            prob_distribution: Tensor of shape (batch_size, n_theta, n_lambda)
            coordinates: Tuple of (theta_grid, lambda_grid)
        """
        sh_coefficients_double = sh_coefficients.to(torch.float64)
        f_values = self.sh_components.coefficients_to_grid(sh_coefficients_double)
        
        exp_f_values = torch.exp(f_values)
        Z = self.sh_components.integrate_over_sphere(exp_f_values)
        
        prob_distribution = exp_f_values / (Z[..., None, None] + self.eps)
        
        coordinates = (self.sh_components.theta_grid, self.sh_components.lambda_grid)
        
        return prob_distribution, coordinates