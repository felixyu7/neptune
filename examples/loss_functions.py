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
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py] Modified to
    use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indicated in [1812.04616] in spite of suggestion
    in Sec. 8.2 of this paper. The change has been validated through comparison
    with exact calculations for `m=2` and `m=3` and found to yield the correct
    results.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa))
            / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )

def log_cmk_exact(m: int, kappa: Tensor) -> Tensor:
    """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
    return LogCMK.apply(m, kappa)

def log_cmk_approx(m: int, kappa: Tensor) -> Tensor:
    """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

    [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
    """
    v = m / 2.0 - 0.5
    a = torch.sqrt((v + 1) ** 2 + kappa**2)
    b = v - 1
    return -a + b * torch.log(b + a)

def log_cmk(m: int, kappa: Tensor, kappa_switch: float = 100.0) -> Tensor:
    """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

    Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
    precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
    this method automatically switches between the two at `kappa_switch`,
    ensuring continuity at this point.
    """
    kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
    mask_exact = kappa < kappa_switch

    # Ensure continuity at `kappa_switch`
    offset = log_cmk_approx(m, kappa_switch) - log_cmk_exact(
        m, kappa_switch
    )
    ret = log_cmk_approx(m, kappa) - offset
    ret[mask_exact] = log_cmk_exact(m, kappa[mask_exact])
    return ret

def VonMisesFisherLoss(prediction: Tensor, target: Tensor) -> Tensor:
    """
    Stable Von Mises-Fisher negative log-likelihood (matches reference implementation).
    Loss = -log C_m(kappa) - kappa * <mu_hat, x> + reg_factor * kappa.
    """
    # Check(s)
    assert prediction.dim() == 2
    assert target.dim() == 2
    assert prediction.size() == target.size()

    # Computing loss
    m = target.size()[1]
    k = torch.norm(prediction, dim=1)
    dotprod = torch.sum(prediction * target, dim=1)
    elements = -log_cmk(m, k) - dotprod

    return elements.mean()