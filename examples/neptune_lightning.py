"""
Neptune Lightning Wrapper for Examples

This creates a PyTorch Lightning wrapper around our clean Neptune model,
preserving the original training interface while using the new package.
"""

import torch
import torch.nn.functional as F
import lightning.pytorch as pl
import numpy as np
from typing import Dict, Any, Tuple, List
from torch import Tensor

from neptune import NeptuneModel
from loss_functions import (
    AngularDistanceLoss, 
    VonMisesFisherLoss, 
    GaussianNLLLoss
)


class NeptuneLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for the clean Neptune model."""
    
    def __init__(
        self,
        in_channels: int = 6,
        num_patches: int = 128,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.1,
        downstream_task: str = 'angular_reco',
        loss_fn: str = 'angular_distance',
        k_neighbors: int = 16,
        pool_method: str = 'max',
        mlp_layers: List[int] = [256, 512, 768],
        batch_size: int = 128, 
        lr: float = 1e-3, 
        lr_schedule: List[int] = [10, 2],
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        # Determine output dimension based on task
        self._set_output_dim()
        self._validate_loss_fn() 
        
        # Create the clean Neptune model
        self.model = NeptuneModel(
            in_channels=self.hparams.in_channels,
            num_patches=self.hparams.num_patches,
            token_dim=self.hparams.token_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout,
            output_dim=self.output_dim,
            k_neighbors=self.hparams.k_neighbors,
            pool_method=self.hparams.pool_method,
            mlp_layers=self.hparams.mlp_layers
        )
        
        # Results storage for test metrics
        self.test_results = {}
        self.test_step_outputs = []
        self.validation_step_outputs = []
        
    def _set_output_dim(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        if task == 'angular_reco':
            self.output_dim = 3
        elif task == 'energy_reco':
            self.output_dim = 2 if loss_choice == 'gaussian_nll' else 1
        else: 
            raise ValueError(f"Unknown task: {task}. Only 'angular_reco' and 'energy_reco' are supported.")

    def _validate_loss_fn(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        valid = {
            'angular_reco': ['angular_distance', 'vmf'],
            'energy_reco': ['gaussian_nll'],
        }
        if task not in valid or loss_choice not in valid[task]:
            raise ValueError(f"Invalid task/loss combo: {task}/{loss_choice}")

    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        """Forward pass through the model."""
        return self.model(coords, features)

    def _get_loss_function(self) -> callable:
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        
        if task == 'angular_reco':
            get_labels = lambda labels: labels[:, 1:4]  # Direction components
            if loss_choice == 'angular_distance':
                return lambda preds, labels: AngularDistanceLoss(preds, get_labels(labels))
            elif loss_choice == 'vmf':
                return lambda preds, labels: VonMisesFisherLoss(preds, get_labels(labels))
        
        elif task == 'energy_reco':
            get_labels = lambda labels: labels[:, 0]  # Energy component
            if loss_choice == 'gaussian_nll': 
                return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], get_labels(labels))
        
        raise ValueError(f"Unhandled task/loss: {task}/{loss_choice}") 

    def step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        coords, features, labels = batch
        preds = self(coords, features)
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        return loss, preds, labels
        
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, _, _ = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        return loss
        
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss, preds, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.validation_step_outputs.append({"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()})
        return {"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()}
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        coords, features, labels = batch
        preds = self(coords, features)
        
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        output = {
            "test_loss": loss.detach(), 
            "preds": preds.detach(), 
            "labels": labels.detach()
        }
        self.test_step_outputs.append(output)
        return output

    def _epoch_end(self, step_outputs: List[Dict[str, Tensor]], stage: str):
        if not step_outputs: 
            return 
        all_preds = torch.cat([x['preds'] for x in step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in step_outputs], dim=0)
        loss_func = self._get_loss_function()
        overall_loss = loss_func(all_preds, all_labels)
        self.log(f'{stage}_loss_epoch', overall_loss, prog_bar=(stage=='val'))
        
        if self.hparams.downstream_task == 'angular_reco':
            true_dirs = all_labels[:, 1:4]
            preds_norm = F.normalize(all_preds, p=2, dim=1)
            angular_errors_rad = AngularDistanceLoss(preds_norm, true_dirs, reduction='none') * np.pi
            median_angular_error_rad = torch.median(angular_errors_rad)
            self.log(f'{stage}_median_angular_error_deg', torch.rad2deg(median_angular_error_rad))
            self.log(f'{stage}_mean_angular_error_deg', torch.rad2deg(angular_errors_rad.mean()))
            
        if self.hparams.downstream_task == 'energy_reco':
            energy_errors = torch.abs(all_preds[:, 0] - all_labels[:, 0])
            self.log(f'{stage}_mean_energy_error', energy_errors.mean())
            
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self._epoch_end(outputs, stage='val')
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self._epoch_end(outputs, stage='test')
        self.test_step_outputs = []

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[1], eta_min=1e-7)
        return [optimizer], [scheduler]