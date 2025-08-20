import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from torch import Tensor
from torchmetrics.classification import BinaryAccuracy
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple, Dict, Any, Optional
import os
import fpsample
import time
import wandb

# Local imports
from neptune.utils import (
    farthest_point_sampling, 
    AngularDistanceLoss, 
    VonMisesFisherLoss, 
    LogCoshLoss, 
    GaussianNLLLoss,
    CombinedAngularVMFDistanceLoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss
)
from neptune.models.transformer_layers import (
    RelativePosTransformerEncoderLayer,
    RelativePosTransformerEncoder
)

class PointCloudTokenizer(nn.Module):
    """Converts point cloud data into tokens for transformer processing."""
    def __init__(self, 
                 feature_dim: int, 
                 max_tokens: int = 128, 
                 token_dim: int = 768, 
                 mlp_layers: List[int] = [256, 512, 768],
                 k_neighbors: int = 16, 
                 pool_method: str = 'max'):
        super().__init__()
        if pool_method not in ['max', 'mean']:
            raise ValueError(f"pool_method must be 'max' or 'mean', got {pool_method}")
            
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.k_neighbors = k_neighbors
        self.pool_method = pool_method
        
        # Per-Point Feature MLP
        mlp = []
        in_dim = feature_dim
        for out_dim in mlp_layers:
            mlp.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = out_dim
        mlp.append(nn.Linear(in_dim, token_dim))
        self.mlp = nn.Sequential(*mlp)
        
        # Neighborhood Aggregation MLP
        self.neighborhood_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )
        
    def forward(self, coordinates: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch_indices = coordinates[:, 0].long()
        xyz = coordinates[:, 1:4] 
        time = coordinates[:, 4:5]

        batch_size = batch_indices.max().item() + 1 if coordinates.numel() > 0 else 0
        if batch_size == 0:
            # Handle empty input
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=coordinates.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=coordinates.device, dtype=coordinates.dtype)
            empty_masks = torch.zeros((0, self.max_tokens), device=coordinates.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks

        all_tokens = []
        all_centroids = []
        all_masks = []
        
        # Process each point cloud in the batch individually
        for batch_idx in range(batch_size):
            batch_mask_indices = batch_indices == batch_idx
            batch_xyz = xyz[batch_mask_indices]  # [M, 3]
            batch_features_data = features[batch_mask_indices]  # [M, F]
            batch_time = time[batch_mask_indices]  # [M, 1]
            num_points = batch_xyz.shape[0]
            
            if num_points == 0:
                # Handle empty point clouds within the batch
                batch_tokens = torch.zeros(self.max_tokens, self.token_dim, device=coordinates.device, dtype=features.dtype)
                batch_centroids = torch.zeros(self.max_tokens, 4, device=coordinates.device, dtype=coordinates.dtype)
                batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
                all_tokens.append(batch_tokens)
                all_centroids.append(batch_centroids)
                all_masks.append(batch_mask_valid.unsqueeze(0))
                continue

            # Combine spatial coordinates and time for FPS and neighborhood search: [M, 4]
            points_for_sampling = torch.cat([batch_xyz, batch_time], dim=-1)
            
            # Apply Per-Point MLP
            all_point_features = self.mlp(batch_features_data)
            
            if num_points <= self.max_tokens:
                # Use all points if fewer than max_tokens
                batch_centroids = points_for_sampling
                batch_tokens = all_point_features
                num_valid_tokens = num_points
            else:
                # Select centroids using Farthest Point Sampling (bucket_fps_kdline_sampling)
                fps_indices = fpsample.bucket_fps_kdline_sampling(points_for_sampling.detach().cpu().numpy(), self.max_tokens, h=3)
                batch_centroids = points_for_sampling[fps_indices]  # [max_tokens, 4]
                
                # Find k-Nearest Neighbors for each centroid
                dist_matrix = torch.cdist(batch_centroids, points_for_sampling)
                k = min(self.k_neighbors, num_points)
                _, knn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)
                
                # Gather features
                flat_knn_indices = knn_indices.view(-1)
                gathered_features = all_point_features[flat_knn_indices]
                neighborhood_features = gathered_features.view(self.max_tokens, k, self.token_dim)
                                
                # Pool features
                if self.pool_method == 'max':
                    pooled_features = torch.max(neighborhood_features, dim=1)[0]
                else: # self.pool_method == 'mean'
                    pooled_features = torch.mean(neighborhood_features, dim=1)
                
                # Apply aggregation MLP
                batch_tokens = self.neighborhood_mlp(pooled_features)
                num_valid_tokens = self.max_tokens
            
            # Padding
            if num_valid_tokens < self.max_tokens:
                num_padding = self.max_tokens - num_valid_tokens
                pad_tokens = torch.zeros((num_padding, self.token_dim), device=batch_tokens.device, dtype=batch_tokens.dtype)
                batch_tokens = torch.cat([batch_tokens, pad_tokens], dim=0)
                pad_centroids = torch.zeros((num_padding, 4), device=batch_centroids.device, dtype=batch_centroids.dtype)
                batch_centroids = torch.cat([batch_centroids, pad_centroids], dim=0)
            
            # Create boolean mask
            batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
            batch_mask_valid[:num_valid_tokens] = True
            
            all_tokens.append(batch_tokens)
            all_centroids.append(batch_centroids)
            all_masks.append(batch_mask_valid.unsqueeze(0))
        
        # Stack results
        tokens = torch.stack(all_tokens, dim=0)
        centroids = torch.stack(all_centroids, dim=0)
        masks = torch.cat(all_masks, dim=0)
        return tokens, centroids, masks

class PointTransformerEncoder(nn.Module):
    """Transformer encoder for point cloud tokens."""
    def __init__(self, token_dim=768, num_layers=12, num_heads=12,
                 hidden_dim=3072, dropout=0.1, pre_norm=False):
        super().__init__()
        self.token_dim = token_dim
        # Position embedding component
        self.pos_embed = PositionEmbedding(out_dim=token_dim)
        
        # # Transformer layers
        # encoder_layer = TransformerEncoderLayer(
        #     d_model=token_dim,
        #     nhead=num_heads,
        #     dim_feedforward=hidden_dim,
        #     dropout=dropout,
        #     activation='gelu',
        #     batch_first=True,
        #     norm_first=False
        # )
        # self.layers = TransformerEncoder(encoder_layer, num_layers)
        encoder_layer = RelativePosTransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            learnable_geom=True,
            pre_norm=pre_norm
        )
        self.layers = RelativePosTransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output normalization
        self.ln = nn.LayerNorm(token_dim)
    
    def forward(self, tokens, centroids, masks=None):
        """Process tokens through transformer architecture."""
        # Add positional embeddings
        pos_embed_out = self.pos_embed(centroids)
        tokens = tokens + pos_embed_out

        # Apply transformer layers
        if masks is not None:
            attention_mask = ~masks
            tokens = self.layers(tokens, centroids, src_key_padding_mask=attention_mask)
        else:
            tokens = self.layers(tokens, centroids)
        
        # Global average pooling
        if masks is not None:
            valid_tokens = tokens * masks.unsqueeze(-1)
            token_sum = valid_tokens.sum(dim=1)
            valid_count = masks.sum(dim=1, keepdim=True).clamp(min=1)
            global_features = token_sum / valid_count
        else:
            global_features = tokens.mean(dim=1)
        
        # Apply final normalization
        return self.ln(global_features)

class PositionEmbedding(nn.Module):
    """MLP for encoding position information into tokens."""
    def __init__(self, in_dim=4, hidden_dims=[64, 256, 768], out_dim=768):
        super().__init__()
        layers = []
        last_dim = in_dim
        # Build MLP with Linear -> GELU -> LayerNorm pattern
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim)
            ])
            last_dim = hidden_dim
        # Final projection layer
        layers.append(nn.Linear(last_dim, out_dim)) 
        self.mlp = nn.Sequential(*layers)

    def forward(self, centroids: Tensor) -> Tensor:
        """Args: centroids [B, N, 4]"""
        return self.mlp(centroids)

class Neptune(pl.LightningModule):
    """Point cloud transformer model for reconstruction tasks."""
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
        pre_norm: bool = False,
        mlp_layers: List[int] = [256, 512, 768],
        batch_size: int = 128, 
        lr: float = 1e-3, 
        lr_schedule: List[int] = [10, 2],
        weight_decay: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model Components
        self.tokenizer = PointCloudTokenizer(
            feature_dim=self.hparams.in_channels,
            max_tokens=self.hparams.num_patches,
            token_dim=self.hparams.token_dim,
            mlp_layers=self.hparams.mlp_layers,
            k_neighbors=self.hparams.k_neighbors,
            pool_method=self.hparams.pool_method
        )
        self.encoder = PointTransformerEncoder(
            token_dim=self.hparams.token_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout,
            pre_norm=self.hparams.pre_norm
        )
        
        # Determine output dimension based on task
        self._set_output_dim()
        self._validate_loss_fn() 
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hparams.token_dim, self.hparams.token_dim),
            nn.GELU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.token_dim, self.output_dim)
        )
        
        # Results storage for test metrics
        self.test_results = {}
        # Manual storage for test outputs when not in training mode
        self.test_step_outputs = []
        self.validation_step_outputs = []
        
        if self.hparams.downstream_task == 'background_classification':
            self.accuracy = BinaryAccuracy(threshold=0.5)
        
    def _set_output_dim(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        if task == 'angular_reco': 
            self.output_dim = 3
        elif task == 'energy_reco' or task == 'visible_energy_reco': 
            self.output_dim = 2 if loss_choice == 'gaussian_nll' else 1
        elif task == 'morphology_classification':
            self.output_dim = 6
        elif task == 'simple_morphology_classification':
            self.output_dim = 4
        elif task == 'bundleness_classification':
            self.output_dim = 3
        elif task == 'background_classification':
            self.output_dim = 1  # binary
        
        else: raise ValueError(f"Unknown task: {task}")

    def _validate_loss_fn(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        valid = {
            'angular_reco': ['angular_distance', 'vmf', 'combined_angular_vmf'],
            'energy_reco': ['log_cosh', 'gaussian_nll'],
            'visible_energy_reco': ['log_cosh', 'gaussian_nll'],
            'morphology_classification': ['cross_entropy'],
            'simple_morphology_classification': ['cross_entropy'],
            'bundleness_classification': ['cross_entropy'],
            'background_classification': ['binary_cross_entropy'],
        }
        if task not in valid or loss_choice not in valid[task]:
            raise ValueError(f"Invalid task/loss combo: {task}/{loss_choice}")

    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        """Forward pass through the model."""
        # 1. Tokenize
        tokens, centroids, masks = self.tokenizer(coords, features)
        # 2. Encode
        global_features = self.encoder(tokens, centroids, masks)
        # 3. Classify
        output = self.classifier(global_features)
        return output

    def _get_loss_function(self) -> callable:
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        if task == 'angular_reco':
            get_labels = lambda labels: labels[:, 1:4]
            if loss_choice == 'angular_distance': 
                return lambda preds, labels: AngularDistanceLoss(preds, get_labels(labels))
            if loss_choice == 'vmf': 
                return lambda preds, labels: VonMisesFisherLoss(preds, get_labels(labels))
            if loss_choice == 'combined_angular_vmf': 
                return lambda preds, labels: CombinedAngularVMFDistanceLoss(preds, get_labels(labels), angular_weight=0.5)
        
        elif task == 'energy_reco':
            get_labels = lambda labels: labels[:, 0]
            if loss_choice == 'log_cosh': 
                return lambda preds, labels: LogCoshLoss(preds.squeeze(-1) if preds.dim() > 1 else preds, get_labels(labels))
            if loss_choice == 'gaussian_nll': 
                return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], get_labels(labels))
        
        elif task == 'visible_energy_reco':
            get_labels = lambda labels: labels[:, 7]
            if loss_choice == 'log_cosh': 
                return lambda preds, labels: LogCoshLoss(preds.squeeze(-1) if preds.dim() > 1 else preds, get_labels(labels))
            if loss_choice == 'gaussian_nll': 
                return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], get_labels(labels))
        
        elif task == 'morphology_classification':
            # 0 = cascade
            # 1 = thru track
            # 2 = starting track
            # 3 = stopping track
            # 4 = passing track
            # 5 = bundle/multiple events
            num_classes = 6
            get_labels = lambda labels: labels[:, 4]
            return lambda preds, labels: CrossEntropyLoss(preds, get_labels(labels).long())
        
        elif task == 'simple_morphology_classification':
            # 0 = cascade
            # 1 = thru track
            # 2 = starting track
            # 3 = passing track
            num_classes = 4
            def get_labels(labels):
                morph = labels[:, 4]
                morph[morph==4] = 3
            return lambda preds, labels: CrossEntropyLoss(preds, get_labels(labels).long())

        elif task == 'bundleness_classification':
            # 0 = cascade
            # 1 = single track
            # 2 = bundle/multiple events
            num_classes = 3
            
            get_labels = lambda labels: labels[:, 5]
            return lambda preds, labels: CrossEntropyLoss(preds, get_labels(labels))
        
        elif task == 'background_classification':
            # 0 = neutrino, 1 = CORSIKA
            num_classes = 2
            
            get_labels = lambda labels: labels[:, 6]
            return lambda preds, labels: BinaryCrossEntropyLoss(preds.squeeze(-1), get_labels(labels))

        
        raise ValueError(f"Unhandled task/loss: {task}/{loss_choice}") 

    def step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        coords, features, labels = batch
        preds = self(coords, features)
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        return loss, preds, labels
        
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, preds, labels = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
        probs = torch.sigmoid(preds).detach().cpu().float().numpy()
        self.logger.experiment.log(
            {"train/p_positive_hist": wandb.Histogram(probs)},
            commit=False
        )
        return loss
        
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        loss, preds, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        self.validation_step_outputs.append({"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()})
        return {"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()}
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        coords, features, labels = batch
        start_time = time.time()
        preds = self(coords, features)
        end_time = time.time()
        forward_time = end_time - start_time
        
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
        output = {
            "test_loss": loss.detach(), 
            "preds": preds.detach(), 
            "labels": labels.detach(),
            "forward_time": torch.tensor(forward_time, device=loss.device)
        }
        self.test_step_outputs.append(output)
        return output

    def _epoch_end(self, step_outputs: List[Dict[str, Tensor]], stage: str):
        if not step_outputs: return 
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
        if self.hparams.downstream_task == 'background_classification':
            true_label = all_labels[:, 6]
            probs = torch.sigmoid(all_preds.view(-1))     # P(y=1)
            preds = (probs >= 0.5).long()
            
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        self._epoch_end(outputs, stage='val')
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        self._epoch_end(outputs, stage='test')
        
        # Process and save test results
        if not outputs:
            print("WARNING: No test outputs were collected. Results will not be saved.")
            return
            
        all_preds = torch.cat([x['preds'] for x in outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in outputs], dim=0)
        all_forward_times = torch.cat([x['forward_time'].unsqueeze(0) for x in outputs], dim=0)

        # Convert to numpy arrays for saving
        preds_np = all_preds.detach().cpu().numpy()
        labels_np = all_labels.detach().cpu().numpy()
        forward_times_np = all_forward_times.detach().cpu().numpy()

        self.test_results = {}

        if self.hparams.downstream_task == 'angular_reco':
            # Collect angular reconstruction results
            truth = labels_np[:, 1:4]  # Direction components
            
            # Calculate angular differences
            preds_norm = F.normalize(all_preds, p=2, dim=1)
            preds_norm_np = preds_norm.detach().cpu().numpy()
            
            # Compute directions and metrics
            angle_errors_rad = AngularDistanceLoss(preds_norm, truth, reduction='none') * np.pi
            angle_errors_deg = torch.rad2deg(angle_errors_rad)
            angle_errors_deg_np = angle_errors_deg.detach().cpu().numpy()
            
            # Store results
            self.test_results['angle_diff'] = angle_errors_deg_np
            self.test_results['true_e'] = labels_np[:, 0]  # Energy component
            
            # Calculate kappa (magnitude of direction vectors)
            self.test_results['kappa'] = np.linalg.norm(preds_np, axis=1)
            
            # Calculate zenith and azimuth angles
            pred_zenith = np.arccos(preds_norm_np[:, 2])
            pred_azimuth = np.arctan2(preds_norm_np[:, 1], preds_norm_np[:, 0])
            self.test_results['pred_zenith'] = pred_zenith
            self.test_results['pred_azimuth'] = pred_azimuth
            
            # True zenith and azimuth
            true_norm = np.linalg.norm(truth, axis=1, keepdims=True)
            true_norm = np.where(true_norm == 0, 1.0, true_norm)  # Prevent division by zero
            truth_norm = truth / true_norm
            true_zenith = np.arccos(truth_norm[:, 2])
            true_azimuth = np.arctan2(truth_norm[:, 1], truth_norm[:, 0])
            self.test_results['true_zenith'] = true_zenith
            self.test_results['true_azimuth'] = true_azimuth
            
        elif self.hparams.downstream_task == 'energy_reco':
            # Collect energy reconstruction results
            if self.hparams.loss_fn == 'gaussian_nll':
                self.test_results['preds_mean'] = preds_np[:, 0]
                self.test_results['preds_var'] = preds_np[:, 1]
            else:
                self.test_results['preds'] = preds_np.squeeze()
            self.test_results['truth'] = labels_np[:, 0]
        
        # Add forward times to the results
        self.test_results['forward_time'] = forward_times_np

        # Save results to file
        if self.logger is not None:
            save_path = f"./results/{self.logger.name}_{self.logger.version}_results.npy"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, self.test_results)
            print(f"Test results saved to {save_path}")
            
        self.test_step_outputs = []

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[1], eta_min=1e-7)
        return [optimizer], [scheduler]