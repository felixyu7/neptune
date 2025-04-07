import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple, Dict, Any, Optional

# Local imports
from neptune.utils import (
    farthest_point_sampling, 
    AngularDistanceLoss, 
    VonMisesFisherLoss, 
    LogCoshLoss, 
    GaussianNLLLoss
)

class PointCloudTokenizer(nn.Module):
    """Tokenizes irregular point clouds into a fixed number of tokens using FPS and k-NN.

    Applies a per-point MLP, selects centroids via Farthest Point Sampling (FPS),
    finds k-Nearest Neighbors (k-NN) for each centroid, aggregates neighborhood
    features using pooling, and applies a final MLP to produce tokens.
    Handles padding for point clouds smaller than max_tokens.
    """
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
        
        # --- Per-Point Feature MLP ---
        point_mlp = []
        in_dim = feature_dim
        for layer_dim in mlp_layers:
            point_mlp.extend([
                nn.Linear(in_dim, layer_dim),
                nn.ReLU(inplace=True)
            ])
            in_dim = layer_dim
        point_mlp.append(nn.Linear(in_dim, token_dim))
        self.point_feature_mlp = nn.Sequential(*point_mlp)
        
        # --- Neighborhood Aggregation MLP ---
        self.neighborhood_agg_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim)
        )
        
    def forward(self, coordinates: Tensor, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            coordinates: Tensor of shape [N, 5] (batch_idx, x, y, z, time).
            features: Tensor of shape [N, F].
            
        Returns:
            Tuple containing:
                - tokens: Tensor of shape [B, max_tokens, token_dim].
                - centroids: Tensor of shape [B, max_tokens, 4] (x, y, z, time).
                - masks: Bool tensor of shape [B, max_tokens] (True for valid tokens).
        """
        batch_indices = coordinates[:, 0].long()
        # Assuming batch indices are contiguous and start from 0
        batch_size = batch_indices.max().item() + 1 if coordinates.numel() > 0 else 0
        if batch_size == 0:
             # Handle empty input gracefully
            empty_tokens = torch.zeros((0, self.max_tokens, self.token_dim), device=coordinates.device, dtype=features.dtype)
            empty_centroids = torch.zeros((0, self.max_tokens, 4), device=coordinates.device, dtype=coordinates.dtype)
            empty_masks = torch.zeros((0, self.max_tokens), device=coordinates.device, dtype=torch.bool)
            return empty_tokens, empty_centroids, empty_masks
            
        all_tokens = []
        all_centroids = []
        all_masks = []
        
        # Process each point cloud in the batch individually
        for batch_idx in range(batch_size):
            batch_mask = batch_indices == batch_idx
            xyz = coordinates[batch_mask, 1:4]  # [M, 3]
            batch_features = features[batch_mask]  # [M, F]
            time = coordinates[batch_mask, 4:5]  # [M, 1]
            num_points = xyz.shape[0]
            
            # --- Handle Empty Point Clouds within Batch ---
            if num_points == 0:
                batch_tokens = torch.zeros(self.max_tokens, self.token_dim, device=coordinates.device, dtype=features.dtype)
                batch_centroids = torch.zeros(self.max_tokens, 4, device=coordinates.device, dtype=coordinates.dtype)
                batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
                all_tokens.append(batch_tokens)
                all_centroids.append(batch_centroids)
                all_masks.append(batch_mask_valid.unsqueeze(0))
                continue
            
            # Combine spatial coordinates and time for FPS and neighborhood search: [M, 4]
            points_for_sampling = torch.cat([xyz, time], dim=-1)
            
            # --- Apply Per-Point MLP ---
            point_embeddings = self.point_feature_mlp(batch_features)  # [M, token_dim]
            
            # --- Select Centroids and Aggregate Neighborhoods ---
            if num_points <= self.max_tokens:
                # Use all points if fewer than max_tokens
                batch_centroids = points_for_sampling
                batch_tokens = point_embeddings # Use per-point embeddings directly
                num_valid_tokens = num_points
            else:
                # Select centroids using Farthest Point Sampling
                fps_indices = farthest_point_sampling(points_for_sampling, self.max_tokens)
                batch_centroids = points_for_sampling[fps_indices]  # [max_tokens, 4]
                
                # Find k-Nearest Neighbors for each centroid
                # Pairwise distances: [max_tokens, M]
                dist_matrix = torch.cdist(batch_centroids, points_for_sampling)  
                
                # Determine actual k (cannot exceed number of points)
                k = min(self.k_neighbors, num_points)
                # Get indices of k nearest neighbors: [max_tokens, k]
                _, knn_indices = torch.topk(dist_matrix, k=k, largest=False, dim=1)  
                
                # Gather features of neighbors: [max_tokens, k, token_dim]
                # knn_indices needs to be expanded for gather
                # expanded_knn_indices = knn_indices.unsqueeze(-1).expand(-1, -1, self.token_dim)
                # neighborhood_features = point_embeddings.gather(0, expanded_knn_indices) # Incorrect gather dim
                
                # Correct gather using flat indices
                flat_knn_indices = knn_indices.view(-1) #[max_tokens * k]
                gathered_features = point_embeddings[flat_knn_indices] #[max_tokens * k, token_dim]
                neighborhood_features = gathered_features.view(self.max_tokens, k, self.token_dim) #[max_tokens, k, token_dim]
                                
                # Pool features within each neighborhood
                if self.pool_method == 'max':
                    pooled_features = torch.max(neighborhood_features, dim=1)[0]  # [max_tokens, token_dim]
                else: # self.pool_method == 'mean'
                    pooled_features = torch.mean(neighborhood_features, dim=1)  # [max_tokens, token_dim]
                
                # Apply aggregation MLP
                batch_tokens = self.neighborhood_agg_mlp(pooled_features)
                num_valid_tokens = self.max_tokens
            
            # --- Padding --- 
            # Pad tokens and centroids if fewer than max_tokens were generated
            if num_valid_tokens < self.max_tokens:
                num_padding = self.max_tokens - num_valid_tokens
                pad_tokens = torch.zeros((num_padding, self.token_dim),
                                          device=batch_tokens.device, dtype=batch_tokens.dtype)
                batch_tokens = torch.cat([batch_tokens, pad_tokens], dim=0)
                
                pad_centroids = torch.zeros((num_padding, 4),
                                           device=batch_centroids.device, dtype=batch_centroids.dtype)
                batch_centroids = torch.cat([batch_centroids, pad_centroids], dim=0)
            
            # Create boolean mask indicating valid (non-padded) tokens
            batch_mask_valid = torch.zeros(self.max_tokens, dtype=torch.bool, device=coordinates.device)
            batch_mask_valid[:num_valid_tokens] = True
            
            all_tokens.append(batch_tokens)
            all_centroids.append(batch_centroids)
            all_masks.append(batch_mask_valid.unsqueeze(0))
        
        # Stack results from all batches
        tokens = torch.stack(all_tokens, dim=0)
        centroids = torch.stack(all_centroids, dim=0)
        masks = torch.cat(all_masks, dim=0)
        return tokens, centroids, masks
    
class PointTransformerEncoder(nn.Module):
    """Transformer Encoder block for processing point cloud tokens."""
    def __init__(self, 
                 token_dim: int = 768, 
                 num_layers: int = 12, 
                 num_heads: int = 12,
                 hidden_dim: int = 3072, 
                 dropout: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.pos_embed = PositionEmbedding(in_dim=4, out_dim=token_dim)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=token_dim,       # Input feature dimension
            nhead=num_heads,         # Number of attention heads
            dim_feedforward=hidden_dim, # Dimension of feedforward network
            dropout=dropout,         # Dropout probability
            activation='gelu',       # Activation function
            batch_first=True         # Input/output shape (batch, seq, feature)
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.output_norm = nn.LayerNorm(token_dim)
    
    def forward(self, tokens: Tensor, centroids: Tensor, masks: Optional[Tensor]=None) -> Tensor:
        """
        Args:
            tokens: Input tokens [B, num_tokens, token_dim].
            centroids: Centroid coordinates corresponding to tokens [B, num_tokens, 4].
            masks: Boolean mask indicating valid tokens [B, num_tokens] (True=valid).
            
        Returns:
            Global feature representation pooled from tokens [B, token_dim].
        """
        # Add positional embeddings derived from centroids
        positional_embeddings = self.pos_embed(centroids)  # [B, num_tokens, token_dim]
        tokens_with_pos = tokens + positional_embeddings

        # Prepare attention mask (True values indicate positions to be ignored)
        # PyTorch TransformerEncoderLayer expects True for padding
        padding_mask = ~masks if masks is not None else None 
            
        # Pass through transformer layers
        encoded_tokens = self.transformer_encoder(tokens_with_pos, src_key_padding_mask=padding_mask)
        
        # Global average pooling over valid tokens
        if masks is not None:
            # Zero out padded tokens before summing
            valid_encoded_tokens = encoded_tokens * masks.unsqueeze(-1)
            # Sum features of valid tokens
            token_sum = valid_encoded_tokens.sum(dim=1) 
            # Count valid tokens per batch item, ensure > 0 for division
            valid_count = masks.sum(dim=1, keepdim=True).clamp(min=1.0) 
            global_features = token_sum / valid_count
        else:
            # If no mask, average pool all tokens
            global_features = encoded_tokens.mean(dim=1)
        
        # Final layer normalization
        return self.output_norm(global_features)

class PositionEmbedding(nn.Module):
    """MLP-based positional embedding network for 4D centroid coordinates (x,y,z,t)."""
    def __init__(self, 
                 in_dim: int = 4, 
                 hidden_dims: List[int] = [64, 128, 256], # Adjusted hidden dims for example
                 out_dim: int = 768):
        super().__init__()
        layers = []
        current_dim = in_dim
        mlp_dims = hidden_dims + [out_dim] # Add output dim to the list
        
        for i, dim in enumerate(mlp_dims):
            layers.append(nn.Linear(current_dim, dim))
            # Apply activation and normalization to all layers except the last
            if i < len(mlp_dims) - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm(dim))
            current_dim = dim
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, centroids: Tensor) -> Tensor:
        """Args: centroids [B, N, 4]"""
        return self.mlp(centroids)

class Neptune(pl.LightningModule):
    """Neptune: Efficient Point Transformer for Neutrino Event Reconstruction.
    
    Combines a PointCloudTokenizer, PointTransformerEncoder, and a task-specific MLP head.
    Configurable for different reconstruction tasks (angular, energy) and loss functions.
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_patches: int = 128,
        token_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.1,
        downstream_task: str = 'angular_reco', # 'angular_reco' or 'energy_reco'
        loss_fn: str = 'angular_distance', # See _validate_loss_fn for options per task
        k_neighbors: int = 16,
        pool_method: str = 'max', # 'max' or 'mean'
        batch_size: int = 128, 
        lr: float = 1e-3, 
        lr_schedule: List[int] = [10, 2], # Example: T_0=10, T_mult=2
        weight_decay: float = 1e-5
    ):
        super().__init__()
        # This saves all hyperparameters passed to __init__ (like in_channels, lr, etc.)
        # Makes them accessible via self.hparams and ensures they are saved in checkpoints
        self.save_hyperparameters()

        # --- Model Components ---
        self.tokenizer = PointCloudTokenizer(
            feature_dim=self.hparams.in_channels,
            max_tokens=self.hparams.num_patches,
            token_dim=self.hparams.token_dim,
            k_neighbors=self.hparams.k_neighbors,
            pool_method=self.hparams.pool_method)
        
        self.encoder = PointTransformerEncoder(
            token_dim=self.hparams.token_dim,
            num_layers=self.hparams.num_layers,
            num_heads=self.hparams.num_heads,
            hidden_dim=self.hparams.hidden_dim,
            dropout=self.hparams.dropout
        )
        
        # Determine MLP head output dimension based on task and loss
        self._set_output_dim()
        self._validate_loss_fn() # Ensure valid task/loss combo
            
        self.mlp_head = self._build_mlp_head()

    def _set_output_dim(self):
        """Sets the self.output_dim based on task and loss function."""
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        
        if task == 'angular_reco':
            self.output_dim = 3  # 3D direction vector
        elif task == 'energy_reco':
            self.output_dim = 2 if loss_choice == 'gaussian_nll' else 1
        else:
            raise ValueError(f"Unknown downstream task: {task}")
            
    def _build_mlp_head(self) -> nn.Sequential:
         """Builds the final MLP projection head."""
         # Example: A simple two-layer MLP
         return nn.Sequential(
            nn.Linear(self.hparams.token_dim, self.hparams.token_dim // 2),
            nn.GELU(),
            nn.LayerNorm(self.hparams.token_dim // 2), # Added LayerNorm
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.token_dim // 2, self.output_dim)
        )

    def _validate_loss_fn(self):
        """Check if the chosen loss_fn is valid for the downstream_task."""
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        valid_losses = {
            'angular_reco': ['angular_distance', 'vmf'],
            'energy_reco': ['log_cosh', 'gaussian_nll']
        }
        if task not in valid_losses or loss_choice not in valid_losses[task]:
            raise ValueError(
                f"Invalid combination: downstream_task='{task}', loss_fn='{loss_choice}'. "
                f"Valid losses for {task}: {valid_losses[task]}"
            )

    def forward(self, coords: Tensor, features: Tensor) -> Tensor:
        """Performs the forward pass through tokenizer, encoder, and head."""
        tokens, centroids, masks = self.tokenizer(coords, features)
        global_features = self.encoder(tokens, centroids, masks)
        output = self.mlp_head(global_features)
        
        # Normalize output only if using angular distance loss for angular reconstruction
        if self.hparams.downstream_task == 'angular_reco' and self.hparams.loss_fn == 'angular_distance':
            output = F.normalize(output, p=2, dim=1)
                
        return output

    def _get_loss_function(self) -> callable:
        """Helper to get the appropriate loss function based on config."""
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn

        # Assuming labels are always [B, 4] -> [logE, dx, dy, dz]
        if task == 'angular_reco':
            true_dirs = lambda labels: labels[:, 1:]
            if loss_choice == 'angular_distance':
                return lambda preds, labels: AngularDistanceLoss(preds, true_dirs(labels))
            elif loss_choice == 'vmf':
                return lambda preds, labels: VonMisesFisherLoss(preds, true_dirs(labels))
            # _validate_loss_fn should prevent reaching else

        elif task == 'energy_reco':
            true_log_energy = lambda labels: labels[:, 0]
            if loss_choice == 'log_cosh':
                # Expects preds [B, 1] or [B]
                pred_energy = lambda preds: preds.squeeze(-1) if preds.dim() > 1 else preds
                return lambda preds, labels: LogCoshLoss(pred_energy(preds), true_log_energy(labels))
            elif loss_choice == 'gaussian_nll':
                # Expects preds [B, 2] -> [mu, var]
                pred_mu = lambda preds: preds[:, 0]
                pred_var = lambda preds: preds[:, 1] # softplus applied in loss func
                return lambda preds, labels: GaussianNLLLoss(pred_mu(preds), pred_var(preds), true_log_energy(labels))
            # _validate_loss_fn should prevent reaching else
            
        # Should not be reached due to validation in __init__
        raise ValueError(f"Unhandled task/loss combination: {task}/{loss_choice}") 

    # --- PyTorch Lightning Methods ---

    def step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Shared logic for train/val/test steps."""
        coords, features, labels = batch
        preds = self(coords, features)
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        return loss, preds, labels
        
    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """Performs a single training step."""
        loss, _, _ = self.step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Performs a single validation step."""
        loss, preds, labels = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Return detached tensors for aggregation
        return {"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()}
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Performs a single test step."""
        loss, preds, labels = self.step(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        # Return detached tensors for aggregation
        return {"test_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()}

    def _epoch_end(self, step_outputs: List[Dict[str, Tensor]], stage: str):
        """Shared logic for validation and test epoch end."""
        if not step_outputs: # Handle case where validation/test is skipped
            return 
            
        all_preds = torch.cat([x['preds'] for x in step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in step_outputs], dim=0)
        
        # Calculate and log overall loss for the epoch
        loss_func = self._get_loss_function()
        overall_loss = loss_func(all_preds, all_labels)
        self.log(f'{stage}_loss_epoch', overall_loss, prog_bar=(stage=='val')) # Show val loss on bar

        # Calculate and log median angular error if task is angular_reco
        if self.hparams.downstream_task == 'angular_reco':
            true_dirs = all_labels[:, 1:] # Assumes labels format [logE, dx, dy, dz]
            
            # Normalize predictions if VMF loss was used (preds are unnormalized)
            if self.hparams.loss_fn == 'vmf':
                preds_normalized = F.normalize(all_preds, p=2, dim=1)
            else: # angular_distance loss, preds are already normalized
                preds_normalized = all_preds 
                
            # Calculate angular errors in radians using the standard metric
            angular_errors_rad = AngularDistanceLoss(preds_normalized, true_dirs, reduction='none') * np.pi
            median_angular_error_rad = torch.median(angular_errors_rad)
            self.log(f'{stage}_median_angular_error_rad', median_angular_error_rad, prog_bar=(stage=='val'))
            
            # Also log in degrees for easier interpretation
            median_angular_error_deg = torch.rad2deg(median_angular_error_rad)
            self.log(f'{stage}_median_angular_error_deg', median_angular_error_deg)
            
    def on_validation_epoch_end(self):
        """Calculates and logs metrics at the end of the validation epoch."""
        self._epoch_end(self.trainer.validation_step_outputs, stage='val')

    def on_test_epoch_end(self):
        """Calculates and logs metrics at the end of the test epoch."""
        self._epoch_end(self.trainer.test_step_outputs, stage='test')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Sets up the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        # Cosine Annealing with Warm Restarts scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.hparams.lr_schedule[0], # Number of iterations for the first restart
            T_mult=self.hparams.lr_schedule[1] # Factor increase of T_i after a restart
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch", # Step scheduler every epoch
                "monitor": "val_loss_epoch" # Optional: Monitor validation loss epoch end
            }
        } 