import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import List, Tuple, Dict, Any, Optional
import os
import fpsample
import time

# Local imports
from utils.utils import (
    farthest_point_sampling, 
    AngularDistanceLoss, 
    VonMisesFisherLoss, 
    LogCoshLoss, 
    GaussianNLLLoss,
    CombinedAngularVMFDistanceLoss
)

from utils.fb_losses import FB5NLLLoss, FB6NLLLoss, EfficientFB8NLLLoss, create_matrix_Gamma_torch, spherical_coordinates_to_nu_torch

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
        
        # Transformer layers
        encoder_layer = TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=pre_norm
        )
        self.layers = TransformerEncoder(encoder_layer, num_layers)
        
        # Output normalization
        self.ln = nn.LayerNorm(token_dim)
    
    def forward(self, tokens, centroids, masks=None, return_per_token_features=False):
        """Process tokens through transformer architecture."""
        # Add positional embeddings
        pos_embed_out = self.pos_embed(centroids)
        tokens = tokens + pos_embed_out

        # Apply transformer layers
        if masks is not None:
            attention_mask = ~masks
            tokens = self.layers(tokens, src_key_padding_mask=attention_mask)
        else:
            tokens = self.layers(tokens)
        
        # Apply final normalization
        normed_tokens = self.ln(tokens)

        if return_per_token_features:
            return normed_tokens
        # Global average pooling
        if masks is not None:
            valid_tokens = normed_tokens * masks.unsqueeze(-1)
            token_sum = valid_tokens.sum(dim=1)
            valid_count = masks.sum(dim=1, keepdim=True).clamp(min=1)
            global_features = token_sum / valid_count
        else:
            global_features = normed_tokens.mean(dim=1)
        return global_features

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
        weight_decay: float = 1e-5,
        training_mode: str = 'supervised',
        centroid_loss_weight: float = 1.0,
        pretrain_masking_ratio: float = 0.15,
        coral_regularization: bool = False,
        coral_regularization_weight: float = 1.0
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
        
        # Mask Embeddings for MPM Pretraining
        self.centroid_mask_embedding = nn.Parameter(torch.randn(1, 1, 3))
        
        # MPM Heads for Pretraining
        self.mpm_centroid_predictor = nn.Linear(self.hparams.token_dim, 3)
        
        # Downstream task specific setup
        if self.hparams.training_mode != 'pretrain':
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
        else:
            self.classifier = None # Not used in pretrain mode
            self.output_dim = None # Not applicable in pretrain mode

        # if self.hparams.training_mode == "finetune":
        #     # 1) freeze every parameter in the model
        #     for p in self.parameters():
        #         p.requires_grad = False

        #     # 2) un-freeze ONLY the final LayerNorm inside the encoder
        #     for p in self.encoder.ln.parameters():   # <-- the `ln` you defined
        #         p.requires_grad = True

        #     # 3) un-freeze the classifier head
        #     for p in self.classifier.parameters():
        #         p.requires_grad = True

        # Results storage for test metrics
        self.test_results = {}
        # Manual storage for test outputs when not in training mode
        self.test_step_outputs = []
        self.validation_step_outputs = []
        
    def _set_output_dim(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        if task == 'angular_reco': 
            if loss_choice == 'fb8':
                self.output_dim = 8
            elif loss_choice == 'fb6':
                self.output_dim = 6
            elif loss_choice == 'kent':
                self.output_dim = 5
            else:
                self.output_dim = 3
        elif task == 'energy_reco': self.output_dim = 2 if loss_choice == 'gaussian_nll' else 1
        else: raise ValueError(f"Unknown task: {task}")

    def _validate_loss_fn(self):
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        valid = {
            'angular_reco': ['angular_distance', 'vmf', 'combined_angular_vmf', 'kent', 'fb8', 'fb6'],
            'energy_reco': ['log_cosh', 'gaussian_nll']
        }
        if task not in valid or loss_choice not in valid[task]:
            raise ValueError(f"Invalid task/loss combo: {task}/{loss_choice}")

    def forward(self, coords: Tensor, features: Tensor):
        """Forward pass through the model. Handles both supervised and pretrain modes."""
        tokens, centroids, valid_token_masks = self.tokenizer(coords, features) # valid_token_masks was 'masks'

        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            # 1. Clone original features and centroids for loss calculation
            original_centroids_for_loss = centroids.clone()

            # 2. Masking logic (applied per batch item)
            batch_size, num_tokens, _ = tokens.shape
            device = tokens.device
            
            input_features = tokens.clone()
            input_centroids = centroids.clone()
            
            # Store selected masked indices for each batch item (for loss calculation)
            # This needs to be a list of tensors, as the number of masked tokens can vary per item if num_valid varies.
            # However, the design in training_step implies masked_indices is already handled as a list.
            # Let's ensure it's created correctly here.
            actual_masked_indices_list = []

            for b in range(batch_size):
                # Consider only valid tokens for masking
                current_valid_indices = torch.where(valid_token_masks[b])[0]
                num_valid_tokens_for_item = current_valid_indices.shape[0]

                if num_valid_tokens_for_item == 0:
                    actual_masked_indices_list.append(torch.empty(0, dtype=torch.long, device=device))
                    continue

                # Determine number of tokens to mask for this specific item
                num_to_mask_for_item = max(1, int(self.hparams.pretrain_masking_ratio * num_valid_tokens_for_item))
                
                # Randomly select among the valid tokens
                perm = torch.randperm(num_valid_tokens_for_item, device=device)
                selected_indices_in_valid_set = perm[:num_to_mask_for_item]
                
                # Map these back to original token indices for this batch item
                item_masked_indices = current_valid_indices[selected_indices_in_valid_set]
                actual_masked_indices_list.append(item_masked_indices)

                # Apply mask embeddings, ensuring dtype compatibility
                input_centroids[b, item_masked_indices, :3] = self.centroid_mask_embedding.squeeze(0).to(input_centroids.dtype)

            # 3. Encode masked features and centroids
            encoded_per_token_output = self.encoder(
                input_features, input_centroids, valid_token_masks, return_per_token_features=True
            )

            # 4. Predict original features and centroids from encoded output
            predicted_centroids = self.mpm_centroid_predictor(encoded_per_token_output)

            # 5. Return structure for loss computation
            return {
                "predicted_centroids": predicted_centroids,
                "original_centroids_for_loss": original_centroids_for_loss,
                "masked_indices": actual_masked_indices_list, # List of tensors with actual masked indices per batch item
                "valid_token_masks": valid_token_masks,
            }
        else:
            # Standard supervised forward
            global_features = self.encoder(tokens, centroids, valid_token_masks) # was 'masks'
            output = self.classifier(global_features)
            if self.hparams.coral_regularization:
                return output, global_features
            return output

    def _get_loss_function(self) -> callable:
        task = self.hparams.downstream_task
        loss_choice = self.hparams.loss_fn
        if task == 'angular_reco':
            if loss_choice == 'fb8':
                fb8_loss = EfficientFB8NLLLoss()
                def fb8_loss_wrapper(preds, labels):
                    target = labels[:, 1:]
                    pred_theta_raw = torch.sigmoid(preds[:, 0:1].to(target.dtype)) * torch.pi  # [B, 1]
                    pred_phi_raw = torch.sigmoid(preds[:, 1:2].to(target.dtype)) * torch.pi * 2  # [B, 1]
                    pred_psi_raw = torch.sigmoid(preds[:, 2:3].to(target.dtype)) * torch.pi * 2  # [B, 1]
                    pred_kappa_raw = F.softplus(preds[:, 3:4]).to(target.dtype)   # [B, 1]
                    pred_beta_raw = F.softplus(preds[:, 4:5]).clamp(max=10.0).to(target.dtype)    # [B, 1]
                    pred_eta_raw = F.tanh(preds[:, 5:6]).to(target.dtype)         # [B, 1]
                    pred_alpha_raw = torch.sigmoid(preds[:, 6:7].to(target.dtype)) * torch.pi               # [B, 1]
                    pred_rho_raw = torch.sigmoid(preds[:, 7:8].to(target.dtype)) * torch.pi * 2               # [B, 1]
                    return fb8_loss(target, pred_theta_raw, pred_phi_raw, pred_psi_raw,
                                    pred_kappa_raw, pred_beta_raw, pred_eta_raw, pred_alpha_raw, pred_rho_raw)
                return fb8_loss_wrapper
            elif loss_choice == 'fb6':
                fb6_loss = FB6NLLLoss()
                def fb6_loss_wrapper(preds, labels):
                    target = labels[:, 1:]
                    pred_theta_raw = torch.sigmoid(preds[:, 0:1].to(target.dtype)) * torch.pi
                    pred_phi_raw = torch.sigmoid(preds[:, 1:2].to(target.dtype)) * torch.pi * 2
                    pred_psi_raw = torch.sigmoid(preds[:, 2:3].to(target.dtype)) * torch.pi * 2
                    pred_kappa_raw = F.softplus(preds[:, 3:4]).to(target.dtype)
                    pred_beta_raw = F.softplus(preds[:, 4:5]).to(target.dtype)
                    pred_eta_raw = F.tanh(preds[:, 5:6]).to(target.dtype)
                    return fb6_loss(target, pred_theta_raw, pred_phi_raw, pred_psi_raw,
                                    pred_kappa_raw, pred_beta_raw, pred_eta_raw)
                return fb6_loss_wrapper
            elif loss_choice == 'kent':
                # Kent distribution loss: expects 8-dim output [mu(3), axis(3), kappa(1), beta(1)]
                # and 3-dim target (unit vector)
                kent_loss = FB5NLLLoss()
                def kent_loss_wrapper(preds, labels):
                    target = labels[:, 1:]  # Extract direction labels [B, 3]
                    pred_theta_raw = torch.sigmoid(preds[:, 0:1].to(target.dtype)) * torch.pi  # [B, 1]
                    pred_phi_raw = torch.sigmoid(preds[:, 1:2].to(target.dtype)) * torch.pi * 2  # [B, 1]
                    pred_psi_raw = torch.sigmoid(preds[:, 2:3].to(target.dtype)) * torch.pi * 2  # [B, 1]
                    pred_kappa_raw = F.softplus(preds[:, 3:4]).to(target.dtype)   # [B, 1]
                    pred_beta_raw = F.softplus(preds[:, 4:5]).to(target.dtype)    # [B, 1]
                    return kent_loss(target, pred_theta_raw, pred_phi_raw, pred_psi_raw, pred_kappa_raw, pred_beta_raw)
                return kent_loss_wrapper
            else:
                # Other angular losses expect 3-dim output
                get_labels = lambda labels: labels[:, 1:]
                if loss_choice == 'angular_distance': return lambda preds, labels: AngularDistanceLoss(preds, get_labels(labels))
                if loss_choice == 'vmf': return lambda preds, labels: VonMisesFisherLoss(preds, get_labels(labels))
                if loss_choice == 'combined_angular_vmf': 
                    return lambda preds, labels: CombinedAngularVMFDistanceLoss(preds, get_labels(labels), angular_weight=0.5)
        elif task == 'energy_reco':
            get_labels = lambda labels: labels[:, 0]
            if loss_choice == 'log_cosh': return lambda preds, labels: LogCoshLoss(preds.squeeze(-1) if preds.dim() > 1 else preds, get_labels(labels))
            if loss_choice == 'gaussian_nll': return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], get_labels(labels))
        raise ValueError(f"Unhandled task/loss: {task}/{loss_choice}")

    def coral_loss(self, source_features, target_features):
        d = source_features.size(1)
        
        # source covariance
        xm = torch.mean(source_features, 0, keepdim=True) - source_features
        xc = torch.matmul(torch.transpose(xm, 0, 1), xm)

        # target covariance
        xmt = torch.mean(target_features, 0, keepdim=True) - target_features
        xct = torch.matmul(torch.transpose(xmt, 0, 1), xmt)
        
        # frobenius norm between source and target covariance matrices
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        loss = loss / (4 * d * d)
        
        return loss

    def step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        coords, features, labels = batch
        output = self(coords, features)
        
        # Handle CORAL case where forward returns (preds, global_features)
        if self.hparams.coral_regularization and isinstance(output, tuple):
            preds, _ = output  # Extract predictions, ignore global features
        else:
            preds = output
            
        loss_func = self._get_loss_function()
        loss = loss_func(preds, labels)
        return loss, preds, labels
        
    def training_step(self, batch: List[Any], batch_idx: int) -> Tensor:
        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            coords, features, _ = batch
            out = self(coords, features)

            # Compute reconstruction losses for pretraining
            centroid_reconstruction_loss, total_loss = self._compute_pretrain_losses(out)

            self.log('pretrain_centroid_loss', centroid_reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
            self.log('pretrain_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
            return total_loss
        else:
            if self.hparams.coral_regularization:
                mc_batch, data_batch = batch
                
                # MC forward pass
                mc_coords, mc_features, mc_labels = mc_batch
                mc_preds, mc_global_features = self(mc_coords, mc_features)
                
                # Data forward pass
                data_coords, data_features = data_batch
                _, data_global_features = self(data_coords, data_features)
                
                # Standard loss
                loss_func = self._get_loss_function()
                loss = loss_func(mc_preds, mc_labels)
                
                # CORAL loss
                coral_loss = self.coral_loss(mc_global_features, data_global_features)
                total_loss = loss + self.hparams.coral_regularization_weight * coral_loss
                
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
                self.log('train_coral_loss', coral_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
                self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
                
                return total_loss
            else:
                loss, _, _ = self.step(batch)
                self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)
                return loss
        
    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            coords, features, _ = batch
            out = self(coords, features)

            # Compute reconstruction losses for pretraining
            centroid_reconstruction_loss, total_loss = self._compute_pretrain_losses(out)

            self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
            
            # For pretrain mode, we don't have meaningful preds/labels for downstream tasks
            # Store the loss components instead
            output = {"val_loss": total_loss.detach(), "preds": torch.tensor(0.0), "labels": torch.tensor(0.0)}
            self.validation_step_outputs.append(output)
            return output
        else:
            loss, preds, labels = self.step(batch)
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
            self.validation_step_outputs.append({"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()})
            return {"val_loss": loss.detach(), "preds": preds.detach(), "labels": labels.detach()}
    
    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        coords, features, labels = batch
        start_time = time.time()
        
        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            out = self(coords, features)
            end_time = time.time()
            forward_time = end_time - start_time
            
            # Compute reconstruction losses for pretraining
            centroid_reconstruction_loss, total_loss = self._compute_pretrain_losses(out)

            self.log('test_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.hparams.batch_size)
            output = {
                "test_loss": total_loss.detach(), 
                "preds": torch.tensor(0.0), 
                "labels": torch.tensor(0.0),
                "forward_time": torch.tensor(forward_time, device=total_loss.device)
            }
            self.test_step_outputs.append(output)
            return output
        else:
            output = self(coords, features)
            end_time = time.time()
            forward_time = end_time - start_time
            
            # Handle CORAL case where forward returns (preds, global_features)
            if self.hparams.coral_regularization and isinstance(output, tuple):
                preds, _ = output  # Extract predictions, ignore global features
            else:
                preds = output
            
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
        
        # In pretrain mode, skip downstream task metrics since we don't have meaningful preds/labels
        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            # Just log the average loss from the step outputs
            all_losses = torch.stack([x[f'{stage}_loss'] for x in step_outputs])
            avg_loss = all_losses.mean()
            self.log(f'{stage}_loss_epoch', avg_loss, prog_bar=(stage=='val'))
            return
            
        all_preds = torch.cat([x['preds'] for x in step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in step_outputs], dim=0)
        loss_func = self._get_loss_function()
        overall_loss = loss_func(all_preds, all_labels)
        self.log(f'{stage}_loss_epoch', overall_loss, prog_bar=(stage=='val'))
        if self.hparams.downstream_task == 'angular_reco':
            true_dirs = all_labels[:, 1:]
            if self.hparams.loss_fn == 'fb8' or self.hparams.loss_fn == 'fb6' or self.hparams.loss_fn == 'kent':
                theta = torch.sigmoid(all_preds[:, 0]) * np.pi
                phi = torch.sigmoid(all_preds[:, 1]) * np.pi * 2
                psi = torch.sigmoid(all_preds[:, 2]) * np.pi * 2
                Gamma = create_matrix_Gamma_torch(theta, phi, psi)
                if self.hparams.loss_fn == 'fb8':
                    alpha = torch.sigmoid(all_preds[:, 6]) * np.pi
                    rho = torch.sigmoid(all_preds[:, 7]) * np.pi * 2
                    nu = spherical_coordinates_to_nu_torch(alpha, rho)
                    preds_norm = torch.bmm(Gamma, nu.unsqueeze(2)).squeeze(2)
                else:
                    preds_norm = Gamma[:,:,0]
            else:
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
        
        # Process and save test results
        if not outputs:
            print("WARNING: No test outputs were collected. Results will not be saved.")
            return
        
        # In pretrain mode, just save forward times and skip downstream task metrics
        if getattr(self.hparams, "training_mode", "supervised") == "pretrain":
            all_forward_times = torch.cat([x['forward_time'].unsqueeze(0) for x in outputs], dim=0)
            forward_times_np = all_forward_times.detach().cpu().numpy()
            
            self.test_results = {'forward_time': forward_times_np}
            
            # Save results to file
            if self.logger is not None:
                save_path = f"./results/{self.logger.name}_{self.logger.version}_pretrain_results.npy"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, self.test_results)
                print(f"Pretrain test results saved to {save_path}")
            
            self.test_step_outputs = []
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
            
            if self.hparams.loss_fn == 'fb8' or self.hparams.loss_fn == 'fb6' or self.hparams.loss_fn == 'kent':
                theta = torch.sigmoid(all_preds[:, 0]) * np.pi
                phi = torch.sigmoid(all_preds[:, 1]) * np.pi * 2
                psi = torch.sigmoid(all_preds[:, 2]) * np.pi * 2
                Gamma = create_matrix_Gamma_torch(theta, phi, psi)
                if self.hparams.loss_fn == 'fb8':
                    alpha = torch.sigmoid(all_preds[:, 6]) * np.pi
                    rho = torch.sigmoid(all_preds[:, 7]) * np.pi * 2
                    nu = spherical_coordinates_to_nu_torch(alpha, rho)
                    preds_norm = torch.bmm(Gamma, nu.unsqueeze(2)).squeeze(2)
                else:
                    preds_norm = Gamma[:,:,0]
                self.test_results['preds'] = all_preds.detach().cpu().numpy()
            else:
                # For other losses, preds are already in Cartesian coordinates
                preds_norm = F.normalize(all_preds, p=2, dim=1)

            preds_norm_np = preds_norm.detach().cpu().numpy()
            
            # Compute directions and metrics
            angle_errors_rad = AngularDistanceLoss(preds_norm, all_labels[:, 1:], reduction='none') * np.pi
            angle_errors_deg = torch.rad2deg(angle_errors_rad)
            angle_errors_deg_np = angle_errors_deg.detach().cpu().numpy()
            
            # Store results
            self.test_results['angle_diff'] = angle_errors_deg_np
            self.test_results['true_e'] = labels_np[:, 0]  # Energy component
            
            if self.hparams.loss_fn == 'vmf':
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
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.lr_schedule[1], eta_min=1e-7)
        return [optimizer], [scheduler]
    
    def _compute_pretrain_losses(self, forward_output: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        """
        Compute reconstruction losses for pretraining from forward pass output.
        
        Args:
            forward_output: Dictionary containing predicted features/centroids and targets
            
        Returns:
            Tuple of (centroid_reconstruction_loss, total_loss)
        """
        pred_cent = forward_output["predicted_centroids"]
        orig_cent = forward_output["original_centroids_for_loss"]
        masked_indices = forward_output["masked_indices"]

        # Compute loss only at masked indices and valid tokens
        centroid_losses = []
        batch_size = pred_cent.shape[0]
        
        for b in range(batch_size):
            idx = masked_indices[b]
            if idx.numel() == 0:
                continue
            cent_loss = F.smooth_l1_loss(pred_cent[b, idx], orig_cent[b, idx, :3], reduction="mean")
            centroid_losses.append(cent_loss)
            
        centroid_reconstruction_loss = torch.stack(centroid_losses).mean() if centroid_losses else torch.tensor(0.0, device=pred_cent.device)
        total_loss = self.hparams.centroid_loss_weight * centroid_reconstruction_loss
        
        return centroid_reconstruction_loss, total_loss