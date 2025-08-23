import os
import csv
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss_functions import AngularDistanceLoss, VonMisesFisherLoss, GaussianNLLLoss


class Trainer:
    def __init__(self, model: torch.nn.Module, device: torch.device, cfg: Dict[str, Any], use_wandb: bool = False):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.use_wandb = use_wandb
        
        # Training options
        training_opts = cfg['training_options']
        self.epochs = training_opts['epochs']
        self.lr = training_opts['lr']
        self.weight_decay = training_opts['weight_decay']
        self.batch_size = training_opts['batch_size']
        self.precision = training_opts.get('precision', 'fp32')
        self.save_epochs = training_opts.get('save_epochs', 5)
        
        # Model options
        model_opts = cfg['model_options']
        self.downstream_task = model_opts['downstream_task']
        self.loss_fn_name = model_opts['loss_fn']
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        lr_schedule = training_opts.get('lr_schedule', [10, 2])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=lr_schedule[1] if len(lr_schedule) > 1 else self.epochs,
            eta_min=1e-7
        )
        
        # Setup mixed precision
        self.use_amp = self.precision in ['16-mixed', 'bf16-mixed'] and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Setup logging
        self.save_dir = Path(cfg['project_save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        if not use_wandb:
            self.csv_file = self.save_dir / 'metrics.csv'
            self.csv_writer = None
            
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        
    def get_loss_function(self):
        if self.downstream_task == 'angular_reco':
            if self.loss_fn_name == 'angular_distance':
                return lambda preds, labels: AngularDistanceLoss(preds, labels[:, 1:4])
            elif self.loss_fn_name == 'vmf':
                return lambda preds, labels: VonMisesFisherLoss(preds, labels[:, 1:4])
        elif self.downstream_task == 'energy_reco':
            if self.loss_fn_name == 'gaussian_nll':
                return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], labels[:, 0])
        
        raise ValueError(f"Unknown task/loss combination: {self.downstream_task}/{self.loss_fn_name}")
    
    def compute_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        
        if self.downstream_task == 'angular_reco':
            true_dirs = labels[:, 1:4]
            preds_norm = F.normalize(preds, p=2, dim=1)
            angular_errors_rad = AngularDistanceLoss(preds_norm, true_dirs, reduction='none') * np.pi
            metrics['median_angular_error_deg'] = torch.rad2deg(torch.median(angular_errors_rad)).item()
            metrics['mean_angular_error_deg'] = torch.rad2deg(angular_errors_rad.mean()).item()
            
        elif self.downstream_task == 'energy_reco':
            energy_errors = torch.abs(preds[:, 0] - labels[:, 0])
            metrics['mean_energy_error'] = energy_errors.mean().item()
            
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except ImportError:
                pass
        else:
            if self.csv_writer is None:
                # Initialize with comprehensive field names for both training and validation metrics
                base_fields = ['epoch', 'step', 'loss']
                if self.downstream_task == 'angular_reco':
                    task_fields = ['mean_angular_error_deg', 'median_angular_error_deg']
                elif self.downstream_task == 'energy_reco':
                    task_fields = ['mean_energy_error']
                else:
                    task_fields = []
                
                # Include both training and validation versions
                all_fields = base_fields.copy()
                for field in task_fields:
                    all_fields.extend([field, f'val_{field}'])
                all_fields.append('val_loss')
                
                self.csv_file.parent.mkdir(parents=True, exist_ok=True)
                csv_file = open(self.csv_file, 'w', newline='')
                self.csv_writer = csv.DictWriter(csv_file, fieldnames=all_fields, extrasaction='ignore')
                self.csv_writer.writeheader()
            
            row = {'epoch': self.current_epoch, 'step': step or 0}
            row.update(metrics)
            self.csv_writer.writerow(row)
            self.csv_file.parent.joinpath(self.csv_file.name).open('a').flush()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)
        loss_fn = self.get_loss_function()
        
        # Create progress bar for training batches
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1}/{self.epochs}', 
                   leave=False, dynamic_ncols=True)
        
        for batch_idx, (coords, features, labels) in enumerate(pbar):
            coords = coords.to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    preds = self.model(coords, features)
                    loss = loss_fn(preds, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(coords, features)
                loss = loss_fn(preds, labels)
                
                loss.backward()
                self.optimizer.step()
            
                
            
            running_loss += loss.item()
            
            # Update progress bar with current loss and learning rate
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.6f}'
            })
            
            # Log every 50 steps
            if batch_idx % 50 == 0:
                step = self.current_epoch * num_batches + batch_idx
                metrics = {
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                }
                
                # Add Gumbel-specific metrics
                temp = F.softplus(self.model.tokenizer.temperature) + 0.1
                metrics['temperature'] = temp.item()
                
                self.log_metrics(metrics, step)
        
        return {'train_loss': running_loss / num_batches}
    
    def validate(self, val_loader: DataLoader, save_predictions: bool = False) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        loss_fn = self.get_loss_function()
        
        with torch.no_grad():
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc='Validation', leave=False, dynamic_ncols=True)
            for coords, features, labels in val_pbar:
                coords = coords.to(self.device)
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                preds = self.model(coords, features)
                loss = loss_fn(preds, labels)
                
                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                
                # Update validation progress bar
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})
        
        # Compute overall metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Save predictions and labels if requested
        if save_predictions:
            predictions_file = self.save_dir / 'results.npz'
            
            np.savez(predictions_file, predictions=all_preds.numpy(), labels=all_labels.numpy())
            
            print(f"Saved predictions and labels to: {predictions_file}")
        
        metrics = {'val_loss': total_loss / len(val_loader)}
        task_metrics = self.compute_metrics(all_preds, all_labels)
        metrics.update({f'val_{k}': v for k, v in task_metrics.items()})
        
        return metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False, is_final: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'cfg': self.cfg
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Regular checkpoint
        if self.current_epoch % self.save_epochs == 0 or is_final:
            if is_final:
                checkpoint_path = self.checkpoint_dir / 'neptune-final.pt'
                artifact_suffix = 'final-checkpoint'
            else:
                checkpoint_path = self.checkpoint_dir / f'neptune-epoch-{self.current_epoch:02d}.pt'
                artifact_suffix = 'checkpoint'
                
            torch.save(checkpoint, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            
            if self.use_wandb:
                try:
                    import wandb
                    artifact_name = f"{wandb.run.name or wandb.run.id}-{artifact_suffix}"
                    metadata = {'epoch': self.current_epoch, **metrics}
                    if is_final:
                        metadata['final'] = True
                    artifact = wandb.Artifact(artifact_name, type='model', metadata=metadata)
                    artifact.add_file(str(checkpoint_path))
                    wandb.log_artifact(artifact)
                    print(f"Logged wandb artifact: {artifact_name}")
                except ImportError:
                    pass # wandb not installed
                except Exception as e:
                    print(f"Could not save wandb artifact: {e}")
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'neptune-best.pt'
            torch.save(checkpoint, best_path)
            print(f'Best model saved: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            
            if self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        
        print()
        start_time = time.time()
        
        # Create overall progress bar for epochs
        epoch_pbar = tqdm(range(self.current_epoch, self.epochs), 
                         desc='Training Progress', 
                         position=0, 
                         dynamic_ncols=True)
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.log_metrics(epoch_metrics)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(epoch_metrics, is_best)
            
            # Update epoch progress bar with metrics
            postfix = {
                'Train Loss': f'{train_metrics["train_loss"]:.6f}',
                'Val Loss': f'{val_metrics["val_loss"]:.6f}',
            }
            
            # Add task-specific metrics
            if 'val_median_angular_error_deg' in val_metrics:
                postfix['Angular Error'] = f'{val_metrics["val_median_angular_error_deg"]:.2f}°'
            elif 'val_mean_energy_error' in val_metrics:
                postfix['Energy Error'] = f'{val_metrics["val_mean_energy_error"]:.4f}'
            
            epoch_pbar.set_postfix(postfix)
        
        # Save final checkpoint as wandb artifact
        final_metrics = {'best_val_loss': self.best_val_loss}
        self.save_checkpoint(final_metrics, is_final=True)
        
        print(f"Training completed! Best validation loss: {self.best_val_loss:.6f}")
    
    def test(self, test_loader: DataLoader):
        print("Running test evaluation...")
        test_metrics = self.validate(test_loader, save_predictions=True)
        
        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.6f}")
        
        # Log test results
        self.log_metrics(test_metrics)
        
        return test_metrics