import os
import csv
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from contextlib import nullcontext

from loss_functions import AngularDistanceLoss, VonMisesFisherLoss, GaussianNLLLoss, FocalLoss, CrossEntropyLoss


class OverfitDataLoader:
    """Wrapper that limits dataloader to first N batches for overfitting experiments."""

    def __init__(self, dataloader: DataLoader, num_batches: int):
        self.dataloader = dataloader
        self.num_batches = num_batches
        self._batches_cache = []
        self._cache_loaded = False

    def _load_cache(self):
        """Load and cache the first num_batches for consistent overfitting."""
        if self._cache_loaded:
            return

        self._batches_cache = []
        for i, batch in enumerate(self.dataloader):
            if i >= self.num_batches:
                break
            self._batches_cache.append(batch)
        self._cache_loaded = True
        print(f"Cached {len(self._batches_cache)} batches for overfitting")

    def __iter__(self):
        self._load_cache()
        return iter(self._batches_cache)

    def __len__(self):
        return min(self.num_batches, len(self.dataloader))


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
        # Precision options
        self.precision = training_opts.get('precision', 'fp32').lower()
        self.test_precision = training_opts.get('test_precision', self.precision).lower()
        self.save_epochs = training_opts.get('save_epochs', 5)
        self.grad_clip = training_opts.get('grad_clip', 1.0)

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

        # Setup mixed precision (fp16 or bf16)
        def _precision_to_dtype(p: str) -> Optional[torch.dtype]:
            if 'bf16' in p:
                return torch.bfloat16
            if p.startswith('16'):
                return torch.float16
            return None

        self.amp_device = 'cuda' if device.type == 'cuda' else ('cpu' if device.type == 'cpu' else None)
        self.amp_dtype = _precision_to_dtype(self.precision)

        # Fallback if CUDA bf16 not supported
        if self.amp_device == 'cuda' and self.amp_dtype is torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                print('Warning: CUDA bf16 is not supported on this device. Falling back to fp16 mixed precision.')
                self.amp_dtype = torch.float16

        self.use_amp = (self.amp_device is not None) and (self.amp_dtype is not None)
        # GradScaler only for fp16 on CUDA
        if self.use_amp:
            self.scaler = torch.amp.GradScaler(
                self.amp_device,
                enabled=(self.amp_device == 'cuda' and self.amp_dtype is torch.float16)
            )
        else:
            self.scaler = None

        # Setup logging
        self.save_dir = Path(cfg['project_save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        if not use_wandb:
            self.csv_file = self.save_dir / 'metrics.csv'
            self.csv_writer = None
            self.csv_file_handle = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # Overfit settings
        self.overfit_batches = training_opts.get('overfit_batches', None)


    def get_loss_function(self):
        if self.downstream_task == 'angular_reco':
            if self.loss_fn_name == 'angular_distance':
                return lambda preds, labels: AngularDistanceLoss(preds, labels[:, 1:4])
            elif self.loss_fn_name == 'vmf':
                return lambda preds, labels: VonMisesFisherLoss(preds, labels[:, 1:4])
        elif self.downstream_task == 'energy_reco':
            if self.loss_fn_name == 'gaussian_nll':
                return lambda preds, labels: GaussianNLLLoss(preds[:, 0], preds[:, 1], labels[:, 0])
        elif self.downstream_task == 'multiclassification':
            if self.loss_fn_name == 'focal_loss':
                from loss_functions import FocalLoss
                return lambda preds, labels: FocalLoss(preds, labels[:,4].long())
            elif self.loss_fn_name == 'cross_entropy':
                return lambda preds, labels: CrossEntropyLoss(preds, labels[:,4].long())

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

        elif self.downstream_task == 'multiclassification':
            pred_classes = torch.argmax(preds, dim=1)
            true_classes = labels[:, 4].long()
            accuracy = (pred_classes == true_classes).float().mean().item()
            metrics['accuracy'] = accuracy

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
                elif self.downstream_task == 'multiclassification':
                    task_fields = ['accuracy']
                else:
                    task_fields = []

                # Include both training and validation versions
                all_fields = base_fields.copy()
                for field in task_fields:
                    all_fields.extend([field, f'val_{field}'])
                all_fields.append('val_loss')

                self.csv_file.parent.mkdir(parents=True, exist_ok=True)
                self.csv_file_handle = open(self.csv_file, 'w', newline='')
                self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=all_fields, extrasaction='ignore')
                self.csv_writer.writeheader()

            row = {'epoch': self.current_epoch, 'step': step or 0}
            row.update(metrics)
            self.csv_writer.writerow(row)
            self.csv_file_handle.flush()

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
            amp_ctx = torch.amp.autocast(self.amp_device, dtype=self.amp_dtype) if self.use_amp else nullcontext()
            with amp_ctx:
                preds = self.model(coords, features)
                extras = None
                if isinstance(preds, tuple):
                    preds, extras = preds
                loss = loss_fn(preds, labels)
                # Add feature-transform regularizer if present
                if extras and 'feat_reg' in extras:
                    lam = self.cfg.get('model_options', {}).get('feat_reg_weight', 0.001)
                    loss = loss + lam * extras['feat_reg']

            # Backward + step (scale only if enabled)
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
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
                self.log_metrics(metrics, step)

        return {'train_loss': running_loss / num_batches}

    def validate(self, val_loader: DataLoader, save_predictions: bool = False, profile: bool = False) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
        loss_fn = self.get_loss_function()

        forward_pass_times = []
        if profile and self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

        total_start_time = time.time()

        with torch.no_grad():
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc='Validation', leave=False, dynamic_ncols=True)
            for coords, features, labels in val_pbar:
                coords, features, labels = coords.to(self.device), features.to(self.device), labels.to(self.device)

                eval_precision = self.test_precision
                def _precision_to_dtype_eval(p: str) -> Optional[torch.dtype]:
                    return {'bf16': torch.bfloat16, '16': torch.float16}.get(p)

                eval_amp_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
                eval_amp_dtype = _precision_to_dtype_eval(eval_precision)
                if eval_amp_device == 'cuda' and eval_amp_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    eval_amp_dtype = torch.float16

                eval_use_amp = eval_amp_dtype is not None
                eval_ctx = torch.amp.autocast(eval_amp_device, dtype=eval_amp_dtype) if eval_use_amp else nullcontext()

                with eval_ctx:
                    forward_start_time = time.time()
                    preds = self.model(coords, features)
                    if self.device.type == 'cuda': torch.cuda.synchronize()
                    forward_end_time = time.time()
                    if profile:
                        forward_pass_times.append(forward_end_time - forward_start_time)

                extras = None
                if isinstance(preds, tuple): preds, extras = preds

                loss = loss_fn(preds, labels)
                if extras and 'feat_reg' in extras:
                    loss += self.cfg.get('model_options', {}).get('feat_reg_weight', 0.001) * extras['feat_reg']

                total_loss += loss.item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                val_pbar.set_postfix({'Val Loss': f'{loss.item():.6f}'})

        total_end_time = time.time()

        if profile:
            total_runtime = total_end_time - total_start_time
            total_forward_time = sum(forward_pass_times)
            avg_forward_time = total_forward_time / len(forward_pass_times) if forward_pass_times else 0

            print("\n--- Inference Profiling Report ---")
            print(f"Total inference runtime (incl. I/O): {total_runtime:.4f} seconds")
            print(f"Total forward pass runtime:             {total_forward_time:.4f} seconds")
            print(f"Average forward pass runtime per batch: {avg_forward_time * 1000:.4f} ms")
            if self.device.type == 'cuda':
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                print(f"Peak CUDA memory usage:                 {peak_mem_gb:.4f} GB")
            print("----------------------------------\n")

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Save predictions and labels if requested
        if save_predictions:
            predictions_file = self.save_dir / 'results.npz'

            np.savez(predictions_file, predictions=all_preds.float().numpy(), labels=all_labels.float().numpy())

            print(f"Saved predictions and labels to: {predictions_file}")

        metrics = {'val_loss': total_loss / len(val_loader)}
        task_metrics = self.compute_metrics(all_preds, all_labels)
        metrics.update({f'val_{k}': v for k, v in task_metrics.items()})

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False, is_final: bool = False):
        # Skip checkpoint saving in overfit mode
        if self.overfit_batches is not None:
            return

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
                checkpoint_path = self.checkpoint_dir / f'final-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'
                artifact_suffix = 'final-checkpoint'
            else:
                checkpoint_path = self.checkpoint_dir / f'epoch-{self.current_epoch:02d}-checkpoint-{time.strftime("%Y%m%d-%H%M%S")}.pt'
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
            best_path = self.checkpoint_dir / 'best-checkpoint.pt'
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

    def prepare_overfit_dataloaders(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[DataLoader, DataLoader]:
        """Prepare dataloaders for overfitting experiments by limiting to specified number of batches."""
        if self.overfit_batches is None:
            return train_loader, val_loader

        print(f"Setting up overfit mode with {self.overfit_batches} batches")

        # Create overfit dataloaders that use the same batches for train and val
        overfit_train_loader = OverfitDataLoader(train_loader, self.overfit_batches)
        overfit_val_loader = overfit_train_loader  # Use same batches for validation

        return overfit_train_loader, overfit_val_loader

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        # Prepare dataloaders for overfit mode if specified
        train_loader, val_loader = self.prepare_overfit_dataloaders(train_loader, val_loader)

        print(f"Starting training for {self.epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.overfit_batches:
            print(f"OVERFIT MODE: Using {self.overfit_batches} batches (train and val are identical)")
            print("OVERFIT MODE: Checkpoint saving disabled")

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
        test_metrics = self.validate(test_loader, save_predictions=True, profile=True)

        print("Test Results:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.6f}")

        # Log test results
        self.log_metrics(test_metrics)

        return test_metrics