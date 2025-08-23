import argparse
import os
import yaml
import torch
from pathlib import Path

from neptune import NeptuneModel
from trainer import Trainer
from dataloaders import create_prometheus_dataloaders, create_icecube_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Neptune: Vanilla PyTorch Training")
    parser.add_argument("-c", "--cfg_file", required=True, help="path to config file")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load configuration
    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
    # Set device
    accelerator = cfg['accelerator'].lower()
    if accelerator == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif accelerator == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif accelerator in ['gpu', 'mps']:  # Requested but not available
        print(f"Warning: {accelerator.upper()} requested but not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create dataloaders
    if cfg['dataloader'] == 'prometheus':
        train_loader, val_loader = create_prometheus_dataloaders(cfg)
    elif cfg['dataloader'] == 'icecube':
        train_loader, val_loader = create_icecube_dataloaders(cfg)
    else:
        raise ValueError(f"Unsupported dataloader: {cfg['dataloader']}. Supported options: 'prometheus', 'icecube'")
    
    # Initialize model
    model_options = cfg['model_options']
    
    # Determine output dimension based on task
    if model_options['downstream_task'] == 'angular_reco':
        output_dim = 3
    elif model_options['downstream_task'] == 'energy_reco':
        output_dim = 2 if model_options['loss_fn'] == 'gaussian_nll' else 1
    else:
        raise ValueError(f"Unknown task: {model_options['downstream_task']}")
    
    model = NeptuneModel(
        in_channels=model_options['in_channels'],
        num_patches=model_options['num_patches'],
        token_dim=model_options['token_dim'],
        num_layers=model_options['num_layers'],
        num_heads=model_options['num_heads'],
        hidden_dim=model_options['hidden_dim'],
        dropout=model_options['dropout'],
        output_dim=output_dim,
        k_neighbors=model_options['k_neighbors'],
        mlp_layers=model_options.get('mlp_layers', [256, 512, 768]),
        importance_hidden_dim=model_options.get('importance_hidden_dim', 256)
    ).to(device)
    
    # Setup logging
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg['project_name'],
                config=cfg,
                dir=cfg['project_save_dir']
            )
        except ImportError:
            print("WandB not available, falling back to CSV logging")
            use_wandb = False
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg,
        use_wandb=use_wandb
    )
    
    # Load checkpoint if specified
    if cfg.get('checkpoint', '') and os.path.exists(cfg['checkpoint']):
        print(f"Loading checkpoint: {cfg['checkpoint']}")
        trainer.load_checkpoint(cfg['checkpoint'], resume_training=cfg.get('resume_training', False))
    
    # Train or test
    if cfg['training']:
        trainer.fit(train_loader, val_loader)
    else:
        trainer.test(val_loader)


if __name__ == '__main__':
    main()