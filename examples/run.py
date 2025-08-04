import argparse
import os
import yaml
import torch
import lightning.pytorch as pl

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# Import cleaned dataloaders
from prometheus_dataloader import PrometheusDataModule

# Import our Neptune Lightning wrapper
from neptune_lightning import NeptuneLightning

def parse_args():
    parser = argparse.ArgumentParser(description="Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events")
    parser.add_argument(
        "-c", "--cfg_file", required=True, help="path to config file"
    )
    return parser.parse_args()

def main():
    # Parse args
    args = parse_args()
    
    # Load configuration
    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # Initialize dataloader
    if cfg['dataloader'] == 'prometheus':
        dm = PrometheusDataModule(cfg)
    else:
        print(f"Unknown dataloader: {cfg['dataloader']}")
        print("Only 'prometheus' dataloader is supported in this example")
        exit(1)
    dm.setup()

    # Initialize model
    if cfg['checkpoint'] != '' and not cfg['resume_training']:
        # Load from checkpoint
        print(f"Loading checkpoint: {cfg['checkpoint']}")
        model = NeptuneLightning.load_from_checkpoint(cfg['checkpoint'])
    else:
        # Initialize new model using our clean Neptune package
        model = NeptuneLightning(
            in_channels=cfg['model_options']['in_channels'],
            num_patches=cfg['model_options']['num_patches'],
            token_dim=cfg['model_options']['token_dim'],
            num_layers=cfg['model_options']['num_layers'],
            num_heads=cfg['model_options']['num_heads'],
            hidden_dim=cfg['model_options']['hidden_dim'],
            dropout=cfg['model_options']['dropout'],
            downstream_task=cfg['model_options']['downstream_task'],
            loss_fn=cfg['model_options']['loss_fn'],
            k_neighbors=cfg['model_options']['k_neighbors'],
            pool_method=cfg['model_options']['pool_method'],
            mlp_layers=cfg['model_options'].get('mlp_layers', [256, 512, 768]),
            batch_size=cfg['training_options']['batch_size'],
            lr=cfg['training_options']['lr'],
            lr_schedule=cfg['training_options']['lr_schedule'],
            weight_decay=cfg['training_options']['weight_decay']
        )

    # Setup trainer
    if cfg['training']:
        # Configure WandB logging
        logger = WandbLogger(
            project=cfg['project_name'],
            save_dir=cfg['project_save_dir']
        )
        
        # Configure callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg['project_save_dir'], 'checkpoints'),
            filename='neptune-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            every_n_epochs=cfg['training_options']['save_epochs']
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=cfg['training_options']['epochs'],
            accelerator=cfg['accelerator'],
            devices=cfg['num_devices'],
            precision=cfg['training_options']['precision'],
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
            enable_checkpointing=True,
            log_every_n_steps=50,
        )
        
        # Train model
        if cfg['resume_training'] and cfg['checkpoint'] != '':
            trainer.fit(model, dm, ckpt_path=cfg['checkpoint'])
        else:
            trainer.fit(model, dm)
    
    else:
        # Testing mode
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'],
            devices=cfg['num_devices'],
            precision=cfg['training_options'].get('test_precision', 'bf16-mixed'),
            logger=False,
            enable_checkpointing=False,
        )
        
        trainer.test(model, dm)

if __name__ == '__main__':
    main()