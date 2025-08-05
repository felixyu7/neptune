#!/usr/bin/env python
import torch
import numpy as np
import lightning.pytorch as pl
import os
import yaml
import argparse
import sys

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from neptune.models import Neptune
from neptune.dataloaders.prometheus import PrometheusDataModule
from neptune.dataloaders.icecube_parquet import ICParquetDataModule

def parse_args():
    parser = argparse.ArgumentParser(description="Neptune: An Efficient Point Transformer for Ultrarelativistic Neutrino Events")
    parser.add_argument(
        "-c", "--config", 
        dest="cfg_file",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    return parser.parse_args()

def main():
    # Configure for better performance
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_start_method('spawn')

    # Parse command line arguments
    args = parse_args()

    # Load configuration
    with open(args.cfg_file, 'r') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    # Initialize dataloader
    if cfg['dataloader'] == 'prometheus':
        dm = PrometheusDataModule(cfg)
    elif cfg['dataloader'] == 'icecube_parquet':
        dm = ICParquetDataModule(cfg)
    else:
        print(f"Unknown dataloader: {cfg['dataloader']}")
        exit(1)
    dm.setup()

    # Initialize model
    if cfg['checkpoint'] != '':
        # Load from checkpoint
        print(f"Loading checkpoint: {cfg['checkpoint']}")
        model = Neptune.load_from_checkpoint(cfg['checkpoint'])
    else:
        # Initialize new model
        model = Neptune(
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
            pre_norm=cfg['model_options']['pre_norm'],
            mlp_layers=cfg['model_options']['mlp_layers'],
            batch_size=cfg['training_options']['batch_size'],
            lr=cfg['training_options']['lr'],
            lr_schedule=cfg['training_options']['lr_schedule'],
            weight_decay=cfg['training_options']['weight_decay']
        )

    # Setup trainer
    if cfg['training']:
        # Configure WandB logging
        os.environ["WANDB_DIR"] = os.path.abspath(cfg['project_save_dir'])
        wandb_logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'], log_model='all')
        wandb_logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']
        
        # Setup callbacks
        lr_monitor = LearningRateMonitor(logging_interval='step')
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(cfg['project_save_dir'], cfg['project_name'], wandb_logger.version, 'checkpoints'),
            filename='model-{epoch:02d}-{val_loss:.2f}.ckpt', 
            every_n_epochs=cfg['training_options']['save_epochs'],
            save_on_train_epoch_end=True
        )
        
        # Initialize trainer
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'], 
            devices=cfg['num_devices'],
            precision=cfg['training_options']['precision'],
            max_epochs=cfg['training_options']['epochs'], 
            log_every_n_steps=1, 
            gradient_clip_val=1.0,
            logger=wandb_logger, 
            callbacks=[checkpoint_callback, lr_monitor],
            num_sanity_val_steps=0
        )
        
        # Train the model
        if cfg.get('resume_training', False):
            print(f"Resuming training from checkpoint: {cfg['checkpoint']}")
            trainer.fit(model=model, datamodule=dm, ckpt_path=cfg['checkpoint'])
        else:
            trainer.fit(model=model, datamodule=dm)
    else:
        # Test mode
        logger = WandbLogger(project=cfg['project_name'], save_dir=cfg['project_save_dir'])
        logger.experiment.config["batch_size"] = cfg['training_options']['batch_size']
        
        trainer = pl.Trainer(
            accelerator=cfg['accelerator'], 
            precision=cfg['training_options']['test_precision'],
            profiler='simple', 
            logger=logger,
            num_sanity_val_steps=0
        )
        
        trainer.test(model=model, datamodule=dm)

if __name__ == "__main__":
    main() 