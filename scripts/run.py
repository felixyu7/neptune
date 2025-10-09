import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

ML_COMMON_ROOT = ROOT.parent / "ml-common"
if ML_COMMON_ROOT.exists():
    sys.path.append(str(ML_COMMON_ROOT))

from ml_common.dataloaders import create_dataloaders
from ml_common.losses import angular_distance_loss, gaussian_nll_loss, von_mises_fisher_loss
from ml_common.training import Trainer
from neptune import NeptuneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neptune training entrypoint")
    parser.add_argument("-c", "--cfg_file", required=True, help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as cfg_file:
        return yaml.load(cfg_file, Loader=yaml.FullLoader)


def normalize_config(cfg: Dict[str, Any]) -> None:
    dataloader = cfg.get("dataloader", "mmap").lower()
    dataloader_map = {
        "prometheus": "mmap",
        "icecube": "mmap",
        "mmap": "mmap",
        "kaggle": "kaggle",
        "i3": "i3",
    }
    cfg["dataloader"] = dataloader_map.get(dataloader, dataloader)

    training_opts = cfg.get("training_options", {})
    precision_map = {
        "bf16-mixed": "bf16",
        "fp16-mixed": "fp16",
        "float32": "fp32",
    }
    value = training_opts.get("precision")
    if isinstance(value, str):
        training_opts["precision"] = precision_map.get(value.lower(), value.lower())

    schedule = training_opts.pop("lr_schedule", None)
    if schedule is not None and "T_max" not in training_opts:
        if isinstance(schedule, (list, tuple)) and len(schedule) > 1:
            training_opts["T_max"] = schedule[1]


def select_device(cfg: Dict[str, Any]) -> torch.device:
    accelerator = cfg.get("accelerator", "cpu").lower()
    if accelerator == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if accelerator in {"gpu", "mps"}:
        print(f"{accelerator.upper()} requested but unavailable, falling back to CPU")
    return torch.device("cpu")


def setup_wandb(cfg: Dict[str, Any]) -> bool:
    if importlib.util.find_spec("wandb") is None:
        print("Weights & Biases not available, falling back to CSV logging")
        return False
    import wandb

    wandb.init(project=cfg["project_name"], config=cfg, dir=cfg["project_save_dir"])
    return True


def build_model(cfg: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    model_opts = cfg["model_options"]
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]

    if task == "angular_reco":
        output_dim = 3
    elif task == "energy_reco":
        output_dim = 2 if loss_name == "gaussian_nll" else 1
    else:
        raise ValueError(f"Unsupported downstream_task '{task}'")

    tokenizer_alias = {
        "point_cloud": "v1",
        "v1": "v1",
        "v2": "v2",
    }
    tokenizer_type = tokenizer_alias.get(model_opts.get("tokenizer_type", "v1"))
    if tokenizer_type is None:
        raise ValueError(f"Unknown tokenizer_type '{model_opts.get('tokenizer_type')}'")

    model = NeptuneModel(
        in_channels=model_opts["in_channels"],
        num_patches=model_opts["num_patches"],
        token_dim=model_opts["token_dim"],
        num_layers=model_opts["num_layers"],
        num_heads=model_opts["num_heads"],
        hidden_dim=model_opts["hidden_dim"],
        dropout=model_opts["dropout"],
        output_dim=output_dim,
        k_neighbors=model_opts["k_neighbors"],
        mlp_layers=model_opts.get("mlp_layers", [256, 512, 768]),
        tokenizer_type=tokenizer_type,
    ).to(device)

    return model, {"task": task, "loss": loss_name}


def build_loss_function(task: str, loss_name: str):
    if task == "angular_reco":
        if loss_name == "angular_distance":
            return lambda preds, labels: angular_distance_loss(preds, labels[:, 1:4])
        if loss_name == "vmf":
            return lambda preds, labels: von_mises_fisher_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
    if task == "energy_reco":
        if loss_name == "gaussian_nll":
            return lambda preds, labels: gaussian_nll_loss(
                preds[:, 0], preds[:, 1], labels[:, 0]
            )
        if loss_name == "mse":
            return lambda preds, labels: F.mse_loss(preds[:, 0], labels[:, 0])

    raise ValueError(f"Unsupported task/loss combination: {task}/{loss_name}")


def build_metric_function(task: str):
    if task == "angular_reco":
        def metric_fn(preds, labels):
            true_dirs = labels[:, 1:4]
            preds_norm = F.normalize(preds, p=2, dim=1)
            errors_rad = angular_distance_loss(preds_norm, true_dirs, reduction="none") * torch.pi
            return {
                "mean_angular_error_deg": torch.rad2deg(errors_rad.mean()).item(),
                "median_angular_error_deg": torch.rad2deg(torch.median(errors_rad)).item(),
            }
        return metric_fn

    if task == "energy_reco":
        def metric_fn(preds, labels):
            energy_errors = torch.abs(preds[:, 0] - labels[:, 0])
            return {"mean_energy_error": energy_errors.mean().item()}
        return metric_fn

    return None


def prepare_batch(
    coords_b: torch.Tensor, features_b: torch.Tensor, labels_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_ids = coords_b[:, 0].long()
    if coords_b.size(1) < 5:
        spatial = coords_b[:, 1:4]
        time_column = torch.zeros_like(spatial[:, :1])
        coords = torch.cat([spatial, time_column], dim=1)
    else:
        coords = coords_b[:, 1:5]
    return coords, features_b, batch_ids, labels_b


def main() -> None:
    args = parse_args()
    cfg = load_config(args.cfg_file)
    normalize_config(cfg)

    device = select_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(cfg)

    model, meta = build_model(cfg, device)
    loss_fn = build_loss_function(meta["task"], meta["loss"])
    metric_fn = build_metric_function(meta["task"])

    use_wandb = False if args.no_wandb else setup_wandb(cfg)

    trainer = Trainer(
        model=model,
        device=device,
        cfg=cfg,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        batch_prep_fn=prepare_batch,
        use_wandb=use_wandb,
    )

    checkpoint_path = cfg.get("checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path, resume_training=cfg.get("resume_training", False))

    if cfg.get("training", True):
        trainer.fit(train_loader, val_loader)
    else:
        trainer.test(val_loader)


if __name__ == "__main__":
    main()
