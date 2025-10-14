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
from ml_common.losses import (
    CombinedDirectionalLoss,
    angular_distance_loss,
    gaussian_nll_loss,
    von_mises_fisher_loss,
)
from ml_common.training import Trainer
from neptune import NeptuneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Neptune training entrypoint")
    parser.add_argument("-c", "--cfg_file", required=True, help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    return parser.parse_args()


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as cfg_file:
        return yaml.safe_load(cfg_file)


def normalize_config(cfg: Dict[str, Any]) -> None:
    dataloader = cfg.get("dataloader")
    allowed_dataloaders = {"mmap", "kaggle", "i3"}
    if dataloader not in allowed_dataloaders:
        raise ValueError(f"dataloader must be one of {sorted(allowed_dataloaders)}, got {dataloader}")

    training_opts = cfg.get("training_options", {})
    precision = training_opts.get("precision", "fp32")
    allowed_precisions = {"bf16", "fp16", "fp32"}
    if precision not in allowed_precisions:
        raise ValueError(f"precision must be one of {sorted(allowed_precisions)}, got {precision}")
    training_opts["precision"] = precision

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


def build_model(model_opts: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]

    if task == "angular_reco":
        output_dim = 3
    elif task == "energy_reco":
        output_dim = 2 if loss_name == "gaussian_nll" else 1
    else:
        raise ValueError(f"Unsupported downstream_task '{task}'")

    tokenizer_type = model_opts.get("tokenizer_type", "v2")
    if tokenizer_type not in {"v1", "v2"}:
        raise ValueError(f"tokenizer_type must be 'v1' or 'v2', got {tokenizer_type}")

    use_spacetime_bias = bool(model_opts.get("use_spacetime_bias", False))
    spacetime_bias_layers = int(model_opts.get("spacetime_bias_layers", 2))
    bias_kwargs = model_opts.get("bias_kwargs")
    k_neighbors = model_opts.get("k_neighbors", 8)

    token_dim = model_opts["token_dim"]
    num_heads = model_opts["num_heads"]
    if token_dim % num_heads != 0 or (token_dim // num_heads) % 8 != 0:
        raise ValueError(
            f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads}) "
            "and produce a head_dim divisible by 8 for 4D RoPE."
        )

    model = NeptuneModel(
        in_channels=model_opts["in_channels"],
        num_patches=model_opts["num_patches"],
        token_dim=token_dim,
        num_layers=model_opts["num_layers"],
        num_heads=num_heads,
        hidden_dim=model_opts["hidden_dim"],
        dropout=model_opts["dropout"],
        output_dim=output_dim,
        k_neighbors=k_neighbors,
        tokenizer_type=tokenizer_type,
        use_spacetime_bias=use_spacetime_bias,
        spacetime_bias_layers=spacetime_bias_layers,
        bias_kwargs=bias_kwargs,
    )
    return model.to(device)


def build_loss_function(model_opts: Dict[str, Any]):
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]
    loss_kwargs = model_opts.get("loss_kwargs", {})

    if task == "angular_reco":
        if loss_name == "angular_distance":
            return lambda preds, labels: angular_distance_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "vmf":
            return lambda preds, labels: von_mises_fisher_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "combined_vmf_angular":
            mixed_loss = CombinedDirectionalLoss(**loss_kwargs)

            def loss_fn(preds, labels):
                targets = F.normalize(labels[:, 1:4], p=2, dim=1)
                return mixed_loss(preds, targets)

            loss_fn.set_epoch_progress = mixed_loss.set_epoch_progress  # type: ignore[attr-defined]
            loss_fn.current_weights = mixed_loss.current_weights  # type: ignore[attr-defined]
            return loss_fn

    if task == "energy_reco":
        if loss_name == "gaussian_nll":
            return lambda preds, labels: gaussian_nll_loss(preds[:, 0], preds[:, 1], labels[:, 0])
        if loss_name == "mse":
            return lambda preds, labels: F.mse_loss(preds[:, 0], labels[:, 0])

    raise ValueError(f"Unsupported task/loss combination: {task}/{loss_name}")


def build_metric_function(task: str):
    if task == "angular_reco":
        def metric_fn(preds, labels):
            target_dirs = F.normalize(labels[:, 1:4], p=2, dim=1)
            preds_norm = F.normalize(preds, p=2, dim=1)
            errors = angular_distance_loss(preds_norm, target_dirs, reduction="none")
            errors_rad = errors * torch.pi
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

    model_opts = cfg["model_options"]
    model = build_model(model_opts, device)
    loss_fn = build_loss_function(model_opts)
    metric_fn = build_metric_function(model_opts["downstream_task"])

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
