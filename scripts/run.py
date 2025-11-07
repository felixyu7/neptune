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
    FocalLoss,
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
    allowed_dataloaders = {"mmap", "kaggle", "i3", "magnemite"}
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
    elif task == "mag_classification":
        output_dim = model_opts['n_classes']
    elif task == "mag_length_reco":
        output_dim = 2 if loss_name == "gaussian_nll" else 1
    else:
        raise ValueError(f"Unsupported downstream_task '{task}'")

    k_neighbors = model_opts.get("k_neighbors", 8)
    tokenizer_kwargs = model_opts.get("tokenizer_kwargs")
    drop_path_rate = model_opts.get("drop_path_rate", 0.0)

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
        drop_path_rate=drop_path_rate,
        output_dim=output_dim,
        k_neighbors=k_neighbors,
        tokenizer_kwargs=tokenizer_kwargs,
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

    if task == "mag_classification":
        if loss_name == "cross_entropy":
            return lambda preds, labels: F.cross_entropy(preds, labels[:, 4].long())
        if loss_name == "nll":
            return lambda preds, labels: F.nll_loss(F.log_softmax(preds, dim=1), labels[:, 4].long())
        if loss_name == "focal_loss":
            return  lambda preds, labels: FocalLoss(preds, labels[:, 4].long(), n_classes=model_opts['n_classes'])

    if task == "mag_length_reco":
        # Get config options for background handling
        bg_length = model_opts.get("background_length", 0.0)  # Target length for backgrounds
        signal_weight = model_opts.get("signal_weight", 1.0)
        background_weight = model_opts.get("background_weight", 1.0)

        def weighted_length_loss(preds, labels):
            interaction = labels[:, 4]  # 0=signal, else=background
            true_length = labels[:, 5]

            # Set background lengths to epsilon
            is_signal = (interaction == 0)
            target_length = torch.where(
                is_signal,
                true_length,
                torch.tensor(bg_length, device=true_length.device, dtype=true_length.dtype)
            )

            # Apply weights
            weights = torch.where(
                is_signal,
                torch.tensor(signal_weight, device=true_length.device),
                torch.tensor(background_weight, device=true_length.device)
            )

            if loss_name == "gaussian_nll":
                # Weighted Gaussian NLL
                nll = gaussian_nll_loss(preds[:, 0], preds[:, 1], target_length, reduction='none')
                main_loss =  (weights * nll).mean()
            elif loss_name == "mse":
                # Weighted MSE
                mse = F.mse_loss(preds[:, 0], target_length, reduction='none')
                main_loss = (weights * mse).mean()
            bg_penalty = torch.relu(preds[~is_signal]).mean() * 0.1
            total_loss = main_loss + bg_penalty
            return total_loss

        return weighted_length_loss

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
    if task == "starting_classification":
        def metric_fn(preds, labels):
            logits = preds.view(-1)
            targets = labels[..., -1].reshape(-1)
            probs = torch.sigmoid(logits)
            preds_binary = (probs >= 0.5).float()
            accuracy = (preds_binary == targets).float().mean().item()
            return {
                "accuracy": accuracy,
                "positive_rate": preds_binary.mean().item(),
                "mean_probability": probs.mean().item(),
            }
        return metric_fn
    if task == "mag_classification":
        def metric_fn(preds, labels):
            logits = preds
            targets = labels[..., 4].reshape(-1)
            probs = torch.softmax(logits, dim=1)
            preds_classes = torch.argmax(probs, dim=1)
            accuracy = (preds_classes == targets).float().mean().item()
            return {
                "accuracy": accuracy,
                "positive_rate": (preds_classes > 0).float().mean().item(),
                "mean_probability": probs.mean().item(),
            }
        return metric_fn

    if task == "mag_length_reco":
        def metric_fn(preds, labels):
            pred_length = preds[:, 0]
            true_length = labels[:, 5]
            interaction = labels[:, 4]

            # Separate signal and background
            is_signal = (interaction == 0)
            is_background = ~is_signal

            metrics = {}

            # Overall metrics
            length_errors = torch.abs(pred_length - true_length)
            metrics["mean_length_error"] = length_errors.mean().item()

            # Signal-specific metrics
            if is_signal.any():
                signal_errors = length_errors[is_signal]
                metrics["signal_mean_error"] = signal_errors.mean().item()
                metrics["signal_median_error"] = torch.median(signal_errors).item()
                metrics["signal_relative_error"] = (signal_errors / (true_length[is_signal] + 1e-6)).mean().item()

            # Background-specific metrics (should be close to 0)
            if is_background.any():
                bg_preds = pred_length[is_background]
                metrics["background_mean_pred"] = bg_preds.mean().item()
                metrics["background_median_pred"] = torch.median(bg_preds).item()
                metrics["background_false_positive_rate"] = (bg_preds > 10.0).float().mean().item()  # % predicting >10m

            # Uncertainty metrics (if using Gaussian NLL)
            if preds.shape[1] > 1:
                uncertainties = preds[:, 1]
                metrics["mean_uncertainty"] = uncertainties.mean().item()
                if is_signal.any():
                    metrics["signal_uncertainty"] = uncertainties[is_signal].mean().item()
                if is_background.any():
                    metrics["background_uncertainty"] = uncertainties[is_background].mean().item()

            return metrics
        return metric_fn

    return None


class LimitedDataLoader:
    """Wrapper to limit number of batches from a dataloader"""
    def __init__(self, dataloader, max_batches=None):
        self.dataloader = dataloader
        self.max_batches = max_batches

    def __iter__(self):
        for i, batch in enumerate(self.dataloader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            yield batch

    def __len__(self):
        if self.max_batches is None:
            return len(self.dataloader)
        return min(self.max_batches, len(self.dataloader))


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
    model_opts = cfg["model_options"]

    if model_opts.get("downstream_task") == "starting_classification":
        cfg["task"] = "starting_classification"
        cfg.setdefault("data_options", {})
        cfg["data_options"]["task"] = "starting_classification"

    device = select_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(cfg)

    # Optional: Limit number of batches for debugging
    max_train_batches = cfg.get("training_options", {}).get("max_train_batches", None)
    max_val_batches = cfg.get("training_options", {}).get("max_val_batches", None)

    if max_train_batches is not None:
        print(f"Limiting training to {max_train_batches} batches")
        train_loader = LimitedDataLoader(train_loader, max_train_batches)
    if max_val_batches is not None:
        print(f"Limiting validation to {max_val_batches} batches")
        val_loader = LimitedDataLoader(val_loader, max_val_batches)

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
