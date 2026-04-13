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
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ML_COMMON_SUBMODULE = ROOT / "ml_common"
ML_COMMON_PACKAGE = ML_COMMON_SUBMODULE / "ml_common" / "__init__.py"
if ML_COMMON_PACKAGE.exists():
    submodule_path = str(ML_COMMON_SUBMODULE)
    if submodule_path not in sys.path:
        sys.path.append(submodule_path)

from ml_common.dataloaders import create_dataloaders
from directional_distributions import VMF, IAG, ESAG, GAG, SIPC, SESPC, GSPC, PowerSpherical, IPT, EPT, GPT
from ml_common.losses import (
    angular_distance_loss,
    gaussian_nll_loss,
    von_mises_fisher_loss,
    iag_nll_loss,
    esag_nll_loss,
    gag_nll_loss,
    sipc_nll_loss,
    sespc_nll_loss,
    gspc_nll_loss,
    ps_nll_loss,
    ipt_nll_loss,
    ept_nll_loss,
    gpt_nll_loss,
)
from ml_common.training import Trainer
from neptune import NeptuneModel, NeptuneMoEModel


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


def _get_output_dim(task: str, loss_name: str) -> int:
    """Determine output dimension from task and loss function."""
    if task == "angular_reco":
        return {
            "iag": 3, "vmf": 3, "sipc": 3, "ps": 3, "angular_distance": 4,
            "esag": 5, "sespc": 5, "gag": 9, "gspc": 9,
            "ipt": 3, "ept": 5, "gpt": 9,
        }.get(loss_name, 4)
    if task == "energy_reco":
        return 2 if loss_name == "gaussian_nll" else 1
    if task == "starting_classification":
        return 1
    if task == "morphology_classification":
        return 6
    if task == "track_cascade_classification":
        return 1
    if task == "neutrino_classification":
        return 1
    raise ValueError(f"Unsupported downstream_task '{task}'")


def _build_neptune_model(model_opts: Dict[str, Any], output_dim: int) -> NeptuneModel:
    """Build a single NeptuneModel from config (without moving to device)."""
    return NeptuneModel(
        in_channels=model_opts["in_channels"],
        num_patches=model_opts.get("num_patches", 128),
        token_dim=model_opts["token_dim"],
        num_layers=model_opts["num_layers"],
        num_heads=model_opts["num_heads"],
        hidden_dim=model_opts["hidden_dim"],
        dropout=model_opts.get("dropout", 0.1),
        drop_path_rate=model_opts.get("drop_path_rate", 0.0),
        output_dim=output_dim,
        k_neighbors=model_opts.get("k_neighbors", 8),
        tokenizer_kwargs=model_opts.get("tokenizer_kwargs"),
        pool_type=model_opts.get("pool_type", "mean"),
    )


def build_model(model_opts: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]
    model_type = model_opts.get("model_type", "neptune").lower()

    if model_type == "neptune_moe":
        expert_cfgs = model_opts.get("expert_configs", {})
        shared = {
            "in_channels": model_opts["in_channels"],
            "token_dim": model_opts["token_dim"],
            "num_layers": model_opts["num_layers"],
            "num_heads": model_opts["num_heads"],
            "hidden_dim": model_opts["hidden_dim"],
            "dropout": model_opts.get("dropout", 0.1),
            "drop_path_rate": model_opts.get("drop_path_rate", 0.0),
            "k_neighbors": model_opts.get("k_neighbors", 8),
            "pool_type": model_opts.get("pool_type", "mean"),
            "tokenizer_kwargs": model_opts.get("tokenizer_kwargs"),
            "num_patches": model_opts.get("num_patches", 64),
        }
        dir_loss = model_opts.get("loss_kwargs", {}).get("dir_loss", "iag")
        dir_dim = _get_output_dim("angular_reco", dir_loss)

        def _expert(overrides, output_dim):
            return _build_neptune_model({**shared, **overrides}, output_dim)

        model = NeptuneMoEModel(
            router=_expert(expert_cfgs.get("router", {}), 6),
            energy_experts={
                "contained": _expert(expert_cfgs.get("energy_contained", {}), 2),
                "uncontained": _expert(expert_cfgs.get("energy_uncontained", {}), 2),
            },
            direction_experts={
                "cascade": _expert(expert_cfgs.get("dir_cascade", {}), dir_dim),
                "low_track": _expert(expert_cfgs.get("dir_low_track", {}), dir_dim),
                "high_track": _expert(expert_cfgs.get("dir_high_track", {}), dir_dim),
            },
            energy_gate_threshold=model_opts.get("energy_gate_threshold", 10000.0),
            noise_threshold=model_opts.get("noise_threshold", 0.8),
        )
        return model.to(device)

    output_dim = _get_output_dim(task, loss_name)

    token_dim = model_opts["token_dim"]
    num_heads = model_opts["num_heads"]
    if token_dim % num_heads != 0 or (token_dim // num_heads) % 8 != 0:
        raise ValueError(
            f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads}) "
            "and produce a head_dim divisible by 8 for 4D RoPE."
        )

    model = _build_neptune_model(model_opts, output_dim)
    return model.to(device)


def build_loss_function(model_opts: Dict[str, Any]):
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]
    loss_kwargs = model_opts.get("loss_kwargs", {})

    if task == "angular_reco":
        if loss_name == "angular_distance":
            return lambda preds, labels: angular_distance_loss(
                preds[:, :3], F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "vmf":
            return lambda preds, labels: von_mises_fisher_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "iag":
            return lambda preds, labels: iag_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "esag":
            return lambda preds, labels: esag_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "gag":
            return lambda preds, labels: gag_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "sipc":
            return lambda preds, labels: sipc_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "sespc":
            return lambda preds, labels: sespc_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "gspc":
            return lambda preds, labels: gspc_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "ps":
            return lambda preds, labels: ps_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "ipt":
            return lambda preds, labels: ipt_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "ept":
            return lambda preds, labels: ept_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )
        if loss_name == "gpt":
            return lambda preds, labels: gpt_nll_loss(
                preds, F.normalize(labels[:, 1:4], p=2, dim=1)
            )

    if task == "energy_reco":
        if loss_name == "gaussian_nll":
            return lambda preds, labels: gaussian_nll_loss(preds[:, 0], preds[:, 1], labels[:, 0])
        if loss_name == "mse":
            return lambda preds, labels: F.mse_loss(preds[:, 0], labels[:, 0])

    if task == "starting_classification":
        if loss_name != "bce":
            raise ValueError("starting_classification currently supports only the 'bce' loss")

        def loss_fn(preds, labels):
            logits = preds.view(-1)
            targets = labels[..., -1].reshape(-1).float()
            return F.binary_cross_entropy_with_logits(logits, targets)

        return loss_fn

    if task == "morphology_classification":
        if loss_name != "cross_entropy":
            raise ValueError("morphology_classification currently supports only the 'cross_entropy' loss")

        # Class weights for imbalanced morphology classes (sqrt-scaled inverse frequency)
        # 0: cascade, 1: starting track, 2: throughgoing track, 3: stopping track, 4: uncontained, 5: bundle
        class_weights = loss_kwargs.get("class_weights")
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        def loss_fn(preds, labels):
            targets = labels[:, 4].long()
            weight = class_weights.to(preds.device) if class_weights is not None else None
            return F.cross_entropy(preds, targets, weight=weight)

        return loss_fn

    if task == "track_cascade_classification":
        if loss_name != "bce":
            raise ValueError("track_cascade_classification currently supports only the 'bce' loss")

        def loss_fn(preds, labels):
            logits = preds.view(-1)
            targets = (labels[:, 4] == 0).float()
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(10.0))

        return loss_fn

    if task == "neutrino_classification":
        if loss_name == "bce":
            def loss_fn(preds, labels):
                logits = preds.view(-1)
                targets = labels[..., -1].reshape(-1).float()
                return F.binary_cross_entropy_with_logits(logits, targets)
            return loss_fn

        if loss_name == "focal":
            gamma = loss_kwargs.get("gamma", 2.0)
            alpha = loss_kwargs.get("alpha", 0.25)

            def loss_fn(preds, labels):
                logits = preds.view(-1)
                targets = labels[..., -1].reshape(-1).float()
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                p_t = torch.exp(-bce)
                # alpha weighting: alpha for positives (signal), 1-alpha for negatives
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                focal_weight = alpha_t * (1 - p_t) ** gamma
                return (focal_weight * bce).mean()
            return loss_fn

        raise ValueError(f"neutrino_classification supports 'bce' or 'focal' loss, got '{loss_name}'")

    if task == "moe_reconstruction":
        dir_loss_name = loss_kwargs.get("dir_loss", "iag")
        dir_loss_map = {
            "vmf": von_mises_fisher_loss, "iag": iag_nll_loss, "esag": esag_nll_loss,
            "gag": gag_nll_loss, "sipc": sipc_nll_loss, "sespc": sespc_nll_loss,
            "gspc": gspc_nll_loss, "ps": ps_nll_loss,
            "ipt": ipt_nll_loss, "ept": ept_nll_loss, "gpt": gpt_nll_loss,
        }
        if dir_loss_name not in dir_loss_map:
            raise ValueError(f"Unknown dir_loss '{dir_loss_name}'")

        class MoELoss:
            def __init__(self):
                self._dir_loss_fn = dir_loss_map[dir_loss_name]
                self.w_morph = loss_kwargs.get("w_morph", 1.0)
                self.w_energy = loss_kwargs.get("w_energy", 1.0)
                self.w_dir = loss_kwargs.get("w_dir", 1.0)
                cw = loss_kwargs.get("class_weights")
                self._cw = torch.tensor(cw, dtype=torch.float32) if cw else None
                self.dir_dim = _get_output_dim("angular_reco", dir_loss_name)
                self._components = {}

            def __call__(self, preds, labels):
                morph_logits = preds[:, :6]
                energy_pred = preds[:, 6:8]
                dir_pred = preds[:, 8:8+self.dir_dim]

                cw = self._cw.to(preds.device) if self._cw is not None else None
                morph_loss = F.cross_entropy(morph_logits, labels[:, 4].long(), weight=cw)

                # Exclude noise events (uncontained=4) AND NaN preds (hard routing)
                valid = (labels[:, 4].long() != 4) & ~torch.isnan(energy_pred[:, 0])
                if valid.any():
                    energy_loss = gaussian_nll_loss(
                        energy_pred[valid, 0], energy_pred[valid, 1], labels[valid, 0])
                    dir_loss = self._dir_loss_fn(
                        dir_pred[valid], F.normalize(labels[valid, 1:4], p=2, dim=1))
                else:
                    energy_loss = torch.tensor(0.0, device=preds.device)
                    dir_loss = torch.tensor(0.0, device=preds.device)

                self._components = {
                    'morph_loss': morph_loss.item(),
                    'energy_loss': energy_loss.item(),
                    'dir_loss': dir_loss.item(),
                }
                return self.w_morph * morph_loss + self.w_energy * energy_loss + self.w_dir * dir_loss

            def current_weights(self):
                return self._components, None

        return MoELoss()

    raise ValueError(f"Unsupported task/loss combination: {task}/{loss_name}")


def _mean_direction(preds, loss_name, loss_kwargs):
    """Extract the mean direction from raw model predictions using the distribution classes."""
    if loss_name == "vmf":
        return VMF(preds).mean_direction
    if loss_name == "iag":
        return IAG(preds).mean_direction
    if loss_name == "esag":
        return ESAG(preds).mean_direction
    if loss_name == "gag":
        return GAG(preds).mean_direction
    if loss_name == "sipc":
        return SIPC(preds).mean_direction
    if loss_name == "sespc":
        return SESPC(preds).mean_direction
    if loss_name == "gspc":
        return GSPC(preds).mean_direction
    if loss_name == "ps":
        return PowerSpherical(preds).mean_direction
    if loss_name == "ipt":
        return IPT(preds).mean_direction
    if loss_name == "ept":
        return EPT(preds).mean_direction
    if loss_name == "gpt":
        return GPT(preds).mean_direction
    # angular_distance or unknown: fall back to normalizing first 3 dims
    return F.normalize(preds[:, :3], p=2, dim=1)


def build_metric_function(model_opts: Dict[str, Any]):
    task = model_opts["downstream_task"]
    loss_name = model_opts["loss_fn"]
    loss_kwargs = model_opts.get("loss_kwargs", {})

    if task == "angular_reco":
        def metric_fn(preds, labels):
            target_dirs = F.normalize(labels[:, 1:4], p=2, dim=1)
            pred_dirs = _mean_direction(preds, loss_name, loss_kwargs)
            errors = angular_distance_loss(pred_dirs, target_dirs, reduction="none")
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
            return {"accuracy": accuracy}
        return metric_fn

    if task == "morphology_classification":
        def metric_fn(preds, labels):
            targets = labels[:, 4].long()
            probs = torch.softmax(preds, dim=1)
            pred_classes = probs.argmax(dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            return {"accuracy": accuracy}
        return metric_fn

    if task == "track_cascade_classification":
        def metric_fn(preds, labels):
            logits = preds.view(-1)
            targets = (labels[:, 4] == 0).float()
            probs = torch.sigmoid(logits)
            preds_binary = (probs >= 0.5).float()
            accuracy = (preds_binary == targets).float().mean().item()
            return {"accuracy": accuracy}
        return metric_fn

    if task == "neutrino_classification":
        def metric_fn(preds, labels):
            logits = preds.view(-1)
            targets = labels[..., -1].reshape(-1)
            probs = torch.sigmoid(logits)

            is_signal = targets == 1.0
            is_bg = targets == 0.0
            n_signal = is_signal.sum().item()
            n_bg = is_bg.sum().item()

            metrics = {}

            # AUC-ROC (sort-based, no sklearn dependency)
            if n_signal > 0 and n_bg > 0:
                # Count concordant pairs via ranking
                sorted_idx = torch.argsort(probs, descending=True)
                sorted_targets = targets[sorted_idx]
                # Accumulate: for each signal event, count how many bg events are ranked below it
                bg_cumsum = (1 - sorted_targets).cumsum(0)
                auc = (sorted_targets * bg_cumsum).sum().item() / (n_signal * n_bg)
                metrics["auc_roc"] = auc

            # Signal efficiency at fixed background rejection rates
            if n_signal > 0 and n_bg > 0:
                signal_probs = probs[is_signal].float()
                bg_probs = probs[is_bg].float()
                for bg_rej in [0.90, 0.99, 0.999]:
                    # Threshold = quantile of bg distribution at (1 - bg_rej) from the top
                    threshold = torch.quantile(bg_probs, bg_rej).item()
                    sig_eff = (signal_probs > threshold).float().mean().item()
                    label = f"sig_eff_at_{bg_rej:.3f}_bg_rej".replace(".", "")
                    metrics[label] = sig_eff

            return metrics
        return metric_fn

    if task == "moe_reconstruction":
        dir_loss_name = loss_kwargs.get("dir_loss", "iag")
        dir_dim = _get_output_dim("angular_reco", dir_loss_name)

        def metric_fn(preds, labels):
            morph_logits = preds[:, :6]
            energy_pred = preds[:, 6:8]
            dir_pred = preds[:, 8:8+dir_dim]

            morph_targets = labels[:, 4].long()
            morph_acc = (morph_logits.argmax(dim=1) == morph_targets).float().mean().item()

            metrics = {"morph_accuracy": morph_acc}

            # Exclude noise labels and NaN predictions
            valid = (morph_targets != 4) & ~torch.isnan(energy_pred[:, 0])
            if valid.any():
                metrics["mean_energy_error"] = torch.abs(
                    energy_pred[valid, 0] - labels[valid, 0]).mean().item()
                target_dirs = F.normalize(labels[valid, 1:4], p=2, dim=1)
                pred_dirs = _mean_direction(dir_pred[valid], dir_loss_name, loss_kwargs)
                errors_rad = angular_distance_loss(pred_dirs, target_dirs, reduction="none") * torch.pi
                metrics["mean_angular_error_deg"] = torch.rad2deg(errors_rad.mean()).item()
                metrics["median_angular_error_deg"] = torch.rad2deg(torch.median(errors_rad)).item()

            return metrics
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
    model_opts = cfg["model_options"]

    if model_opts.get("downstream_task") == "starting_classification":
        cfg["task"] = "starting_classification"
        cfg.setdefault("data_options", {})
        cfg["data_options"]["task"] = "starting_classification"

    device = select_device(cfg)
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(cfg)

    model = build_model(model_opts, device)
    loss_fn = build_loss_function(model_opts)
    metric_fn = build_metric_function(model_opts)

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
