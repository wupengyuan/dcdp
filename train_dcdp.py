import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zarr
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.model.dcdp.asymmetric_vae import AsymmetricVAE
from diffusion_policy.model.dcdp.dynamics_extractor import DCDPDynamicsExtractor


try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def resolve_repo_path(path_like):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the DCDP dynamics extractor and asymmetric action VAE for Push-T image data."
    )

    parser.add_argument("--zarr_path", type=str, default="data/pusht/pusht_cchi_v7_replay.zarr")
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--validation_split", type=float, default=0.1)

    parser.add_argument("--reduction", type=int, default=4)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--fusion_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--fusion_num_heads", type=int, default=4)
    parser.add_argument("--fusion_num_layers", type=int, default=1)
    parser.add_argument("--cross_attn_drop", type=float, default=0.5)
    parser.add_argument("--cross_proj_drop", type=float, default=0.5)
    parser.add_argument("--temp_attn_drop", type=float, default=0.5)
    parser.add_argument("--temp_proj_drop", type=float, default=0.5)

    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--cond_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=2)

    parser.add_argument("--device", type=str, default="cuda", help="Use cuda, cuda:0, or cpu.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--feature_lr", type=float, default=1e-4)
    parser.add_argument("--vae_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="checkpoints/dcdp")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--kld_weight", type=float, default=0.001)
    parser.add_argument("--diff_weight", type=float, default=0.001)
    parser.add_argument("--method_name", type=str, default="dcdp")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="pusht-dcdp")
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()


def validate_args(args):
    if not 0 <= args.validation_split < 1:
        raise ValueError("--validation_split must be in [0, 1).")
    if args.history_len != 5:
        raise ValueError("The current DCDP feature extractor expects --history_len 5.")
    if args.action_horizon != 4:
        raise ValueError("The current DCDP VAE expects --action_horizon 4.")
    if args.reduction != 4:
        raise ValueError("The current DCDP evaluation path expects --reduction 4.")
    if args.freeze_backbone and not args.pretrained_backbone:
        raise ValueError("Use --pretrained_backbone when --freeze_backbone is enabled.")
    if args.action_dim != 2:
        raise ValueError("Push-T actions are 2D, so --action_dim must be 2.")
    if args.cond_dim != 128:
        raise ValueError("The DCDP dynamics extractor outputs 128D features, so --cond_dim must be 128.")


def make_config(args):
    zarr_path = resolve_repo_path(args.zarr_path)
    save_root = resolve_repo_path(args.save_dir)
    return {
        "method_name": args.method_name,
        "data": {
            "zarr_path": str(zarr_path),
            "history_len": args.history_len,
            "action_horizon": args.action_horizon,
            "validation_split": args.validation_split,
        },
        "feature_extractor": {
            "reduction": args.reduction,
            "freeze_backbone": args.freeze_backbone,
            "pretrained_backbone": args.pretrained_backbone,
            "fusion_type": args.fusion_type,
            "fusion_transformer": {
                "dropout": args.fusion_dropout,
                "num_heads": args.fusion_num_heads,
                "num_layers": args.fusion_num_layers,
            },
            "cross_attn_drop": args.cross_attn_drop,
            "cross_proj_drop": args.cross_proj_drop,
            "temp_attn_drop": args.temp_attn_drop,
            "temp_proj_drop": args.temp_proj_drop,
        },
        "vae": {
            "action_dim": args.action_dim,
            "cond_dim": args.cond_dim,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "action_horizon": args.action_horizon,
        },
        "training": {
            "device": args.device,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "feature_lr": args.feature_lr,
            "vae_lr": args.vae_lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "num_workers": args.num_workers,
            "save_freq": args.save_freq,
            "save_dir": str(save_root),
            "patience": args.patience,
            "random_seed": args.random_seed,
            "kld_weight": args.kld_weight,
            "diff_weight": args.diff_weight,
            "use_wandb": args.use_wandb,
            "wandb_project": args.wandb_project,
            "wandb_name": args.wandb_name,
        },
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PushTDCDPDataset(Dataset):
    """Return 4 action-aligned image histories and one normalized 4-step action chunk."""

    def __init__(self, zarr_path, history_len=5, action_horizon=4):
        self.zarr_path = str(zarr_path)
        self.history_len = history_len
        self.action_horizon = action_horizon

        self.dataset = zarr.open(self.zarr_path, mode="r")
        self.data = self.dataset["data"]
        self.meta = self.dataset["meta"]
        self.episode_ends = self.meta["episode_ends"][:]

        self._compute_action_normalization()
        self._build_valid_indices()

        logger.info("Loaded Push-T zarr: %s", self.zarr_path)
        logger.info("Images: %s, actions: %s", self.data["img"].shape, self.data["action"].shape)
        logger.info("Valid DCDP samples: %d", len(self.valid_indices))
        logger.info("Action min: %s", self.action_min.numpy())
        logger.info("Action max: %s", self.action_max.numpy())

    def _compute_action_normalization(self):
        actions = self.data["action"][:]
        self.action_min = torch.as_tensor(np.min(actions, axis=0, keepdims=True), dtype=torch.float32)
        self.action_max = torch.as_tensor(np.max(actions, axis=0, keepdims=True), dtype=torch.float32)

    def _build_valid_indices(self):
        self.valid_indices = []
        episode_start = 0
        for episode_end in self.episode_ends:
            first_valid = episode_start + self.history_len - 1
            last_valid = episode_end - self.action_horizon + 1
            if first_valid < last_valid:
                self.valid_indices.extend(range(first_valid, last_valid))
            episode_start = episode_end

        if len(self.valid_indices) == 0:
            raise RuntimeError(
                "No valid samples were found. Check history_len, action_horizon, and the zarr episodes."
            )

    def __len__(self):
        return len(self.valid_indices)

    def _normalize_action(self, action):
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1.0

    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]

        sequence_images = []
        for t in range(self.action_horizon):
            frame_idx = data_idx + t
            start_idx = frame_idx - self.history_len + 1
            end_idx = frame_idx + 1
            sequence_images.append(self.data["img"][start_idx:end_idx])

        sequence_images = np.stack(sequence_images, axis=0)
        sequence_images = torch.as_tensor(sequence_images, dtype=torch.float32).permute(0, 1, 4, 2, 3)
        sequence_images = sequence_images / 255.0
        sequence_images = (sequence_images - IMAGENET_MEAN) / IMAGENET_STD

        action_start = data_idx
        action_end = data_idx + self.action_horizon
        action_sequence = torch.as_tensor(self.data["action"][action_start:action_end], dtype=torch.float32)

        return {
            "sequence_images": sequence_images,
            "action_sequence": self._normalize_action(action_sequence),
            "data_idx": data_idx,
        }


class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = self._resolve_device(config["training"]["device"])
        logger.info("Using device: %s", self.device)

        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.best_metric = float("inf")
        self.best_epoch = -1
        self.patience_counter = 0

        self._init_models()
        self._init_dataset()
        self.action_min = self.full_dataset.action_min.cpu()
        self.action_max = self.full_dataset.action_max.cpu()
        self._init_optimizer()
        self._create_save_dir()
        self._init_wandb()

    def _resolve_device(self, requested):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    def _init_models(self):
        feature_cfg = self.config["feature_extractor"]
        vae_cfg = self.config["vae"]

        self.feature_extractor = DCDPDynamicsExtractor(
            reduction=feature_cfg["reduction"],
            freeze_backbone=feature_cfg["freeze_backbone"],
            pretrained_backbone=feature_cfg["pretrained_backbone"],
            fusion_type=feature_cfg["fusion_type"],
            fusion_transformer_config=feature_cfg["fusion_transformer"],
            cross_attn_drop=feature_cfg["cross_attn_drop"],
            cross_proj_drop=feature_cfg["cross_proj_drop"],
            temp_attn_drop=feature_cfg["temp_attn_drop"],
            temp_proj_drop=feature_cfg["temp_proj_drop"],
        ).to(self.device)

        self.vae_model = AsymmetricVAE(
            action_dim=vae_cfg["action_dim"],
            cond_dim=vae_cfg["cond_dim"],
            hidden_dim=vae_cfg["hidden_dim"],
            latent_dim=vae_cfg["latent_dim"],
            action_horizon=vae_cfg["action_horizon"],
        ).to(self.device)

        feature_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        vae_params = sum(p.numel() for p in self.vae_model.parameters() if p.requires_grad)
        logger.info("Trainable dynamics extractor parameters: %s", f"{feature_params:,}")
        logger.info("Trainable VAE parameters: %s", f"{vae_params:,}")

    def _init_dataset(self):
        data_cfg = self.config["data"]
        train_cfg = self.config["training"]
        self.full_dataset = PushTDCDPDataset(
            zarr_path=data_cfg["zarr_path"],
            history_len=data_cfg["history_len"],
            action_horizon=data_cfg["action_horizon"],
        )

        total_size = len(self.full_dataset)
        val_size = int(total_size * data_cfg["validation_split"])
        train_size = total_size - val_size
        if train_size <= 0:
            raise RuntimeError("Training split is empty. Reduce --validation_split.")

        self.has_validation = val_size > 0
        generator = torch.Generator().manual_seed(train_cfg["random_seed"])
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size], generator=generator
        )

        loader_kwargs = {
            "batch_size": train_cfg["batch_size"],
            "num_workers": train_cfg["num_workers"],
            "pin_memory": self.device.type == "cuda",
        }
        if train_cfg["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = (
            DataLoader(self.val_dataset, shuffle=False, **loader_kwargs)
            if self.has_validation
            else None
        )

        logger.info("Dataset split: train=%d, val=%d", train_size, val_size)

    def _init_optimizer(self):
        train_cfg = self.config["training"]
        self.optimizer = optim.AdamW(
            [
                {"params": self.feature_extractor.parameters(), "lr": train_cfg["feature_lr"]},
                {"params": self.vae_model.parameters(), "lr": train_cfg["vae_lr"]},
            ],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
        )
        self.model_parameters = list(self.feature_extractor.parameters()) + list(self.vae_model.parameters())

    def _create_save_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_root = Path(self.config["training"]["save_dir"])
        self.save_dir = save_root / f"{self.config['method_name']}_pusht_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info("Checkpoints will be saved to: %s", self.save_dir)

    def _init_wandb(self):
        train_cfg = self.config["training"]
        if not train_cfg["use_wandb"]:
            return
        if not WANDB_AVAILABLE:
            logger.warning("wandb is not installed. Training will continue without wandb logging.")
            return
        run_name = train_cfg["wandb_name"] or f"{self.config['method_name']}_{datetime.now():%Y%m%d_%H%M%S}"
        wandb.init(project=train_cfg["wandb_project"], name=run_name, config=self.config)

    def _log(self, metrics, step=None):
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(metrics, step=step)

    def _forward_batch(self, batch):
        sequence_images = batch["sequence_images"].to(self.device, non_blocking=True)
        action_sequence = batch["action_sequence"].to(self.device, non_blocking=True)

        _, action_horizon, _, _, _, _ = sequence_images.shape
        dynamic_features = []
        diff_features = []
        for t in range(action_horizon):
            frame_history = sequence_images[:, t]
            dynamic_feature, diff_feature = self.feature_extractor(frame_history)
            dynamic_features.append(dynamic_feature)
            diff_features.append(diff_feature)

        temporal_cond = torch.stack(dynamic_features, dim=1)
        temporal_diff_label = torch.stack(diff_features, dim=1)
        recon_actions, mu, logvar = self.vae_model(action_sequence, temporal_cond)

        recon_loss = nn.functional.mse_loss(recon_actions, action_sequence)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        diff_loss = nn.functional.kl_div(
            torch.log_softmax(temporal_cond, dim=-1),
            torch.softmax(temporal_diff_label, dim=-1),
            reduction="batchmean",
        )
        loss = (
            recon_loss
            + self.config["training"]["kld_weight"] * kld_loss
            + self.config["training"]["diff_weight"] * diff_loss
        )
        return loss, recon_loss, kld_loss, diff_loss

    def train_epoch(self, epoch):
        self.feature_extractor.train()
        self.vae_model.train()
        return self._run_epoch(self.train_loader, epoch, train=True)

    def validate_epoch(self, epoch):
        if not self.has_validation:
            return None
        self.feature_extractor.eval()
        self.vae_model.eval()
        with torch.no_grad():
            return self._run_epoch(self.val_loader, epoch, train=False)

    def _run_epoch(self, loader, epoch, train):
        mode = "train" if train else "val"
        totals = {"loss": 0.0, "recon": 0.0, "kld": 0.0, "diff": 0.0}
        progress = tqdm(loader, desc=f"{mode} epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress):
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.device.type == "cuda"):
                loss, recon_loss, kld_loss, diff_loss = self._forward_batch(batch)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.config["training"]["grad_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()

            totals["loss"] += loss.item()
            totals["recon"] += recon_loss.item()
            totals["kld"] += kld_loss.item()
            totals["diff"] += diff_loss.item()

            if train:
                global_step = epoch * len(loader) + batch_idx
                self._log(
                    {
                        "batch/train_loss": loss.item(),
                        "batch/train_recon_loss": recon_loss.item(),
                        "batch/train_kld_loss": kld_loss.item(),
                        "batch/train_diff_loss": diff_loss.item(),
                        "batch/lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            progress.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kld": f"{kld_loss.item():.4f}",
                    "diff": f"{diff_loss.item():.4f}",
                }
            )

        count = max(len(loader), 1)
        metrics = {key: value / count for key, value in totals.items()}
        self._log({f"epoch/{mode}_{key}": value for key, value in metrics.items()})
        return metrics

    def save_checkpoint(self, epoch, train_metrics, val_metrics, is_best=False):
        metric_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        checkpoint = {
            "epoch": epoch,
            "feature_extractor_state_dict": self.feature_extractor.state_dict(),
            "vae_model_state_dict": self.vae_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": train_metrics["loss"],
            "val_loss": None if val_metrics is None else val_metrics["loss"],
            "selection_loss": metric_loss,
            "config": self.config,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "best_val_loss": self.best_metric,
            "best_epoch": self.best_epoch,
        }

        latest_path = self.save_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.save_dir / "best_checkpoint.pth"
            torch.save(checkpoint, best_path)
            logger.info("Saved best checkpoint: %s", best_path)

        if (epoch + 1) % self.config["training"]["save_freq"] == 0:
            epoch_path = self.save_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, epoch_path)

        logger.info("Saved latest checkpoint: %s", latest_path)

    def train(self):
        logger.info("Starting DCDP training")
        last_train_metrics = None
        last_val_metrics = None

        for epoch in range(self.config["training"]["epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            self.scheduler.step()

            selection_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            is_best = selection_loss < self.best_metric
            if is_best:
                self.best_metric = selection_loss
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            val_text = "none" if val_metrics is None else f"{val_metrics['loss']:.4f}"
            logger.info(
                "Epoch %d/%d | train %.4f | val %s | best %.4f at epoch %d | lr %.6f",
                epoch + 1,
                self.config["training"]["epochs"],
                train_metrics["loss"],
                val_text,
                self.best_metric,
                self.best_epoch + 1,
                self.optimizer.param_groups[0]["lr"],
            )
            self._log(
                {
                    "epoch/best_metric": self.best_metric,
                    "epoch/patience_counter": self.patience_counter,
                    "epoch/is_best": is_best,
                }
            )

            if is_best or (epoch + 1) % self.config["training"]["save_freq"] == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=is_best)

            last_train_metrics = train_metrics
            last_val_metrics = val_metrics

            if self.patience_counter >= self.config["training"]["patience"]:
                logger.info("Early stopping after %d epochs without improvement.", self.patience_counter)
                break

        if last_train_metrics is not None:
            self.save_checkpoint(epoch, last_train_metrics, last_val_metrics, is_best=False)
        logger.info("Training finished. Best selection loss %.4f at epoch %d.", self.best_metric, self.best_epoch + 1)

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.finish()


def log_config(config):
    logger.info("=" * 60)
    logger.info("DCDP training config")
    logger.info("Data: %s", config["data"]["zarr_path"])
    logger.info("Save root: %s", config["training"]["save_dir"])
    logger.info("Device: %s", config["training"]["device"])
    logger.info("Batch size: %d", config["training"]["batch_size"])
    logger.info("Epochs: %d", config["training"]["epochs"])
    logger.info("Feature LR: %.2e", config["training"]["feature_lr"])
    logger.info("VAE LR: %.2e", config["training"]["vae_lr"])
    logger.info("KLD weight: %.4g", config["training"]["kld_weight"])
    logger.info("Diff weight: %.4g", config["training"]["diff_weight"])
    logger.info("=" * 60)


def main():
    args = parse_args()
    validate_args(args)
    set_seed(args.random_seed)
    config = make_config(args)
    log_config(config)

    trainer = TrainingPipeline(config)
    trainer.train()


if __name__ == "__main__":
    main()
