import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from .asymmetric_vae import DecoderRNN, EncoderCNN
from .dynamics_extractor import DCDPDynamicsExtractor


def _checkpoint_section(checkpoint, section_name):
    config = checkpoint.get("config", {})
    section = config.get(section_name, {})
    return section if isinstance(section, dict) else {}


def _vae_config(checkpoint):
    cfg = _checkpoint_section(checkpoint, "vae")
    return {
        "action_dim": cfg.get("action_dim", 2),
        "cond_dim": cfg.get("cond_dim", 128),
        "hidden_dim": cfg.get("hidden_dim", 128),
        "latent_dim": cfg.get("latent_dim", 2),
        "action_horizon": cfg.get("action_horizon", 4),
    }


def _feature_config(checkpoint):
    cfg = _checkpoint_section(checkpoint, "feature_extractor")
    return {
        "reduction": cfg.get("reduction", 4),
        "freeze_backbone": True,
        "pretrained_backbone": False,
        "fusion_type": cfg.get("fusion_type", "transformer"),
        "fusion_transformer_config": cfg.get("fusion_transformer", None),
        "cross_attn_drop": cfg.get("cross_attn_drop", 0.5),
        "cross_proj_drop": cfg.get("cross_proj_drop", 0.5),
        "temp_attn_drop": cfg.get("temp_attn_drop", 0.5),
        "temp_proj_drop": cfg.get("temp_proj_drop", 0.5),
    }


class LatentActionEncoder(nn.Module):
    """Encode a normalized action chunk into a latent action."""

    def __init__(self, checkpoint):
        super().__init__()
        cfg = _vae_config(checkpoint)
        self.action_dim = cfg["action_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.action_horizon = cfg["action_horizon"]
        self.encoder = EncoderCNN(
            action_dim=cfg["action_dim"],
            hidden_dim=cfg["hidden_dim"],
            latent_dim=cfg["latent_dim"],
            action_horizon=cfg["action_horizon"],
        )
        vae_state_dict = checkpoint["vae_model_state_dict"]
        encoder_state_dict = {
            key.replace("encoder.", ""): value
            for key, value in vae_state_dict.items()
            if key.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state_dict)
        self.encoder.eval()

    def forward(self, actions):
        with torch.no_grad():
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions)
            device = next(self.encoder.parameters()).device
            actions = actions.to(device=device, dtype=torch.float32)
            mu, _ = self.encoder(actions)
            return mu


class LatentActionDecoder(nn.Module):
    """Decode one latent action and a dynamics window into low-level actions."""

    def __init__(self, checkpoint):
        super().__init__()
        cfg = _vae_config(checkpoint)
        self.action_dim = cfg["action_dim"]
        self.latent_dim = cfg["latent_dim"]
        self.action_horizon = cfg["action_horizon"]
        self.register_buffer("action_min", torch.as_tensor(checkpoint["action_min"], dtype=torch.float32))
        self.register_buffer("action_max", torch.as_tensor(checkpoint["action_max"], dtype=torch.float32))
        self.decoder = DecoderRNN(
            action_dim=cfg["action_dim"],
            latent_dim=cfg["latent_dim"],
            cond_dim=cfg["cond_dim"],
            hidden_dim=cfg["hidden_dim"],
        )
        vae_state = checkpoint["vae_model_state_dict"]
        decoder_state_dict = {
            key.replace("decoder.", ""): value
            for key, value in vae_state.items()
            if key.startswith("decoder.")
        }
        self.decoder.load_state_dict(decoder_state_dict)
        self.decoder.eval()

    def denormalize_actions(self, normalized_actions):
        return (normalized_actions + 1.0) * (self.action_max - self.action_min + 1e-8) / 2.0 + self.action_min

    def forward(self, latent_actions, index, dynamic_buffer):
        if index != -1:
            latent = latent_actions[:, index // self.action_horizon, :]
            dynamic_features = torch.stack(list(dynamic_buffer), dim=1)
            dynamic_features = self._pad_tail_keep_order(dynamic_features, index % self.action_horizon)
        else:
            latent = latent_actions.reshape(-1, self.latent_dim)
            dynamic_features = rearrange(dynamic_buffer, "(b n) d -> b n d", n=self.action_horizon)

        with torch.no_grad():
            normalized_actions = self.decoder(latent, dynamic_features)
            actions = self.denormalize_actions(normalized_actions)

        if index == -1:
            num_chunks = latent_actions.shape[1] if latent_actions.ndim == 3 else 1
            actions = actions.reshape(
                -1, num_chunks, self.action_horizon, self.action_dim
            ).reshape(-1, num_chunks * self.action_horizon, self.action_dim)
        return actions

    def _pad_tail_keep_order(self, x, n):
        if n == 0:
            return x[:, -1:, :].expand(-1, self.action_horizon, -1)
        kept = x[:, -n:, :]
        pad = x[:, -1:, :].expand(-1, self.action_horizon - n, -1)
        return torch.cat([kept, pad], dim=1)


class DynamicExtractor(nn.Module):
    """Extract visual dynamics from a 5-frame image buffer."""

    def __init__(self, checkpoint):
        super().__init__()
        self.model = DCDPDynamicsExtractor(**_feature_config(checkpoint))
        self.model.load_state_dict(checkpoint["feature_extractor_state_dict"], strict=True)
        self.model.eval()

    def forward(self, obs_buffer):
        data = torch.from_numpy(np.stack(obs_buffer).transpose(1, 0, 2, 3, 4)).float()
        device = next(self.model.parameters()).device
        data = data.to(device)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        data = (data - mean) / std

        with torch.no_grad():
            features, _ = self.model(data)
        return features
