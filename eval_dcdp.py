import pathlib
import sys
import copy
import json
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if __name__ == "__main__":
    os.chdir(ROOT_DIR)

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.dcdp.latent_action_encoder import (
    DynamicExtractor,
    LatentActionDecoder,
    LatentActionEncoder,
)
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class EvalDiffusionUnetImageWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

    def _resolve_checkpoint_path(self, cfg, path_keys, dir_keys, default_name):
        for key in path_keys:
            path = cfg.get(key, None)
            if path is not None:
                path = pathlib.Path(path)
                if not path.is_file():
                    raise FileNotFoundError(f"{key} does not point to a file: {path}")
                return path

        for key in dir_keys:
            directory = cfg.get(key, None)
            if directory is not None:
                path = pathlib.Path(directory).joinpath(default_name)
                if not path.is_file():
                    raise FileNotFoundError(
                        f"{key} was provided, but {default_name} was not found: {path}"
                    )
                return path

        raise ValueError(
            f"Missing checkpoint path. Set one of {path_keys}, or provide one of "
            f"{dir_keys} containing {default_name}."
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        policy_ckpt_path = self._resolve_checkpoint_path(
            cfg,
            path_keys=('policy_checkpoint', 'policy_checkpoint_path', 'checkpoint'),
            dir_keys=('policy_checkpoint_dir',),
            default_name='latest.ckpt')
        vae_ckpt_path = self._resolve_checkpoint_path(
            cfg,
            path_keys=('vae_checkpoint', 'vae_checkpoint_path'),
            dir_keys=('vae_checkpoint_dir',),
            default_name='latest_checkpoint.pth')

        print(f"Policy checkpoint: {policy_ckpt_path}")
        print(f"VAE checkpoint: {vae_ckpt_path}")

        env_runner: BaseImageRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)

        device = torch.device(cfg.training.device)

        self.load_checkpoint(path=str(policy_ckpt_path))
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        policy = self.ema_model if cfg.training.use_ema and self.ema_model is not None else self.model
        policy.eval()

        vae_checkpoint = torch.load(vae_ckpt_path, map_location='cpu')
        action_latent_encoder = LatentActionEncoder(vae_checkpoint).to(device)
        action_latent_decoder = LatentActionDecoder(vae_checkpoint).to(device)
        extract_dynamic_features = DynamicExtractor(vae_checkpoint).to(device)

        runner_log = env_runner.run(
            policy,
            extract_dynamic_features,
            action_latent_encoder,
            action_latent_decoder)
        test_score = runner_log.get('test/mean_score', 0.0)
        print(f"test/mean_score: {test_score:.4f}")

        result = {
            "policy": policy_ckpt_path.stem,
            "policy_checkpoint": str(policy_ckpt_path),
            "vae": vae_ckpt_path.stem,
            "vae_checkpoint": str(vae_ckpt_path),
            "mean_score": test_score
        }

        json_name = f"score{test_score:.4f}_{vae_ckpt_path.stem}_{policy_ckpt_path.stem}.json"
        output_dir = cfg.get('eval_output_dir', self.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, json_name)


        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nSaved result: {output_file}")


@hydra.main(
    version_base=None,
    config_path=str(ROOT_DIR.joinpath("diffusion_policy", "config")),
    config_name="eval_diffusion_unet_image_workspace")
def main(cfg):
    workspace = EvalDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
