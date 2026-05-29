# DCDP

Official codebase for **Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy**.

DCDP adds a lightweight dynamic correction module on top of a pretrained image-based Diffusion Policy. The current release focuses on Push-T image experiments.

## Installation

```bash
cd dcdp
mamba env create -f conda_environment.yaml
mamba activate dcdp
pip install -r requirement.txt
pip install -e .
```

## Data

Download the Push-T dataset:

```bash
bash scripts/download_datasets.sh pusht
```

The default dataset path is:

```text
data/pusht/pusht_cchi_v7_replay.zarr
```

To download all datasets inherited from Diffusion Policy:

```bash
bash scripts/download_datasets.sh all
```

## Train the Base Policy

DCDP uses a pretrained image Diffusion Policy as the base policy. For Push-T image training:

```bash
python train.py --config-name=train_diffusion_unet_image_workspace task=pusht_image
```

The policy checkpoints are saved under:

```text
data/outputs/.../checkpoints/
```

For example:

```text
data/outputs/.../checkpoints/latest.ckpt
```

## Train DCDP

Train the DCDP dynamics extractor and asymmetric action VAE:

```bash
python train_dcdp.py \
  --zarr_path data/pusht/pusht_cchi_v7_replay.zarr \
  --save_dir checkpoints/dcdp \
  --batch_size 32 \
  --epochs 30 \
  --num_workers 8 \
  --pretrained_backbone \
  --freeze_backbone
```

Useful options:

```bash
--device cuda:0
--validation_split 0.1
--feature_lr 1e-4
--vae_lr 1e-4
--kld_weight 0.001
--diff_weight 0.001
--use_wandb
```

DCDP checkpoints are saved to:

```text
checkpoints/dcdp/dcdp_pusht_YYYYMMDD_HHMMSS/
  config.json
  latest_checkpoint.pth
  best_checkpoint.pth
```

## Evaluate DCDP

Evaluate a base Diffusion Policy checkpoint with a DCDP checkpoint:

```bash
python eval_dcdp.py \
  policy_checkpoint=/path/to/latest.ckpt \
  vae_checkpoint=checkpoints/dcdp/dcdp_pusht_YYYYMMDD_HHMMSS/best_checkpoint.pth \
  eval_output_dir=data/dcdp_eval \
  training.device=cuda:0
```

You can also provide checkpoint directories:

```bash
python eval_dcdp.py \
  policy_checkpoint_dir=/path/to/policy/checkpoints \
  vae_checkpoint_dir=checkpoints/dcdp/dcdp_pusht_YYYYMMDD_HHMMSS \
  training.device=cuda:0
```

Directory mode expects:

```text
/path/to/policy/checkpoints/latest.ckpt
checkpoints/dcdp/.../latest_checkpoint.pth
```

## Perturbation Evaluation

Push-T perturbation settings are defined in:

```text
diffusion_policy/config/task/pusht_image_dcdp.yaml
```

```yaml
env_runner:
  perturb_level: 0.0
  perturb_type: 0
```

`perturb_type: 0` applies a constant diagonal offset.  
`perturb_type: 1` applies a seeded random-direction offset.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wu2026closed,
  title={Closed-Loop Action Chunks with Dynamic Corrections for Training-Free Diffusion Policy},
  author={Wu, Pengyuan and Zhang, Pingrui and Wang, Zhigang and Wang, Dong and Zhao, Bin and Li, Xuelong},
  journal={arXiv preprint arXiv:2603.01953},
  year={2026}
}
```

## Acknowledgements

This repository builds on:

- [BID Diffusion](https://github.com/YuejiangLIU/bid_diffusion)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)

## License

See [LICENSE](LICENSE).
