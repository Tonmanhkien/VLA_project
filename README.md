# VLA_project (Minimal Vision–Language–Action)

A small Vision–Language–Action (VLA) codebase for Meta-World (MuJoCo) that learns **continuous actions** from **image + instruction text + proprio/state** using:
- modality encoders (image / text / state)
- a simple fusion MLP
- a **DDPM-style diffusion head** for action generation

The goal is to keep the project **simple, readable, and easy to extend** (swap encoders/fusion/heads without rewriting training code).

## Requirements

- Python 3.10 recommended
- Linux (Ubuntu recommended for MuJoCo rendering)
- GPU is optional (training is faster with CUDA)

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```
## How it work

### 1) Collect expert demonstrations
Example (task: `reach-v3`):
```bash
python scripts/collect_data.py \
  --env_name reach-v3 \
  --num_episodes 50 \
  --max_steps 150 \
  --instruction "reach the goal" \
  --output_dir ./data/demo_reach_v3.npz
```

Expected `.npz` fields (may vary by script version):
- `images`  (N, H, W, 3) uint8
- `states`  (N, state_dim) float32
- `actions` (N, action_dim) float32
- `text_ids` (+ `vocab`) **or** `texts`

### 2) Train (imitation learning)
```bash
python scripts/train.py \
  --data_path ./data/demo_reach_v3.npz \
  --save_path ./checkpoints/reach_v3.pt \
  --epochs 50 \
  --batch_size 128
```

### 3) Test / Rollout
```bash
python scripts/test.py \
  --ckpt_path ./checkpoints/reach_v3.pt \
  --env_name reach-v3 \
  --instruction "reach the goal"
```

## Future improvements (TODO)
- At the moment, it imitated the expert movement but fail to touch the obj (change perpective camera suggest)
