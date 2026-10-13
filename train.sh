#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --time=4:00:00

hostname
echo $CUDA_VISIBLE_DEVICES

export HF_HUB_OFFLINE=1

DATASET=/home/loever/Documents/teleop_data/ur5e_pi0_dataset_20261013_153627

uv run python -c '
import lerobot.datasets.lerobot_dataset as lds
import lerobot.datasets.dataset_metadata as dm
lds.get_safe_version = dm.get_safe_version = lambda repo_id, revision, *a, **k: revision
from lerobot.scripts.lerobot_train import main
main()
' \
--dataset.repo_id=local/ur5e_pi0_dataset \
--dataset.root="$DATASET" \
--dataset.revision=v3.0 \
--output_dir=outputs/train/ur5e_gr00t_finetune_$(date +%Y%m%d_%H%M%S) \
--policy.type=groot \
--policy.base_model_path=nvidia/GR00T-N1.7-3B \
--policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
--policy.push_to_hub=false \
--policy.device=cuda \
--batch_size=8 \
--steps=500 \
--save_freq=250