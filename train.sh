#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH -p gpu --gres=gpu:a40:1
#SBATCH --time=4:00:00

hostname
echo $CUDA_VISIBLE_DEVICES
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1

COMBINED_DIR="/mnt/big/teleop_data/UR5_data_merged"
OUTPUT_DIR="/mnt/big/Gr00t_checkpoints/train/ur5e_gr00t_expert_only"

uv run python -c '
import lerobot.datasets.lerobot_dataset as lds
import lerobot.datasets.dataset_metadata as dm
lds.get_safe_version = dm.get_safe_version = lambda repo_id, revision, *a, **k: revision
from lerobot.scripts.lerobot_train import main
main()
' \
--dataset.repo_id=local/ur5_dataset \
--dataset.root="$COMBINED_DIR" \
--output_dir="$OUTPUT_DIR" \
--policy.type=groot \
--policy.base_model_path=nvidia/GR00T-N1.7-SimplerEnv-Bridge \
--policy.repo_id=ur5_gr00t_finetuned \
--policy.push_to_hub=false \
--policy.device=cuda \
--policy.embodiment_tag=new_embodiment \
--policy.tune_llm=false \
--policy.tune_visual=false \
--policy.tune_projector=true \
--policy.tune_diffusion_model=true \
--policy.tune_vlln=true \
--policy.use_bf16=true \
--policy.model_params_fp32=false \
--policy.warmup_ratio=0.05 \
--dataset.image_transforms.enable=true \
--wandb.enable=true \
--wandb.disable_artifact=true \
--batch_size=32 \
--prefetch_factor=2 \
--num_workers=8 \
--persistent_workers=true \
--steps=10000 \
--log_freq=50 \
--save_freq=500
