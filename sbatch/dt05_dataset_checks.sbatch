#!/bin/bash
#SBATCH -N 1 -n 1 -c 8 -t 1:00:00 -J dt_checks -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1 --mem=64000 --exclude=holygpu8a31105
#SBATCH -o /n/home08/jackbjed/dexteleop-vla/sbatch/out.%j -e /n/home08/jackbjed/dexteleop-vla/sbatch/err.%j
# Dataset validation + norm-stat analysis + dataloader smoke for an existing LeRobot dataset / config. Params: CONFIG, MANIFEST.
set -e; source ~/.bashrc
REPO="/n/home08/jackbjed/ManiSkill"; DT="/n/home08/jackbjed/dexteleop-vla"; NS="/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla"
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="${REPO}/pi/cache/uv"; export OPENPI_DATA_HOME="${REPO}/pi/cache/openpi"; export HF_HOME="${REPO}/pi/cache/hf"
export HF_LEROBOT_HOME="${NS}/lerobot"; export HF_DATASETS_CACHE="${NS}/hf_datasets_cache"
CONFIG=${CONFIG:-pi05_dexteleop_v1}; MANIFEST=${MANIFEST:-${DT}/configs/lerobot/dexteleop_train.yaml}
cd "${REPO}/pi/openpi"
uv run python "${DT}/scripts/dataset_checks.py" --config ${CONFIG} --manifest "${MANIFEST}" --out "${NS}/checks/${CONFIG}"
echo "DT05_DONE"
