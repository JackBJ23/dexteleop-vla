#!/bin/bash
#SBATCH -N 1 -n 1 -c 16 -t 6:00:00 -J dt_overfit1_build -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1 --mem=64000 --exclude=holygpu8a31105
#SBATCH -o /n/home08/jackbjed/dexteleop-vla/sbatch/out.%j -e /n/home08/jackbjed/dexteleop-vla/sbatch/err.%j
# DexTeleop overfit1 build: canonical episode -> LeRobot (node-local, then rsync to netscratch; LeRobot's
# save_episode rmtree fails on NFS -- same pattern as ManiSkill pi/sbatch/mp75b_v4_build_train.sbatch),
# then norm stats for pi05_dexteleop_overfit1. TRAIN=1 additionally chains pi/sbatch_fetch/02c_train_fetch_manip.sbatch.
set -e; source ~/.bashrc
REPO="/n/home08/jackbjed/ManiSkill"; DT="/n/home08/jackbjed/dexteleop-vla"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${REPO}/pi/cache/uv"; export OPENPI_DATA_HOME="${REPO}/pi/cache/openpi"; export HF_HOME="${REPO}/pi/cache/hf"
NS="/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla"
export HF_LEROBOT_HOME="${NS}/lerobot"; export HF_DATASETS_CACHE="${NS}/hf_datasets_cache"; mkdir -p "${HF_LEROBOT_HOME}" "${HF_DATASETS_CACHE}"
REPO_ID=${REPO_ID:-local/dexteleop_overfit1}; CONFIG=${CONFIG:-pi05_dexteleop_overfit1}; EXP=${EXP:-dexteleop_overfit1_lora}
MANIFEST=${MANIFEST:-${DT}/configs/lerobot/dexteleop_overfit1.yaml}
cd "${REPO}/pi/openpi"
LOCAL_LR="${TMPDIR:-/tmp}/${USER}_dexteleop_lerobot"; rm -rf "${LOCAL_LR}"; mkdir -p "${LOCAL_LR}"
echo "=== convert ${MANIFEST} -> ${REPO_ID} on node-local ${LOCAL_LR} ==="
HF_LEROBOT_HOME="${LOCAL_LR}" uv run python "${DT}/scripts/convert_canonical_to_lerobot.py" \
  --manifest "${MANIFEST}" --repo-id ${REPO_ID} --image-size 224 --image-writer-processes 0 --image-writer-threads 0
echo "=== rsync -> ${HF_LEROBOT_HOME}/${REPO_ID} ==="
mkdir -p "${HF_LEROBOT_HOME}/$(dirname ${REPO_ID})"
rsync -a --delete "${LOCAL_LR}/${REPO_ID}/" "${HF_LEROBOT_HOME}/${REPO_ID}/"; rm -rf "${LOCAL_LR}"
du -sh "${HF_LEROBOT_HOME}/${REPO_ID}"
echo "=== norm stats (${CONFIG}) ==="
uv run scripts/compute_norm_stats.py --config-name ${CONFIG}
find "${REPO}/pi/openpi/assets/${CONFIG}" -name norm_stats.json
if [ "${POST_CHECKS:-0}" = "1" ]; then
  echo "=== dataset checks + dataloader smoke (${CONFIG}) ==="
  uv run python "${DT}/scripts/dataset_checks.py" --config ${CONFIG} --manifest "${MANIFEST}" --out "${NS}/checks/${CONFIG}"
fi
if [ "${TRAIN:-0}" = "1" ]; then
  cd "${REPO}"
  TJ=$(CONFIG=${CONFIG} EXP=${EXP} HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
       CKPT_BASE_DIR="${NS}/checkpoints" WANDB_PROJECT=openpi_dexteleop sbatch --parsable pi/sbatch_fetch/02c_train_fetch_manip.sbatch)
  echo "chained train job ${TJ} (ckpts -> ${NS}/checkpoints/${CONFIG}/${EXP})"
fi
echo "DT01_BUILD_DONE"
