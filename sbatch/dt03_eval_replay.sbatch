#!/bin/bash
#SBATCH -N 1 -n 1 -c 8 -t 2:00:00 -J dt_replay -p seas_gpu,gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1 --mem=64000 --exclude=holygpu8a31105
#SBATCH -o /n/home08/jackbjed/dexteleop-vla/sbatch/out.%j -e /n/home08/jackbjed/dexteleop-vla/sbatch/err.%j
# Open-loop replay of a pi0.5 checkpoint on the canonical episode. Params: CONFIG, STEP, CKROOT, CANONICAL, STRIDE.
set -e; source ~/.bashrc
REPO="/n/home08/jackbjed/ManiSkill"; DT="/n/home08/jackbjed/dexteleop-vla"; NS="/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla"
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="${REPO}/pi/cache/uv"; export OPENPI_DATA_HOME="${REPO}/pi/cache/openpi"; export HF_HOME="${REPO}/pi/cache/hf"
CONFIG=${CONFIG:-pi05_dexteleop_overfit1}; EXP=${EXP:-dexteleop_overfit1_lora}; STEP=${STEP:-500}
CKPT="${CKROOT:-${NS}/checkpoints}/${CONFIG}/${EXP}/${STEP}"; [ -d "${CKPT}/params" ] || { echo "no ckpt ${CKPT}"; exit 1; }
CANONICAL=${CANONICAL:-/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/rec_20260821_052359_f54c26c9/t125.86-210.86_hz15}
OUT="${NS}/eval/${CONFIG}/${EXP}_${STEP}"
cd "${REPO}/pi/openpi"
uv run python "${DT}/scripts/eval_replay_checkpoint.py" --config ${CONFIG} --ckpt "${CKPT}" --canonical "${CANONICAL}" --out "${OUT}" --stride ${STRIDE:-5}
echo "DT03_DONE ${OUT}"
