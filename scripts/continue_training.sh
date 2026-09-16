#!/bin/bash
# Continue a finished pi05_dexteleop_v1 run (10k steps) to 20k steps under config pi05_dexteleop_v1_20k.
#   scripts/continue_training.sh <EXP> [DEPENDENCY_JOBID]
# Copies the latest checkpoint step of <CKPT_BASE>/pi05_dexteleop_v1/<EXP>/ (plus wandb_id.txt, so the W&B curve
# continues in the same run) into <CKPT_BASE>/pi05_dexteleop_v1_20k/<EXP>/ and submits 02c with RESUME=1.
# With DEPENDENCY_JOBID the copy+submit itself is deferred (afterok) so it can be queued while the 10k run is still going.
set -e
EXP=${1:?usage: continue_training.sh <EXP> [DEPENDENCY_JOBID]}; DEP=${2:-}
REPO=/n/home08/jackbjed/ManiSkill; NS=/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla; CK=$NS/checkpoints
SRC=$CK/pi05_dexteleop_v1/$EXP; DST=$CK/pi05_dexteleop_v1_20k/$EXP
do_prepare_and_submit() {
  LATEST=$(ls "$SRC" | grep -E '^[0-9]+$' | sort -n | tail -1); [ -n "$LATEST" ] || { echo "no checkpoint in $SRC"; exit 1; }
  [ -d "$SRC/$LATEST/params" ] || { echo "$SRC/$LATEST incomplete"; exit 1; }
  mkdir -p "$DST"; echo "copying $SRC/$LATEST -> $DST/$LATEST"; rsync -a "$SRC/$LATEST/" "$DST/$LATEST/"; cp -n "$SRC/wandb_id.txt" "$DST/" 2>/dev/null || true
  cd "$REPO"
  TJ=$(CONFIG=pi05_dexteleop_v1_20k EXP=$EXP RESUME=1 HF_LEROBOT_HOME=$NS/lerobot HF_DATASETS_CACHE=$NS/hf_datasets_cache \
       CKPT_BASE_DIR=$CK sbatch --parsable pi/sbatch_fetch/02c_train_fetch_manip.sbatch)
  echo "continuation job $TJ: $EXP resumes from step $LATEST -> 20000 (ckpts in $DST)"; echo $TJ > /n/home08/jackbjed/dexteleop-vla/sbatch/.dt10_continue_jid
}
if [ -n "$DEP" ]; then
  # defer the whole prepare+submit until the 10k job has finished successfully
  sbatch --parsable --dependency=afterok:$DEP -p shared,sapphire,serial_requeue -t 0:30:00 --mem=4000 -J dt_continue \
    -o /n/home08/jackbjed/dexteleop-vla/sbatch/out.%j -e /n/home08/jackbjed/dexteleop-vla/sbatch/err.%j \
    --wrap "bash $(realpath "$0") $EXP" | sed 's/^/deferred prepare+submit job /'
else
  do_prepare_and_submit
fi
