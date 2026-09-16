#!/bin/bash
#SBATCH -N 1 -n 1 -c 8 -t 6:00:00 -J dt_phase2 -p shared --mem=48000
#SBATCH -o /n/home08/jackbjed/dexteleop-vla/sbatch/out.%j -e /n/home08/jackbjed/dexteleop-vla/sbatch/err.%j
# Phase 2 (canonical extraction of user-annotated segments) on a CPU node -- big bags exceed login-node memory.
# Params: RECORD (record_id), DATE (YYYY-MM-DD), SEGFILE (yaml list of {start_s,end_s,prompt,success}).
set -e; source ~/.bashrc
DT="/n/home08/jackbjed/dexteleop-vla"; PY=/n/holylabs/LABS/hankyang_lab/Lab/jackbjed/conda_envs/dexteleop-data/bin/python
cd "${DT}"
$PY scripts/ingest_phase2.py "${RECORD:?}" --date "${DATE:?}" --segments-file "${SEGFILE:?}"
echo "DT08_DONE"
