# Recreating the OpenPI checkout that trained/serves the DexTeleop checkpoints

Source of truth: `/n/home08/jackbjed/ManiSkill/pi/openpi_changes/` on the cluster (this directory is a copy,
refreshed whenever the patch is regenerated). Upstream commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`.

    git clone https://github.com/Physical-Intelligence/openpi.git openpi_serve
    cd openpi_serve && git checkout 15a9616a00943ada6c20a0f158e3adb39df2ccac
    git apply ../dexteleop-vla/openpi_changes/openpi.patch
    cp ../dexteleop-vla/openpi_changes/{dexteleop_policy,maniskill_policy,maniskill_fetch_policy,maniskill_fetch_mobile_policy}.py \
       src/openpi/policies/
    GIT_LFS_SKIP_SMUDGE=1 uv sync            # install (see openpi README); base weights: point OPENPI_DATA_HOME at an existing cache
    uv run python -c "from openpi.training.config import get_config; print(get_config('pi05_dexteleop_v1').data.repo_id)"

Serve a DexTeleop checkpoint (directory must contain params/ and assets/<repo_id>/norm_stats.json):

    uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_dexteleop_v1 --policy.dir=<ckpt_dir>/<step>
