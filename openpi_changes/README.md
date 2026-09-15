# openpi modifications (pi0.5 ManiSkill work + DexTeleop / TeleAvatar-2)

`pi/openpi/` is a clone of https://github.com/Physical-Intelligence/openpi at commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac` and is gitignored (own .git + ~9GB .venv). These files reproduce
our changes on a fresh clone:

1. Clone openpi into `pi/openpi` at that commit and install (see `pi/README.md`).
2. Apply the patch (covers `pyproject.toml`, `uv.lock`, `src/openpi/training/config.py`, `src/openpi/transforms.py`,
   `src/openpi/training/data_loader.py`, `src/openpi/policies/policy_config.py`,
   `packages/openpi-client/src/openpi_client/websocket_client_policy.py`):
       git -C pi/openpi apply ../openpi_changes/openpi.patch
3. Copy the new policy transforms in:
       cp maniskill_policy.py maniskill_fetch_policy.py maniskill_fetch_mobile_policy.py dexteleop_policy.py \
          pi/openpi/src/openpi/policies/

config.py adds: LeRobotManiSkill{,Fetch,FetchMobile}DataConfig + LeRobotDexTeleopDataConfig and the
pi05_maniskill_* / pi05_mobile_panda_* / pi05_dexteleop_* TrainConfigs.
pyproject.toml override-dependencies: rerun-sdk==0.23.4 + numpy>=2 (RHEL8/glibc-2.28 cluster fix).
transforms.py / data_loader.py / policy_config.py: OPT-IN `DataConfig.quantile_min_range` (default None = upstream
behaviour, bit-identical). Only `pi05_dexteleop_*` set it (1.0): dims whose q99-q01 is ~0 (an idle gripper) get
their quantile range widened so rare values normalize to O(1) instead of ~1e6.
websocket_client_policy.py: ping_interval=None (first inference triggers a >20 s JAX compile that would
otherwise drop the websocket on keepalive timeout).

Regenerate after any change (from pi/openpi):
    git diff -- pyproject.toml uv.lock src/openpi/training/config.py src/openpi/transforms.py \
        src/openpi/training/data_loader.py src/openpi/policies/policy_config.py \
        packages/openpi-client/src/openpi_client/websocket_client_policy.py > ../openpi_changes/openpi.patch
    cp src/openpi/policies/{maniskill_policy,maniskill_fetch_policy,maniskill_fetch_mobile_policy,dexteleop_policy}.py ../openpi_changes/
