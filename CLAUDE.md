# CLAUDE.md — dexteleop-vla

DexTeleop / TeleAvatar 2 (bimanual mobile humanoid) real-robot data → π0.5 (OpenPI) fine-tuning,
with Diffusion Policy as a later baseline on the SAME canonical data.

**Everything below is verified on this cluster on 2026-09-11.** Do not add unverified claims.

---

## 0. Status

**No commits yet.** Nothing trained. The canonical MCAP→episode pipeline exists and has been run on episode 1
(see §8). OpenPI adapter + LeRobot converter exist (§1b); the overfit1 build job (convert + norm stats) was
ran 2026-09-11 (`sbatch/dt01_build_overfit1.sbatch`; dataset `${NS}/lerobot/local/dexteleop_overfit1`, 306 MB; norm stats in
`pi/openpi/assets/pi05_dexteleop_overfit1/`). **Overfit training launched 2026-09-14** via `02c_train_fetch_manip.sbatch`
(job id in `sbatch/.dt02_train_jid`, ckpts `${NS}/checkpoints/pi05_dexteleop_overfit1/dexteleop_overfit1_lora/`, W&B `openpi_dexteleop`).

Run everything data-side with the `dexteleop-data` conda env:
`PY=/n/holylabs/LABS/hankyang_lab/Lab/jackbjed/conda_envs/dexteleop-data/bin/python`
- tests: `$PY -m pytest -q tests`
- inspect an MCAP: `$PY scripts/inspect_mcap.py <mcap> --out reports/`
- extract: `$PY scripts/extract_episode.py configs/episodes/<ep>.yaml [--rate 45] [--no-frames] [--legacy-npz ..]`
Always `export PIP_CACHE_DIR=/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla/pip_cache` before pip installs.

---

## 0b. STANDARD WORKFLOW for a new recording (3 commands; `$PY` = the dexteleop-data python)

1. **Phase 1** — `$PY scripts/ingest_phase1.py <record_id>` → symlinks `data/raw/<record_id>` from the holylfs05 raw
   store if needed, runs the MCAP inspection, builds the head-only preview MP4 (`${NS}/videos/<record_id>_preview_head_10fps.mp4`,
   burned-in `t` = MCAP log_time − first message; playback time == t), extracts X/Y + button edges, writes
   `reports/<record_id>_phase1.md`. **Send the MP4 to the user and STOP.** The user answers with, per segment:
   `start_s / end_s / prompt / success`.
2. **Phase 2** — `$PY scripts/ingest_phase2.py <record_id> --date YYYY-MM-DD --seg START END "PROMPT" true [--seg ...]`
   → writes `configs/episodes/<record_id>_segNN.yaml`, extracts + validates every segment (all 3 cameras, QC, preview),
   updates `configs/episodes/registry.yaml` (the single source of truth for what exists) and the success-only
   per-recording manifest `configs/lerobot/<record_id>.yaml`, runs pytest, prints the A–G report. No conversion/training.
3. **Training set** — `$PY scripts/build_dataset.py --recordings <record_id> [<record_id> ...]` → composes
   `configs/lerobot/dexteleop_train.yaml` (+ dated copy) from the registry (success-only), submits the build job:
   LeRobot `local/dexteleop_train` (rebuilt each time) → norm stats for `pi05_dexteleop_v1` → dataset checks +
   dataloader smoke + sample PNGs in `${NS}/checks/pi05_dexteleop_v1/`. **Training is never launched automatically**:
   `CONFIG=pi05_dexteleop_v1 EXP=<exp> HF_LEROBOT_HOME=${NS}/lerobot HF_DATASETS_CACHE=${NS}/hf_datasets_cache
   CKPT_BASE_DIR=${NS}/checkpoints WANDB_PROJECT=openpi_dexteleop sbatch pi/sbatch_fetch/02c_train_fetch_manip.sbatch`
   (from `/n/home08/jackbjed/ManiSkill`). `pi05_dexteleop_v1.data.repo_id == local/dexteleop_train` — the config never
   needs editing when the recording list changes. Current training set (2026-09-14): rec_20260821_052219_d4f4fc76 only
   (6 segments, 1002 rows); `local/dexteleop_all` (8 episodes) remains on netscratch as an earlier build.

## 1. REUSE the existing working π0.5 stack — do NOT reinvent it

A working π0.5 LoRA fine-tuning pipeline already exists and must be reused:

| thing | path |
|---|---|
| host repo | `/n/home08/jackbjed/ManiSkill` |
| **handoff doc (read §2.5, §2.6, §5 first)** | `/n/home08/jackbjed/ManiSkill/PROJECT_HANDOFF.md` |
| OpenPI checkout (gitignored from ManiSkill) | `/n/home08/jackbjed/ManiSkill/pi/openpi` |
| local OpenPI mods, as a patch | `/n/home08/jackbjed/ManiSkill/pi/openpi_changes/` |
| LeRobot converter (template) | `pi/convert_maniskill_multitask_to_lerobot.py` |
| node-local convert + norm-stats + chained train | `pi/sbatch/mp75b_v4_build_train.sbatch` |
| train sbatch (the one everything chains into) | `pi/sbatch_fetch/02c_train_fetch_manip.sbatch` |
| serve-a-checkpoint pattern | `pi/sbatch/mp74_atomic_skill_eval.sbatch` |
| older full build example | `pi/sbatch/mp31_build_skills_v4.sbatch` |

### Environments — never mix them
- ManiSkill / demo / eval / plotting: `conda activate dp2`
- OpenPI convert / norm-stats / train / serve: `cd /n/home08/jackbjed/ManiSkill/pi/openpi && uv run ...`

**Do NOT create a second openpi checkout.** Ask first.

### OpenPI commit + local-patch status (verified 2026-09-11)
- upstream: `https://github.com/Physical-Intelligence/openpi.git`
- HEAD: **`15a9616a00943ada6c20a0f158e3adb39df2ccac`** — *"update output objects to support batching (#975)"*
- Working tree is dirty. Modified tracked files:
  - `src/openpi/training/config.py` (+1123 lines: data configs + all `pi05_maniskill*` / `pi05_mobile_panda*` TrainConfigs)
  - `pyproject.toml` (override-deps: `rerun-sdk==0.23.4`, `numpy>=2` — RHEL8/glibc-2.28 fix)
  - `packages/openpi-client/src/openpi_client/websocket_client_policy.py` (`ping_interval=None`; first inference triggers a >20 s JAX compile that would drop the websocket)
  - `uv.lock`
- Untracked (added by us): `src/openpi/policies/maniskill_policy.py`, `maniskill_fetch_policy.py`, `maniskill_fetch_mobile_policy.py`
- **`openpi_changes/openpi.patch` covers ONLY `pyproject.toml` + `config.py`.** It is byte-identical to
  the current diff of those two files (i.e. current *for what it covers*), but it **does NOT contain**
  the `websocket_client_policy.py` fix or `uv.lock`. `openpi_changes/` also ships copies of
  `maniskill_policy.py` and `maniskill_fetch_policy.py` (both identical to the live files) but
  **NOT `maniskill_fetch_mobile_policy.py`**. Regenerating the patch naively from the 2-file diff
  would silently drop those. Fix the coverage before relying on it as the reproduction recipe.
- Regeneration workflow: see `pi/openpi_changes/README.md`.

### How a π0.5 TrainConfig is shaped (template: `pi05_mobile_panda_skills_unified_v4`)
`Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora",
action_expert_variant="gemma_300m_lora")`, `batch_size=64`, `num_train_steps=20_000`, `ema_decay=None`,
`weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`,
`freeze_filter=<same Pi0Config>.get_freeze_filter()`.
Data goes through a `DataConfigFactory` subclass in `config.py` + a transform pair in `src/openpi/policies/`.

**Facts relevant to DexTeleop:**
- pi0/pi0.5 model `action_dim = 32`. `ModelTransformFactory` applies `PadStatesAndActions(32)`, so a
  **19-D state and 19-D action pad automatically** — no manual padding needed.
- π0.5 has **three image slots**: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`. The existing
  ManiSkill transforms only fill two and zero+mask-off the third. DexTeleop's head/left/right map
  onto all three natively — this is the main transform change needed.
- `pi05_base` weights are **already downloaded locally** (12 GB) at
  `/n/holylabs/LABS/hankyang_lab/Lab/jackbjed/copied_things/pi_openpi/openpi_data/openpi-assets/checkpoints/pi05_base`
  (reached via `OPENPI_DATA_HOME=/n/home08/jackbjed/ManiSkill/pi/cache/openpi`, a symlink). **No download needed.**

### 1b. DexTeleop pieces added to the shared openpi checkout (2026-09-11)
- `src/openpi/policies/dexteleop_policy.py` — `DexTeleopInputs` (head→`base_0_rgb`, left→`left_wrist_0_rgb`,
  right→`right_wrist_0_rgb`, all masks True, no inpainting) / `DexTeleopOutputs` (slice to 19).
- `config.py`: `LeRobotDexTeleopDataConfig` (repack keys `head_image/left_image/right_image/state/actions/prompt`)
  and `TrainConfig pi05_dexteleop_overfit1` = the `unified_v4` recipe (LoRA pi0.5 from `pi05_base`, batch 64,
  action_horizon 10, `prompt_from_task=True`) with `num_train_steps=3000, save_interval=500, keep_period=500`,
  `repo_id="local/dexteleop_overfit1"`.
- **Quantile-norm guard (2026-09-14)**: opt-in `DataConfig.quantile_min_range` (transforms.py `_effective_quantiles`,
  threaded through data_loader.py + policy_config.py). Default None = upstream math (verified bit-identical); only
  `pi05_dexteleop_overfit1` sets 1.0 because the left gripper cmd has q01==q99 (open 99.45 % of the episode) and
  its 7 closing frames normalized to 7.2e6 → now 7.2. State dim 14 (left gripper pos) still peaks at |x|=21.8
  (legit 0.10 rad quantile span vs. two full closes) — not degenerate, left as is.
- `pi/openpi_changes/openpi.patch` regenerated and now covers all 7 modified tracked files; all 4 policy files
  are copied there; README documents the regeneration command.
- Converter: `dexteleop-vla/scripts/convert_canonical_to_lerobot.py` — MUST run under openpi's `uv run` (pinned
  lerobot git 0cf8648, "0.1.0" API: `LeRobotDataset.create(...)`, `add_frame({..., "task"})`, `save_episode()`).
  Images stored at 224×224 (dtype "image"); `frame_index == canonical row`; scalar/int extra features are NOT
  accepted by this lerobot's encoder. Manifest: `configs/lerobot/dexteleop_overfit1.yaml` (prompt read from
  canonical `meta.json`; refuses null prompts / episodes with blockers).
- Build job: `sbatch/dt01_build_overfit1.sbatch` (mp75b pattern: node-local convert → rsync to
  `${NS}/lerobot` → `compute_norm_stats.py`; `TRAIN=1` chains `02c_train_fetch_manip.sbatch` with
  `CKPT_BASE_DIR=${NS}/checkpoints`, `HF_DATASETS_CACHE=${NS}/hf_datasets_cache`, `WANDB_PROJECT=openpi_dexteleop`).
  Norm stats land in `pi/openpi/assets/pi05_dexteleop_overfit1/local/dexteleop_overfit1/norm_stats.json`.

### 1c. `pi05_dexteleop_v1` — the 8-episode dataset (2026-09-14)
- LeRobot dataset **`local/dexteleop_all`** at `${NS}/lerobot/local/dexteleop_all` (603 MB): 8 episodes, **2549 frames**
  @15 Hz, 3 prompts, rows bit-identical to the canonical npz files, `frame_index == canonical row`.
  Build: `MANIFEST=…/configs/lerobot/dexteleop_all.yaml REPO_ID=local/dexteleop_all CONFIG=pi05_dexteleop_v1
  EXP=dexteleop_v1_lora [POST_CHECKS=1] sbatch sbatch/dt01_build_overfit1.sbatch` (job 46470279).
- TrainConfig `pi05_dexteleop_v1` = overfit1 recipe with `repo_id="local/dexteleop_all"`, `num_train_steps=10_000`,
  `save_interval=1000`, `keep_period=1000`, `quantile_min_range=1.0`. Patch regenerated.
- Norm stats `pi/openpi/assets/pi05_dexteleop_v1/local/dexteleop_all/norm_stats.json`: **no degenerate dims**
  (smallest q99−q01 = 0.46, both grippers span −1.6…2.0), so the quantile guard is a no-op for v1; normalized
  |x| max 4.24 (state base_motor3_vel), 3.5 (base_motor3_vel_cmd), 2.85 (left_gripper_pos) — all bounded.
- `scripts/dataset_checks.py` (+ `sbatch/dt05_dataset_checks.sbatch`): dataset-vs-canonical validation, norm-stat
  table, dataloader smoke, sample PNGs → `${NS}/checks/<CONFIG>/`. Needs the `__main__` guard (torch spawn workers).

### DO NOT copy from ManiSkill
The ManiSkill π0.5 policy uses **cue inpainting** (yellow ring = pick target, blue X = place target,
`--one-cue`). That is specific to the ManiSkill executor. **Do not add rings/dots to DexTeleop.**
Real-robot data keeps raw camera observations + a natural-language prompt. Cue inpainting, if ever
used here, is a separate explicit experiment.

---

## 2. STORAGE — verified numbers (2026-09-11)

| mount | group/user usage | free | verdict |
|---|---|---|---|
| `/n/home08/jackbjed` | 95 G / 95 G | **424 MB** | **100 % FULL. Write nothing here.** |
| `/n/holylabs` (grp `hankyang_lab`) | 3.7 TiB / 4.0 TiB (92.5 %) | ~0.3 TiB | shared, nearly full — **no large data** |
| `/n/holylfs05` (grp `hankyang_lab`) | 948.6 G / **2 T** (46 %) | ~1.05 TiB | persistent valuable data. NOTE: 2 T group cap, `jackbjed` is 919 G of it |
| `/n/netscratch` (grp `hankyang_lab`) | 43.1 TiB / 50 TiB (86 %) | ~6.9 TiB | large temp / purge-eligible |

`/n/home08/jackbjed/dexteleop-vla` → `/n/holylabs/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla`
(so the repo itself lives on the nearly-full holylabs — **code only, no data**).

Recommended locations for this project:

| artifact | path |
|---|---|
| raw MCAP (immutable) | `/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/raw/` (already there; symlinked into `data/raw/`) |
| canonical episode dataset | `/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/` |
| LeRobot dataset (`HF_LEROBOT_HOME`) | `/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop/lerobot` |
| conversion scratch | node-local `${TMPDIR}` → rsync out (see below) |
| checkpoints (`CKPT_BASE_DIR`) | `/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop/checkpoints` |
| `HF_DATASETS_CACHE` | `/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop/hf_datasets_cache` |
| `HF_HOME`, `OPENPI_DATA_HOME`, `UV_CACHE_DIR` | reuse ManiSkill's (`${REPO}/pi/cache/{hf,openpi,uv}`; `hf`+`openpi` are holylabs symlinks) |

**home08 composition (verified):** `pi/openpi/.venv` 9.3 G and `pi/cache/uv` 9.4 G are **hardlinked to each
other** (7.73 GiB shared, only 0.19 GiB cache-only) — cleaning the uv cache reclaims ~0.2 GB, not 9 GB.
`~/miniconda3` is **~52 G on home08** (before removing `da3`): envs ~~`da3` 14 G~~ (Depth-Anything-3 env, **removed 2026-09-11 with `conda env remove`**), `dp2` 11 G (**this is the one `conda activate dp2` resolves to** — `envs_dirs` puts `~/miniconda3/envs` first, the holylabs `conda_envs/dp2` copy is NOT the active one), `diffusion-policy-ms` 9.0 G (the DP env from the DP README — exists after all), `dp1` 420 M; `pkgs/` 8.7 G apparent (hardlink share with envs not fully measured — scan timed out). `~/.local` 8.9 G, `~/.vscode-server` 1.8 G, `~/.cache/huggingface` 759 M, `scaling-diffusion-policy` 915 M. No deletion done — needs approval.
New project dirs: `/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla/{pip_cache,uv_cache,hf_datasets_cache,tmp}`
and `/n/holylfs05/.../dexteleop-vla-data/{raw,canonical,lerobot,checkpoints}`.

**Established env-var pattern — copy verbatim from `mp75b_v4_build_train.sbatch`:**
```bash
export UV_CACHE_DIR="${REPO}/pi/cache/uv"; export OPENPI_DATA_HOME="${REPO}/pi/cache/openpi"; export HF_HOME="${REPO}/pi/cache/hf"
export HF_LEROBOT_HOME="${NS}/lerobot"; export HF_DATASETS_CACHE="${NS}/hf_datasets_cache"
```

**Node-local LeRobot conversion is mandatory.** LeRobot's `save_episode` does an `rmtree` of the images
dir that fails on NFS ("Directory not empty", stray `.nfs*` handles). Working pattern:
convert with `HF_LEROBOT_HOME=${TMPDIR}/...`, `--image-writer-processes 0 --image-writer-threads 0`
(synchronous), then `rsync -a --delete` the result to netscratch. Verbatim in `mp75b`.

---

## 3. Cluster

SLURM, account `hankyang_lab`. Partitions used: `-p seas_gpu,gpu`.
GPU line used everywhere: `--gres=gpu:nvidia_a100-sxm4-80gb:1`, typically `-c 16 --mem=64000`
(training uses `--mem=120000`), plus `--exclude=holygpu8a31105`.
**Rendering and GPU training must not run on the login node** (no Vulkan).
Copy directives from existing sbatch files rather than inventing them.
Claude's scratchpad dir is **not visible to compute nodes** — helper scripts must live inside the repo.

---

## 4. Source data (episode 1)

`data/raw/rec_20260821_052359_f54c26c9` → symlink to
`/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/raw/rec_20260821_052359_f54c26c9`

**Authoritative source of truth (never modify):**
- `rec_20260821_052359_f54c26c9_0.mcap` — 855 MB, MCAP0 / `ros2` / `rosbag2` / libmcap 0.8.0
- `metadata.yaml` — rosbag2 v5 info: duration **315.030 s**, **1 439 412 messages**, 56 topics

**Prior debugging artifacts — cross-check only, must NOT define the production dataset:**
`rec_..._training.npz`, `rec_..._metadata.json`, `rec_..._0.txt`, `diagnostics/`, `videos/`,
`training_ready/`. These were produced **off-cluster on a Mac** (`/Users/jack/Downloads/...`);
**the code that made them is not in this repo and is not on this cluster.**
The MP4s are remuxed visualization copies — never build training data from them.

### Task interval (explicit metadata; no reliable episode markers)
`start = 125.0 s`, `end = 210.0 s` (≈85 s) in **ORIGINAL MCAP time**.
DexTeleop X/Y markers are not usable; the expected `/xr/left_hand_inputs buttons[2]/buttons[3]`
convention did not appear. Future collection: one recording per demonstration.

### Camera topics (all `ffmpeg_image_transport_msgs/msg/FFMPEGPacket`)
| camera | topic | packets | rate | span |
|---|---|---|---|---|
| left | `/left/color/image_raw/ffmpeg` | 13 950 | 44.999 Hz | 309.98 s |
| right | `/right/color/image_raw/ffmpeg` | 13 950 | 45.000 Hz | 309.98 s |
| head | `/xr_video_topic/ffmpeg` | 10 932 | 45.000 Hz | **242.91 s** |

All three cover 125–210 s. Head stops ≈243 s — **after** the task interval, so it does not affect
this episode, but it will matter for any trim past 243 s. Zero large gaps on all three.

### Other notable topics
Arms: `/{left,right}_arm/{joint_cmd,joint_states,target_ee_pose,current_ee_pose}`.
Grippers: `/{left,right}_gripper/{joint_cmd,joint_states}`.
Base: `/chassis/{joint_cmd,joint_states}`, `/chassis_target_vel`, `/can[0-4]/motor_{cmd,states}`.
Lift: `/kinco/{cmd_velocity,actual_velocity,actual_position}`.
XR: `/xr/{hmd_pose,left_aim_pose,right_aim_pose,left_hand_inputs,right_hand_inputs}`.
`/tf` 124 006; `/tf_static` **0 messages** (no static transform tree recorded).

---

## 5. Canonical ACTION space — 19-D. DO NOT CHANGE without asking.

| dims | meaning | source index in the 72-D DexTeleop vector |
|---|---|---|
| 0:7 | left arm joint target positions | `action[0:7]` |
| 7:14 | right arm joint target positions | `action[8:15]` |
| 14 | left raw gripper command | `action[39]` |
| 15 | right raw gripper command | `action[47]` |
| 16:19 | mobile-base motor velocity commands | `action[65:68]` |

Lift (`action[71]`, `kinco_velocity`) is omitted: **constant 0** in this demonstration.
Gripper commands stay **raw and continuous** for now (no binarization).

**Why 39/47 and not 7/15:** the 72-D layout is
`[pos 0:16 | vel 16:32 | effort 32:48 | ee_pose 48:62 | chassis 62:71 | kinco 71]`, with grippers
occupying slots 7/15 (left/right) inside each block. In the **command** vector,
`gripper position/velocity are identically 0` — `/‌*_gripper/joint_cmd` only populates the
**effort** field. So `action[39]`/`action[47]` (`left/right_gripper_effort`) ARE the gripper command.
Verified over 125–210 s: `act[7]=act[15]=act[23]=act[31]` are all exactly constant 0;
`act[39]` std 0.200 (11 unique), `act[47]` std 1.267 (22 unique), both ranging −1.6 … +2.0.

### Verified physical events (in original MCAP time)
- **t ≈ 180.8 s** — right gripper closes/grabs the object; raw command strongly negative (≈ −1.6).
  → **negative = closing/holding direction.** (Which value is "fully open" is NOT established;
  the +2.0 observation at 224.9 s is outside the task interval and immediately after a command change.)
- **t ≈ 138 s** — base moves forward.
- **t ≈ 197 s** — base moves backward.

---

## 6. PROPRIOCEPTION — DECIDED 2026-09-11 (gripper = measured POSITION)

Canonical 19-D proprio (measured): `0:7` left arm `joint_states.position`, `7:14` right arm, `14/15`
left/right **gripper `joint_states.position`**, `16:19` base `chassis/joint_states.velocity`.
Definition lives in `src/dexteleop/schema.py` (`PROPRIO_DIMS`, `ACTION_DIMS`) and is unit-tested.

**Evidence from the original MCAP (full-bag decode, `reports/rec_..._inspection.md`):**
- `/*_gripper/joint_cmd` (joint `l_joint8`/`r_joint8`): `position` and `velocity` are **exactly 0 in all 18 167
  messages**; `effort` spans −1.6…+2.0 (28/25 unique values). ⇒ the gripper COMMAND is carried in `effort`.
- `/*_gripper/joint_states`: all three fields vary — position −0.045…1.150 (L) / 0.441…1.162 (R), velocity
  ±6.7, effort −8.3…0.6. The schema is stock `sensor_msgs/JointState` (position rad, velocity rad/s, effort Nm).
- Causality (15 Hz canonical, 125–210 s): corr(cmd, position) right = +0.971 same-time, +0.978 at 0.13 s lag;
  left +0.395 → +0.863 at 0.13 s lag (left only blips closed twice). Position FOLLOWS the command ⇒ measured.
- `cmd = +2.0` ⇔ position ≈ 1.02–1.11 (**open**); `cmd = −1.6` ⇔ position drops (**closing**). Right gripper
  at t≈181.7 s closes onto the object and stalls at position ≈ 0.53 until release at ≈193.9 s; left gripper at
  t≈138.9 s closes on nothing and reaches ≈ −0.045 (fully shut).
- **Measured effort IS a contact/load signal**: right `joint_states.effort` ≈ +0.2 when open, **≈ −2.7 sustained
  while holding the object** (181.7–193.9 s); the empty left close only produces a transient spike. Kept as
  `aux[:, 0:2]` in the canonical npz (with measured gripper velocity as `aux[:, 2:4]`) for a later ablation —
  NOT in the 19-D proprio.

## 7. Language prompt — PROVIDED 2026-09-11

`rec_20260821_052359_f54c26c9`: **"Insert the red cylinder onto the vertical peg."** (set in
`configs/episodes/rec_20260821_052359_f54c26c9.yaml`, recorded in the canonical `meta.json`; blocker cleared).
Prompts are per-episode config fields — never invent one; a `null` prompt is reported as a blocker.

## 8. Canonical pipeline — IMPLEMENTED (`src/dexteleop/`, `scripts/`, `tests/`)

`mcap_io.py` (indexed topic read, source indices + log/publish/header times) → `sync.py` (grid + `hold|nearest|next|interp`
alignment, ages, gap reports) → `video.py` (HEVC packet decoder with NAL-parsed keyframes) → `extract.py`
(19-D assembly, tolerances, frames, `meta.json`) → `report.py` (`validation.md`, QC plots, preview.mp4,
legacy-npz cross-check). Config per episode: `configs/episodes/<ep>.yaml` (trim, rate, alignment, tolerances,
camera eye/resize, **`prompt` required key**). 19 unit tests (`tests/`).

**Time base (confirmed):** MCAP `log_time`. `header.stamp` is unusable across topics: arm/CAN topics are
+21 days off, cameras +5.03 s, `/chassis/joint_cmd` ≈ 0. Camera `pts` is µs on the sender clock and agrees
with log_time to a constant −2.6 ms (std 0.0) ⇒ log_time jitter is negligible.
**Semantics:** proprio `hold` (latest ≤ tick), action `next` (earliest ≥ tick), camera `hold`; every row keeps
the source message index, source log_time (and pts / header.stamp for frames) and the age.

**Legacy clock offset (important):** the Mac-side npz/MP4s use a "dataset clock" that is **+0.86 s later than
MCAP time** (legacy t=125.0 ⇔ MCAP 125.86 s). Our extraction reproduces the legacy proprio/action **exactly
(1.0000 of entries)** when evaluated at the legacy rows' own bag timestamps, so the extraction is validated —
but the user-verified event times (gripper close "180.8 s", base fwd "138", back "197") are on the legacy
clock; in MCAP time they are ≈181.7 s, ≈138.3–138.9 s, ≈197.5–198 s.
**DECISION 2026-09-11: canonical trim = 125.86–210.86 s MCAP log_time** (the legacy-clock 125–210 interval).

**Episode 1 outputs (holylfs05):**
`/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/rec_20260821_052359_f54c26c9/`
- **`t125.86-210.86_hz15/` — THE canonical episode**: K=1276, frames for 3 cams (left eye of each stereo pair;
  wrist 1280×800, head 960×960 JPEG q95), 888 MB. `validation.md` + `qc/`.
- `t125.86-210.86_hz45/` — K=3826, npz only (`--no-frames`).
- `t125-210_hz15/`, `t125-210_hz45/` — earlier MCAP-clock trim, **diagnostic only, not the final episode**.
Cameras are **side-by-side stereo**: wrist 2560×800 = 2×1280×800, head 3840×1920 = 2×1920×1920 fisheye.
HEVC, keyframe every 45 packets (1 s). Decoding 85 s × 3 cams ≈ 6 min on a login node.

Stream facts inside 125–210 s: 0 missing / 0 stale samples at tolerances (proprio ≤50 ms, action ≤100 ms,
camera ≤60 ms), 0 reused frames, 0 raw-stream gaps >2× median except the XR-driven gripper cmd (77 Hz, jittery
10–22 ms, harmless). Arm `joint_states` end at 247.7 s and XR/gripper-cmd topics at 235.9 s — after the task.
A person is visible in the workspace ≈149 s and a human hand enters the wrist views ≈161–174 s.

## 8b. Episode 2 — `rec_20260821_035532_74a3cb58` (added 2026-09-14)

| field | value (as supplied by the user, not inferred) |
|---|---|
| record_id | `rec_20260821_035532_74a3cb58` |
| actual_date | 2026-09-14 |
| prompt | **"Insert the cylinder from the left peg into the right peg."** |
| success | **true** |
| trim | **start_s = 10.0, end_s = 28.0** (MCAP log_time − bag start; chosen from the full-recording preview MP4) |
| config | `configs/episodes/rec_20260821_035532_74a3cb58.yaml` |
| canonical output | `/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/rec_20260821_035532_74a3cb58/t10-28_hz15/` (K=271 @15 Hz, 189 MB) |
| preview (full bag) | `/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla/videos/rec_20260821_035532_74a3cb58_preview_full_15hz.mp4` |

Raw: 163 MB MCAP, 59.607 s, 313 239 msgs, all 3 cameras 45 Hz for the whole bag; first decodable left-camera
keyframe at packet 31 (t≈0.7 s). This bag HAS X/Y marker presses (`buttons[2]` rise 1.89 s, `buttons[3]` rise
36.29 s) — recorded for reference, not used. Validation (`validation.md`): 0 missing / 0 out-of-tolerance
proprio/action/camera samples, 0 reused frames; proprio age ≤6.2 ms, action ≤20.8 ms, camera ≤24.4 ms.
Task is done with the **left** gripper (closes 18.53 s, holds with effort ≈ −2.69, releases ≈24.9 s);
the **right gripper command is constant +2.0 and its position std 0.0002 over the whole trim** (a degenerate
dim for any single-episode normalization — covered by `quantile_min_range`). Base moves at both trim edges
(10.0–11.1 s and 26.9–28.0 s). Manifests: added to `configs/lerobot/dexteleop_all.yaml` (both episodes) and
single-demo `configs/lerobot/dexteleop_overfit_rec0821_035532.yaml`. No LeRobot conversion/training run yet.
`scripts/preview_recording.py` makes the full-bag 3-camera preview with the log_time clock burned in.

## 8c. Recording 3 — `rec_20260821_052219_d4f4fc76`, 6 segments (added 2026-09-14)

record_id `rec_20260821_052219_d4f4fc76`, actual_date 2026-09-14, all `success: true`. Raw: 510 MB MCAP, messages
span 152.537 s (metadata.yaml says 157.56 s — its `starting_time` is 2.8 ms before the first message and its
`duration` runs 5 s past the last; the canonical clock is first-message log_time), all 3 cameras 45 Hz for the
whole bag, no dropouts, **no X/Y markers** (`buttons[4]` = the left-gripper trigger: its presses coincide with
every left-gripper close). Boundaries + prompts supplied by the user from the head-only preview
(`videos/rec_20260821_052219_d4f4fc76_preview_head_10fps.mp4`, made by `scripts/preview_head_stream.py` —
streaming, O(1) memory, frames piped to a static ffmpeg from `imageio-ffmpeg`).
Configs `configs/episodes/rec_20260821_052219_d4f4fc76_seg0{1..6}.yaml`; outputs under
`/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/rec_20260821_052219_d4f4fc76/`:

| seg | start_s | end_s | prompt (verbatim) | K | variant dir | size |
|---|---|---|---|---|---|---|
| 01 | 15 | 30 | Stack the red cylinder on the green cylinder. | 226 | `t15-30_hz15` | 145 M |
| 02 | 82 | 93 | Stack the red cylinder on the green cylinder. | 166 | `t82-93_hz15` | 112 M |
| 03 | 93.5 | 105 | Remove the red cylinder from the green cylinder. | 173 | `t93.5-105_hz15` | 120 M |
| 04 | 105.5 | 116 | Stack the red cylinder on the green cylinder. | 158 | `t105.5-116_hz15` | 108 M |
| 05 | 116 | 125 | Remove the red cylinder from the green cylinder. | 136 | `t116-125_hz15` | 94 M |
| 06 | 125.5 | 135 | Stack the red cylinder on the green cylinder. | 143 | `t125.5-135_hz15` | 98 M |

Validation (all 6): 0 missing / 0 out-of-tolerance proprio, action, camera; 0 reused frames; 0 NaN; frames
written for all 3 cameras; proprio age ≤5.3 ms, action ≤21 ms, camera ≤18 ms. Every segment is a LEFT-arm task
(left gripper closes once per segment at 19.87 / 86.07 / 98.97 / 109.23 / 120.47 / 128.63 s); **right gripper
idle (+2.0) and base idle (0 % activity) in all six** → degenerate dims for single-recording normalization
(covered by `quantile_min_range`). A person is visible in the background of the head view.
Manifests: all 6 appended to `configs/lerobot/dexteleop_all.yaml` (now 8 episodes); per-recording
`configs/lerobot/dexteleop_overfit_rec0821_052219.yaml`. No LeRobot conversion/training run yet.

## 8d. Deployment facts + camera-eye decision (2026-09-14)

Official deployment stack: `github.com/dexteleop/openpi` → `examples/teleavatar_v2/` (clone in `${NS}/tmp/dexteleop_openpi`).
Client host must be on the robot LAN (robot pushes a tiled 2720×1280 RTP/H.265 stream to the client's IP, port 8890;
ROS 2 topics via `zenoh-bridge-ros2dds -e tcp/<ROBOT_IP>:9000`, `ROS_DOMAIN_ID=29`); client needs ROS 2 Humble +
GStreamer `nvh265dec` (their conda `environment.yml`). Robot must be in **API** run mode, arms in **joint** mode.
Commands: `/api/{left,right}_arm/joint_cmd` (JointState positions, clamp to `arm_config.yml` limits),
`/api/{left,right}_gripper/cmd` (Float32 trigger∈[0,1]; effort = +2.0·(1−t/0.10) for t<0.10 else −1.6·(t−0.10)/0.90;
our raw-effort action must be mapped with the inverse before publishing), `/api/fsm/enable`. **No chassis/base API
topic** in their client. Policy server = `serve_policy.py` (same as ManiSkill mp74).
**DECISION: deployment machine = the lab workstation** (Ubuntu 22.04, RT kernel 6.8.2-rt11, 2× RTX 6000 Ada 48 GB,
spare NIC `enp37s0f0`; needs a reboot for the NVIDIA driver mismatch + `gstreamer1.0-plugins-bad`). It is client AND
policy server; the cluster stays training/eval only; checkpoints rsync'd from netscratch.
**Camera-eye decision**: at deployment the client provides the INNER eyes — head LEFT eye (960×960), right wrist LEFT
eye, **left wrist RIGHT eye**. All canonical episodes were re-extracted 2026-09-14 with `left: {eye: right}`
(ingest_phase2 template + all configs updated); the earlier `dexteleop_train`/`dexteleop_all` builds and the
`dexteleop_v1_lora` run used the left eye of the left wrist (plumbing test only).
v1 replay @2000 on a training segment: arm corr 0.45–0.99, MAE 0.006–0.085 rad, idle dims predicted ≈ exactly,
inference 94 ms. v2 (`dexteleop_v2_lora`) = same config on the re-extracted set with seg06 of rec_..052219 held out.

## 9. Tooling gaps on this cluster (verified)

- conda env **`dexteleop-data`** exists (`/n/holylabs/LABS/hankyang_lab/Lab/jackbjed/conda_envs/dexteleop-data`,
  Python 3.11.16, 2.0 GB). Has: numpy 2.4.6, opencv 4.13.0, PyYAML 6.0.3, matplotlib 3.9.1, pandas, pyarrow.
  **Added 2026-09-11 (pip, cache on netscratch): `mcap` 1.4.0, `mcap-ros2-support`, `av` 18.1.0 (bundled ffmpeg,
  HEVC decode + libx264 encode), `pytest`, `tqdm`.** Still missing: `h5py`, `torch`, `lerobot` (not needed here).
- `dp2` (`.../conda_envs/dp2`) has torch 2.6.0, lerobot 0.4.4, **av 15.1.0**, h5py 3.15.1, cv2 4.12.0 — but no `mcap`.
- No `ffmpeg`/`ffprobe` binary on PATH and no ffmpeg module. PyAV (in `dp2`) bundles the ffmpeg libs
  and is the realistic decoder for the `FFMPEGPacket` streams.
- `dp3` and `dp_env3` conda envs have **no torch** — they are not usable DP environments.

## 10. Diffusion Policy (located, NOT modified)

`/n/home08/jackbjed/scaling-diffusion-policy` — a fork of ManiSkill (git: `c0e510c changes`, `cc6267f init`).
DP baseline code: `.../ManiSkill/examples/baselines/diffusion_policy/` (`train.py`, `train_rgbd.py`,
`evaluations.py`, `diffusion_policy/`, plus CFM/energy variants). Its README's `diffusion-policy-ms` conda env **exists at `~/miniconda3/envs/diffusion-policy-ms` (9.0 G, home08)** — not yet verified to run.
Do not touch until the canonical representation and π0.5 integration are established.

---

## 11. Rules for this project

1. Inspect the checked-out OpenPI code — **never guess OpenPI APIs from memory.**
2. Never modify or overwrite the original MCAP.
3. Never write large data to `/n/home08` (it is 100 % full) or `/n/holylabs` (92.5 % group quota).
4. Do not change the 19-D action definition without asking.
5. No cue inpainting in DexTeleop.
6. No invented task prompts.
7. Do not create a second openpi checkout.
8. Regenerate `pi/openpi_changes/openpi.patch` before committing ManiSkill-side changes — and fix its
   coverage gaps (§1) first.
