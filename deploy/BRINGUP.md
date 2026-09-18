# TeleAvatar 2 bring-up checklist (workstation `~/tdieudonne/dexteleop/`)

Every stage gates the next. A human holds the E-stop from stage 3 on. Never run `zero.py`/`replay_episode.py` and the
client at the same time (both publish `/api/*/joint_cmd`).

## 0. Software (no robot)
- [ ] `teleavatar_client` conda env from `openpi_teleavatar/environment.yml`; `pip install -e openpi_teleavatar/packages/openpi-client`
- [ ] inside it: `gst-inspect-1.0 nvh265dec h265parse rtph265depay` all OK (else `sudo apt install gstreamer1.0-plugins-bad`, or CPU `avdec_h265` for a first test)
- [ ] `zenoh-bridge-ros2dds` in `~/.local/bin` (version matching the robot's router)
- [ ] `openpi_serve/` per `openpi_changes/APPLY.md`; `OPENPI_DATA_HOME` → existing `robomme_policy_learning/openpi-assets`
- [ ] checkpoint pulled: `rsync -avP jackbjed@login.rc.fas.harvard.edu:/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla/checkpoints/pi05_dexteleop_v1/dexteleop_v3_stack_lora/9999/ ~/tdieudonne/dexteleop/checkpoints/dexteleop_v3_stack_lora/9999/`
      (must contain `params/` and `assets/local/dexteleop_train/norm_stats.json`)
- [ ] one canonical episode pulled for replay/dry-run (≈100 MB): `…/dexteleop-vla-data/canonical/rec_20260821_052219_d4f4fc76/t125.5-135_hz15/`
- [ ] policy server offline test: `cd openpi_serve && uv run scripts/serve_policy.py --port 8000 policy:checkpoint --policy.config=pi05_dexteleop_v1 --policy.dir=<ckpt>/9999`
      then in another shell `uv run python -c "from openpi.policies import dexteleop_policy as d; from openpi_client import websocket_client_policy as w; import numpy as np; p=w.WebsocketClientPolicy('127.0.0.1',8000); a=np.asarray(p.infer(d.make_dexteleop_example())['actions']); print(a.shape, a[0,:19].round(3))"`
      → shape (10, 19) (or (H, 19)), finite values, first call ~40 s (JAX compile), then ~100 ms
- [ ] client dry run (no ROS): `python dexteleop-vla/deploy/dexteleop_client.py --mode replay --episode <ep> --dry-run --speed 100`
      and `python dexteleop-vla/deploy/dexteleop_client.py --mode auto --dry-run --episode <ep> --prompt "Stack the red cylinder on the green cylinder." --max-chunks 3`
      (with the server running) → prints chunks, writes `deploy/logs/run_*.json`

## 1. Network + robot config (robot powered, arms NOT enabled yet)
- [ ] cable robot switch ↔ `enp37s0f0`; static IP on the robot subnet (`nmcli con add type ethernet ifname enp37s0f0 con-name robot ip4 <IP>/24`)
- [ ] robot remoteApp → System Config: Run mode **API**, left+right arm enabled, control mode **joint**, remote IP = workstation `enp37s0f0` IP → Save config → Apply
- [ ] `export ROS_DOMAIN_ID=29; zenoh-bridge-ros2dds -e tcp/<ROBOT_IP>:9000` (keep running); `ros2 topic list` shows `/left_arm/joint_states`, `/left_gripper/joint_states`, `/chassis/joint_states`, …
- [ ] `ros2 topic echo /left_gripper/joint_states --once` → position ≈ 1.1 (open) — matches training data
- [ ] `python openpi_teleavatar/examples/teleavatar_v2/test.py` → six eye crops saved; head_left_eye 960×960, wrist eyes 400×640

## 2. Control path with KNOWN-GOOD actions (E-stop in hand)
- [ ] `python openpi_teleavatar/examples/teleavatar_v2/zero.py` → arms ease to the home pose, exits when converged (proves /api/*/joint_cmd works)
- [ ] replay a recorded demo at 30 % speed, starting from ITS start pose:
      `python dexteleop-vla/deploy/dexteleop_client.py --fork openpi_teleavatar --mode replay --episode <ep> --go-to-start --speed 0.3 --hold-right-arm --mode-confirm-replay`
      (`--go-to-start` eases the arms to the episode's frame-0 pose through the safety layer before replaying)
      PASS = the left arm reproduces the recorded stack (cylinders placed as in the video); grippers open/close at the right moments
- [ ] same at `--speed 1.0`

## 3. Policy, confirm-per-chunk (E-stop in hand; objects placed as in a training demo)
- [ ] server up; `python dexteleop-vla/deploy/dexteleop_client.py --fork openpi_teleavatar --mode confirm --prompt "Stack the red cylinder on the green cylinder." --speed 0.5 --hold-right-arm`
      Use `--go-to-start --episode <ep>` so the policy starts from a pose it saw in training (the generic home pose is out of
      distribution: in the dry run the first chunk was up to 0.44 rad from frame 0). Inspect each printed chunk (max |Δq|, gripper
      trigger) before Enter. A large |Δq| on the FIRST chunk is expected if the start pose differs and is ramped by the safety layer;
      from chunk 1 on, reject anything with |Δq| > 0.3 rad within one chunk.
## 4. Policy, auto (E-stop in hand)
- [ ] `--mode auto --speed 0.7`, then `--speed 1.0`; 5 trials at one placement, then 3 placements × 5 trials = first success rate.
