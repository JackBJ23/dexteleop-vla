#!/usr/bin/env python3
"""DexTeleop 19-D deployment client for TeleAvatar 2 (staged, safety-first).

Wraps the official `examples/teleavatar_v2/ros2_interface.TeleavatarROS2Interface` (dexteleop/openpi fork) so
ROS 2 / zenoh / GStreamer plumbing stays theirs, and adapts it to OUR policy contract:

  observation -> policy   head_left_eye  -> observation/head_image   (960x960)
                          left_wrist_right_eye -> observation/left_image   (400x640, inner eye)
                          right_wrist_left_eye -> observation/right_image  (400x640, inner eye)
                          19-D state: left arm q[7] | right arm q[7] | left gripper pos | right gripper pos | chassis vel[3]
                          (needs the extra subscriptions /{left,right}_gripper/joint_states, /chassis/joint_states)
  policy -> robot         19-D action: arms 0:14 -> /api/{left,right}_arm/joint_cmd (their clamp to arm_config.yml),
                          gripper raw effort 14/15 -> inverse trigger curve -> /api/{left,right}_gripper/cmd,
                          base 16:19 -> DROPPED (no chassis API topic on the platform).

Modes (all run through the same publish path, so stage N validates stage N+1):
  --mode replay  --episode <canonical_dir>   replay the RECORDED actions of a canonical episode (no policy)
  --mode confirm                             policy; print every chunk, wait for Enter before executing it
  --mode auto                                policy; execute chunks automatically (safety layer still on)
  --dry-run                                  no ROS: observations come from --episode frames, actions are logged only.
Safety layer (all modes): per-step joint-delta clamp (--max-joint-delta, rad, vs the last commanded pose),
speed scaling (--speed <1 stretches chunk time), first-command ramp from the measured pose, right-arm hold option,
E-stop = Ctrl+C (stops publishing; the platform holds position).

Run inside the `teleavatar_client` conda env with the dexteleop/openpi fork on sys.path:
  export ROS_DOMAIN_ID=29
  python deploy/dexteleop_client.py --fork ~/tdieudonne/dexteleop/openpi_teleavatar --mode replay --episode <dir> --speed 0.3
  python deploy/dexteleop_client.py --fork ... --mode confirm --prompt "Stack the red cylinder on the green cylinder."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

LOG = logging.getLogger("dexteleop_client")
PROPRIO_NAMES = ["left_joint%d_pos" % i for i in range(1, 8)] + ["right_joint%d_pos" % i for i in range(1, 8)] + \
                ["left_gripper_pos", "right_gripper_pos", "base_motor1_vel", "base_motor2_vel", "base_motor3_vel"]
POLICY_IMAGE_KEYS = {"observation/head_image": "head_left_eye", "observation/left_image": "left_wrist_right_eye",
                     "observation/right_image": "right_wrist_left_eye"}
GRIP_OPEN_EFFORT = 2.0   # raw command for "open" in the training data (+2.0); closing goes to -1.6


def effort_to_trigger(effort: float) -> float:
    """Inverse of the TA2 trigger->effort curve (identical to ros2_interface._gripper_trigger)."""
    t = 0.10 * (1.0 - effort / 2.0) if effort > 0 else 0.10 - effort * 0.90 / 1.6
    return float(np.clip(t, 0.0, 1.0))


def trigger_to_effort(t: float) -> float:
    return 2.0 * (1.0 - t / 0.10) if t < 0.10 else -1.6 * (t - 0.10) / 0.90


class SafetyLayer:
    """Stateless checks + stateful per-step delta clamp on the 14 arm joints."""

    def __init__(self, max_joint_delta: float, hold_right_arm: bool, right_arm_hold_pose: np.ndarray | None):
        self.max_joint_delta = max_joint_delta
        self.hold_right_arm = hold_right_arm
        self.right_hold = right_arm_hold_pose
        self.last_cmd: np.ndarray | None = None       # last commanded arm14
        self.n_clamped = 0

    def filter_step(self, action19: np.ndarray, measured_arm14: np.ndarray | None) -> tuple[np.ndarray, dict]:
        a = np.asarray(action19, dtype=np.float64).copy()
        arm = a[0:14]
        ref = self.last_cmd if self.last_cmd is not None else measured_arm14
        info = {}
        if ref is not None:
            delta = arm - ref
            over = np.abs(delta) > self.max_joint_delta
            if over.any():
                self.n_clamped += 1
                info["clamped_joints"] = np.flatnonzero(over).tolist()
                info["max_delta"] = float(np.abs(delta).max())
                arm = ref + np.clip(delta, -self.max_joint_delta, self.max_joint_delta)
        if self.hold_right_arm and self.right_hold is not None:
            arm[7:14] = self.right_hold
        self.last_cmd = arm.copy()
        a[0:14] = arm
        return a, info


def action19_to_action16(a19: np.ndarray) -> np.ndarray:
    """Our 19-D -> the official interface's 16-D [left(7), left_grip_trigger, right(7), right_grip_trigger].
    NOTE: their publish_action applies _gripper_trigger() itself to indices 7/15 (expects raw EFFORT), so we pass the
    raw effort through unchanged; base dims 16:19 are dropped."""
    a16 = np.zeros(16, dtype=np.float64)
    a16[0:7] = a19[0:7]; a16[7] = a19[14]; a16[8:15] = a19[7:14]; a16[15] = a19[15]
    return a16


# ----------------------------------------------------------------------------------------------------------------------
# ROS side (only imported when not --dry-run)
# ----------------------------------------------------------------------------------------------------------------------
def make_ros_interface_class(fork_root: Path):
    sys.path.insert(0, str(fork_root))
    from examples.teleavatar_v2 import ros2_interface as base            # noqa: E402
    from sensor_msgs.msg import JointState                               # noqa: E402

    class DexTeleopROS2Interface(base.TeleavatarROS2Interface):
        """Adds gripper + chassis joint_states and builds the 19-D DexTeleop state."""

        def __init__(self, **kw):
            super().__init__(node_name="dexteleop_openpi_interface", **kw)
            self.extra_states: dict = {}
            self.extra_stamps: dict = {}
            for grp, topic in (("left_gripper", "/left_gripper/joint_states"), ("right_gripper", "/right_gripper/joint_states"),
                               ("chassis", "/chassis/joint_states")):
                self.create_subscription(JointState, topic, lambda m, g=grp: self._extra_cb(m, g), 10)
            self.logger.info("DexTeleop interface: +/left_gripper, /right_gripper, /chassis joint_states")

        def _extra_cb(self, msg, grp):
            with self.lock:
                self.extra_states[grp] = msg
                self.extra_stamps[grp] = time.time()

        def get_observation19(self):
            """Their observation (images + arm states, staleness-checked) + our 19-D state; None = do not act."""
            obs = self.get_observation()
            if obs is None:
                return None
            now = time.time()
            with self.lock:
                ex = dict(self.extra_states); st = dict(self.extra_stamps)
            dead = [g for g in ("left_gripper", "right_gripper", "chassis") if g not in ex or now - st[g] > self.sensor_timeout]
            if dead:
                self.logger.error(f"Observation unavailable — stale/missing: {dead}", throttle_duration_sec=1.0)
                return None
            s48 = obs["state"]
            state19 = np.zeros(19, dtype=np.float32)
            state19[0:7] = s48[0:7]; state19[7:14] = s48[8:15]
            state19[14] = self._extract_joint_field(ex["left_gripper"], "position", 1)[0]
            state19[15] = self._extract_joint_field(ex["right_gripper"], "position", 1)[0]
            state19[16:19] = self._extract_joint_field(ex["chassis"], "velocity", 3)
            images = {k: obs["images"][{"head_left_eye": "head_camera", "left_wrist_right_eye": "left_color",
                                        "right_wrist_left_eye": "right_color"}[v]] for k, v in POLICY_IMAGE_KEYS.items()}
            return {"state": state19, "images": images}

        def measured_arm14(self):
            return self._current_arm_positions()

    return DexTeleopROS2Interface


class RosRobot:
    """Thin runtime around the interface: spin thread, observation, publish of a 19-D action."""

    def __init__(self, fork_root: Path, control_hz: float, interp_hz: float):
        import rclpy
        cls = make_ros_interface_class(fork_root)
        self._iface = None; started = threading.Event()

        def spin():
            rclpy.init()
            self._iface = cls(control_frequency=control_hz, interp_frequency=interp_hz, interpolate=True)
            ex = rclpy.executors.MultiThreadedExecutor(); ex.add_node(self._iface); started.set()
            try:
                ex.spin()
            finally:
                ex.shutdown(); self._iface.destroy_node(); rclpy.shutdown()
        threading.Thread(target=spin, daemon=True).start()
        t0 = time.time()
        while self._iface is None and time.time() - t0 < 10: time.sleep(0.1)
        if self._iface is None or not started.wait(5): raise RuntimeError("ROS interface failed to start")
        if not self._iface.wait_for_initial_data(timeout=30.0): raise RuntimeError("no initial sensor data (video/joints)")
        LOG.info("ROS interface up; waiting for gripper/chassis states…")
        t0 = time.time()
        while self._iface.get_observation19() is None and time.time() - t0 < 10: time.sleep(0.2)
        if self._iface.get_observation19() is None: raise RuntimeError("gripper/chassis joint_states not received")

    def observe(self): return self._iface.get_observation19()
    def measured_arm14(self): return self._iface.measured_arm14()
    def publish19(self, a19): self._iface.publish_action(action19_to_action16(a19))


class DryRobot:
    """No ROS: observations come from a canonical episode's frames/state; actions are only logged."""

    def __init__(self, episode: Path):
        import cv2
        self.ep = episode; self.z = np.load(episode / "episode.npz"); self.meta = json.loads((episode / "meta.json").read_text())
        self.k = 0; self.cv2 = cv2; self.published: list[np.ndarray] = []
        self.K = self.z["proprio"].shape[0]

    def observe(self):
        k = min(self.k, self.K - 1)
        def img(cam, shape):
            im = self.cv2.cvtColor(self.cv2.imread(str(self.ep / "frames" / cam / f"{k:06d}.jpg")), self.cv2.COLOR_BGR2RGB)
            return self.cv2.resize(im, (shape[1], shape[0]), interpolation=self.cv2.INTER_AREA)
        return {"state": self.z["proprio"][k].astype(np.float32),
                "images": {"observation/head_image": img("head", (960, 960)), "observation/left_image": img("left", (400, 640)),
                           "observation/right_image": img("right", (400, 640))}}

    def measured_arm14(self): return self.z["proprio"][min(self.k, self.K - 1), 0:14].astype(np.float64)
    def publish19(self, a19): self.published.append(np.asarray(a19, dtype=np.float32).copy()); self.k += 1


# ----------------------------------------------------------------------------------------------------------------------
def describe_chunk(chunk: np.ndarray, measured14: np.ndarray | None) -> str:
    c = np.asarray(chunk); lines = [f"chunk {c.shape[0]} steps"]
    if measured14 is not None:
        d = c[:, 0:14] - measured14[None]
        lines.append(f"  max |Δq| vs measured: left {np.abs(d[:, 0:7]).max():.3f} rad, right {np.abs(d[:, 7:14]).max():.3f} rad")
    step = np.abs(np.diff(c[:, 0:14], axis=0)).max() if c.shape[0] > 1 else 0.0
    lines.append(f"  max per-step |Δq|: {step:.3f} rad;  gripper L {c[0,14]:+.2f}→{c[-1,14]:+.2f} (trigger {effort_to_trigger(c[-1,14]):.2f}), "
                 f"R {c[0,15]:+.2f}→{c[-1,15]:+.2f};  base |v| max {np.abs(c[:,16:19]).max():.2f} (dropped)")
    return "\n".join(lines)


def run(args):
    fork = Path(args.fork).expanduser() if args.fork else None
    robot = DryRobot(Path(args.episode)) if args.dry_run else RosRobot(fork, args.control_hz, args.interp_hz)
    dt = 1.0 / args.control_hz / max(args.speed, 1e-3)
    m0 = robot.measured_arm14()
    right_hold = m0[7:14].copy() if (m0 is not None and args.hold_right_arm) else None
    safety = SafetyLayer(args.max_joint_delta, args.hold_right_arm, right_hold)
    log_rows = []

    def execute_chunk(chunk: np.ndarray, tag: str):
        for i, a in enumerate(chunk):
            a_f, info = safety.filter_step(a, robot.measured_arm14())
            robot.publish19(a_f)
            log_rows.append({"t": time.time(), "tag": tag, "i": i, "action": a.tolist(), "sent": a_f.tolist(), **info})
            if info: LOG.warning(f"  step {i}: delta clamp {info}")
            time.sleep(dt)

    if getattr(args, "go_to_start", False):
        if not args.episode: raise SystemExit("--go-to-start needs --episode (its frame-0 proprio is the target pose)")
        z0 = np.load(Path(args.episode) / "episode.npz"); target = z0["proprio"][0].astype(np.float64)
        m = robot.measured_arm14()
        if m is None: raise SystemExit("no measured arm pose")
        n = max(int(np.ceil(np.abs(target[0:14] - m).max() / (args.max_joint_delta * 0.5))), 1)
        LOG.info(f"GO-TO-START: easing arms to the episode's frame-0 pose over {n} steps (max |Δq| {np.abs(target[0:14]-m).max():.3f} rad); grippers OPEN")
        for i in range(1, n + 1):
            a = np.zeros(19); a[0:14] = m + (target[0:14] - m) * i / n; a[14] = a[15] = GRIP_OPEN_EFFORT
            a_f, _ = safety.filter_step(a, robot.measured_arm14()); robot.publish19(a_f); time.sleep(dt)
        time.sleep(1.0)
        if args.mode == "replay" or args.mode_confirm_replay:
            input("  at start pose. Enter to continue… ")

    if args.mode == "replay":
        ep = Path(args.episode); z = np.load(ep / "episode.npz"); A = z["action"]; H = args.chunk
        LOG.info(f"REPLAY {ep.name}: {A.shape[0]} recorded steps at {args.control_hz} Hz x speed {args.speed} (chunks of {H})")
        for s in range(0, A.shape[0], H):
            chunk = A[s:s + H]
            LOG.info(describe_chunk(chunk, robot.measured_arm14()))
            if args.mode_confirm_replay and input("  execute? [Enter=yes / q=quit] ").strip().lower() == "q": break
            execute_chunk(chunk, f"replay:{s}")
    else:
        from openpi_client import websocket_client_policy as wcp
        policy = wcp.WebsocketClientPolicy(host=args.host, port=args.port)
        LOG.info(f"policy server metadata: {policy.get_server_metadata()}")
        n_chunks = 0
        while args.max_chunks <= 0 or n_chunks < args.max_chunks:
            obs = robot.observe()
            if obs is None:
                LOG.error("no observation — not acting"); time.sleep(0.2); continue
            t0 = time.time()
            out = policy.infer({**obs["images"], "observation/state": obs["state"], "prompt": args.prompt})
            chunk = np.asarray(out["actions"])[: args.chunk, :19]
            LOG.info(f"[chunk {n_chunks}] infer {1e3*(time.time()-t0):.0f} ms  state L-grip {obs['state'][14]:.3f}\n" + describe_chunk(chunk, robot.measured_arm14()))
            if args.mode == "confirm":
                ans = input("  execute this chunk? [Enter=yes / s=skip / q=quit] ").strip().lower()
                if ans == "q": break
                if ans == "s": continue
            execute_chunk(chunk, f"policy:{n_chunks}"); n_chunks += 1
    out = Path(args.log); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "n_steps": len(log_rows), "n_delta_clamps": safety.n_clamped, "rows": log_rows}, default=str))
    LOG.info(f"done: {len(log_rows)} steps sent, {safety.n_clamped} delta-clamped; log -> {out}")
    return robot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["replay", "confirm", "auto"], required=True)
    ap.add_argument("--fork", help="path to the dexteleop/openpi clone (examples/teleavatar_v2); required unless --dry-run")
    ap.add_argument("--episode", help="canonical episode dir (replay source / dry-run observations)")
    ap.add_argument("--prompt", default=None); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--control-hz", type=float, default=15.0, help="training grid rate (canonical episodes are 15 Hz)")
    ap.add_argument("--interp-hz", type=float, default=200.0); ap.add_argument("--chunk", type=int, default=10, help="steps executed per policy query (training horizon 10)")
    ap.add_argument("--speed", type=float, default=1.0, help="<1 slows execution (0.3 = 3x slower)")
    ap.add_argument("--max-joint-delta", type=float, default=0.15, help="rad per control step, per joint")
    ap.add_argument("--hold-right-arm", action="store_true", help="freeze the right arm at its start pose (left-arm tasks)")
    ap.add_argument("--max-chunks", type=int, default=0); ap.add_argument("--mode-confirm-replay", action="store_true", help="confirm each replay chunk too")
    ap.add_argument("--go-to-start", action="store_true", help="first ease the arms to --episode's frame-0 pose (through the safety layer)")
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--log", default="deploy/logs/run_%d.json" % int(time.time()))
    args = ap.parse_args()
    if args.mode != "replay" and not args.prompt: ap.error("--prompt is required for policy modes (never invent one)")
    if (args.mode == "replay" or args.dry_run) and not args.episode: ap.error("--episode required")
    if not args.dry_run and not args.fork: ap.error("--fork required unless --dry-run")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    run(args)


if __name__ == "__main__":
    main()
