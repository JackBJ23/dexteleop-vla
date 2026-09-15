"""Canonical DexTeleop episodes -> ONE LeRobot dataset for the OpenPI pi0.5 stack.

Run with OpenPI's environment so the dataset format matches its pinned lerobot:
    cd /n/home08/jackbjed/ManiSkill/pi/openpi && HF_LEROBOT_HOME=<node-local> uv run python \
        /n/home08/jackbjed/dexteleop-vla/scripts/convert_canonical_to_lerobot.py --manifest <yaml> --repo-id local/<name>

Manifest (YAML list); the prompt comes from each episode's meta.json (required, non-null):
    - canonical: /n/holylfs05/.../canonical/rec_.../t125.86-210.86_hz15
      prompt_override: null     # optional; normally leave unset so meta.json is authoritative

LeRobot keys (consumed by openpi LeRobotDexTeleopDataConfig): head_image, left_image, right_image [S,S,3] uint8,
state [19] float32, actions [19] float32, task (prompt). fps = the canonical grid rate (must be integer).
Frames are added in canonical row order 0..K-1 with no gaps, so LeRobot `frame_index` == canonical row
(and via episode.npz `src_index/*`, `frame_src_index/*` -> the MCAP message/packet indices).
"""
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tyro
import yaml

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

CAMS = ("head", "left", "right")


@dataclass
class Args:
    manifest: str
    repo_id: str = "local/dexteleop_overfit1"
    image_size: int = 224          # pi0.5 resizes to 224 anyway; storing at 224 keeps the dataset small
    robot_type: str = "teleavatar2"
    max_frames_per_episode: int = 0   # >0: debug/smoke only
    image_writer_processes: int = 0   # SYNCHRONOUS by default (NFS-safe; see mp75b pattern)
    image_writer_threads: int = 0
    push_to_hub: bool = False


def _load_rgb(p: Path, s: int) -> np.ndarray:
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(p)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[0] != s or im.shape[1] != s:
        im = cv2.resize(im, (s, s), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(im, dtype=np.uint8)


def main(args: Args):
    entries = yaml.safe_load(Path(args.manifest).read_text())
    assert isinstance(entries, list) and entries, "manifest must be a non-empty YAML list"
    eps = []
    for e in entries:
        d = Path(e["canonical"]); meta = json.loads((d / "meta.json").read_text())
        prompt = e.get("prompt_override") or meta.get("prompt")
        if not prompt:
            raise ValueError(f"{d}: no prompt in meta.json and no prompt_override -- refusing to invent one")
        if meta.get("blockers"):
            raise ValueError(f"{d}: canonical episode has blockers {meta['blockers']}")
        for c in CAMS:
            assert (d / "frames" / c).is_dir(), f"{d}: missing frames/{c}"
        eps.append((d, meta, prompt))
    fps_f = float(eps[0][1]["grid"]["rate_hz"])
    assert all(float(m["grid"]["rate_hz"]) == fps_f for _, m, _ in eps), "all episodes must share rate_hz"
    assert fps_f == int(fps_f), f"LeRobot needs an integer fps, got {fps_f}"
    fps = int(fps_f)
    P0 = eps[0][1]["proprio_names"]; A0 = eps[0][1]["action_names"]
    assert len(P0) == 19 and len(A0) == 19

    out = HF_LEROBOT_HOME / args.repo_id
    if out.exists():
        shutil.rmtree(out)
    s = args.image_size
    features = {
        **{f"{c}_image": {"dtype": "image", "shape": (s, s, 3), "names": ["height", "width", "channel"]} for c in CAMS},
        "state": {"dtype": "float32", "shape": (19,), "names": P0},
        "actions": {"dtype": "float32", "shape": (19,), "names": A0},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, robot_type=args.robot_type, fps=fps, features=features,
                               image_writer_processes=args.image_writer_processes,
                               image_writer_threads=args.image_writer_threads)
    total = 0
    for d, meta, prompt in eps:
        z = np.load(d / "episode.npz")
        P, A = z["proprio"], z["action"]
        assert meta["proprio_names"] == P0 and meta["action_names"] == A0
        K = P.shape[0] if args.max_frames_per_episode <= 0 else min(P.shape[0], args.max_frames_per_episode)
        assert not np.isnan(P[:K]).any() and not np.isnan(A[:K]).any()
        for k in range(K):
            ds.add_frame({
                **{f"{c}_image": _load_rgb(d / "frames" / c / f"{k:06d}.jpg", s) for c in CAMS},
                "state": P[k].astype(np.float32), "actions": A[k].astype(np.float32),
                "task": prompt,
            })
        ds.save_episode()
        total += K
        print(f"  + {d.name} ({d.parent.name}): {K} frames, prompt={prompt!r}")
    src = {"episodes": [{"canonical": str(d), "episode_id": m["episode_id"], "variant": m["variant"], "prompt": p,
                         "mcap": m["mcap"]["path"], "mcap_id": m["mcap"]["sha256_head_1mb"], "schema": m["schema_version"]}
                        for d, m, p in eps], "fps": fps, "image_size": s, "frame_index_equals_canonical_row": True}
    (out / "dexteleop_sources.json").write_text(json.dumps(src, indent=1))
    if args.push_to_hub:
        ds.push_to_hub(tags=["dexteleop", "teleavatar2"], private=True, push_videos=True)
    print(f"DONE: {total} frames, {len(eps)} episodes -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
