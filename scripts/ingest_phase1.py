#!/usr/bin/env python
"""PHASE 1 of ingesting a DexTeleop recording: inspect the MCAP + make the head-camera preview for trim annotation.

    $PY scripts/ingest_phase1.py <record_id>            # e.g. rec_20260821_052219_d4f4fc76

Finds the raw recording (data/raw/<record_id>/, symlinking it from the holylfs05 raw store if needed), runs the
full-bag inspection, the streaming head-only preview MP4 (MCAP log_time clock burned in), extracts X/Y marker and
button edges, and writes reports/<record_id>_phase1.md. Does NOT choose task/trim, extract, convert or train.
"""
import argparse, subprocess, sys, json, os
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RAW_STORE = Path("/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/raw")
VIDEOS = Path("/n/netscratch/hankyang_lab/Lab/jackbjed/dexteleop-vla/videos")
PY = sys.executable


def find_raw(rid: str) -> Path:
    d = REPO / "data" / "raw" / rid
    if not d.exists():
        src = RAW_STORE / rid
        if not src.is_dir():
            sys.exit(f"raw recording not found: {d} nor {src}")
        d.symlink_to(src); print(f"symlinked {d} -> {src}")
    mc = sorted(d.glob("*.mcap"))
    if len(mc) != 1 or not (d / "metadata.yaml").exists():
        sys.exit(f"{d}: expected exactly one .mcap + metadata.yaml, found {[p.name for p in d.iterdir()]}")
    return mc[0]


def marker_edges(mcap: Path):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    with mcap.open("rb") as f:
        r = make_reader(f, decoder_factories=[DecoderFactory()]); t0 = r.get_summary().statistics.message_start_time; T, B = [], []
        for d in r.iter_decoded_messages(topics=["/xr/left_hand_inputs"]):
            T.append((d.message.log_time - t0) / 1e9); B.append(list(d.decoded_message.buttons))
    if not T:
        return {}
    T, B = np.array(T), np.array(B); out = {}
    for i in range(B.shape[1]):
        b = B[:, i]; rise = np.flatnonzero(np.diff(b) > 0) + 1; fall = np.flatnonzero(np.diff(b) < 0) + 1
        if rise.size or fall.size:
            out[f"buttons[{i}]" + (" X" if i == 2 else " Y" if i == 3 else "")] = dict(rising=np.round(T[rise], 2).tolist(), falling=np.round(T[fall], 2).tolist())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record_id"); ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args(); rid = a.record_id; mcap = find_raw(rid); env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    print(f"=== [1/3] inspect {mcap.name} ==="); subprocess.run([PY, str(REPO / "scripts/inspect_mcap.py"), str(mcap), "--out", str(REPO / "reports")], check=True, env=env)
    insp = json.loads((REPO / "reports" / f"{rid}_inspection.json").read_text())
    preview = VIDEOS / f"{rid}_preview_head_{a.fps:g}fps.mp4"
    print(f"=== [2/3] head preview -> {preview} ==="); subprocess.run([PY, str(REPO / "scripts/preview_head_stream.py"), str(mcap), "--out", str(preview), "--fps", str(a.fps), "--size", str(a.size)], check=True, env=env)
    print("=== [3/3] marker edges ==="); edges = marker_edges(mcap)
    # ---- summary ----------------------------------------------------------------------------------------------------
    tp = insp["topics"]; cams = {c: tp.get(t, {}) for c, t in {"left": "/left/color/image_raw/ffmpeg", "right": "/right/color/image_raw/ffmpeg", "head": "/xr_video_topic/ffmpeg"}.items()}
    core = ["/left_arm/joint_states", "/right_arm/joint_states", "/left_gripper/joint_states", "/right_gripper/joint_states", "/chassis/joint_states",
            "/left_arm/joint_cmd", "/right_arm/joint_cmd", "/left_gripper/joint_cmd", "/right_gripper/joint_cmd", "/chassis/joint_cmd"]
    dur = insp["duration_s"]; problems = []
    for c, d in cams.items():
        if not d or d.get("count_seen", 0) == 0: problems.append(f"camera {c}: MISSING")
        else:
            if d["last_s"] < dur - 1.0: problems.append(f"camera {c}: ends at {d['last_s']:.1f}s (< bag end {dur:.1f}s)")
            if d.get("dt_ms", {}).get("max", 0) > 100: problems.append(f"camera {c}: max inter-frame gap {d['dt_ms']['max']:.0f} ms")
    for t in core:
        d = tp.get(t, {})
        if not d or d.get("count_seen", 0) == 0: problems.append(f"{t}: MISSING")
        elif d["last_s"] < dur - 1.0: problems.append(f"{t}: ends at {d['last_s']:.1f}s")
        elif d.get("dt_ms", {}).get("max", 0) > 200: problems.append(f"{t}: max gap {d['dt_ms']['max']:.0f} ms")
    L = [f"# Phase-1 summary: {rid}", "", f"- MCAP: `{mcap}` ({insp['size_bytes']/2**20:.0f} MiB)", f"- messages span **{dur:.3f} s** ({insp['message_count']} msgs); t=0 == first message log_time",
         f"- preview: `{preview}` ({a.fps:g} fps, head/left eye, {a.size}px). Burned-in `t` = (tick log_time − first message)/1e9 s; playback time == t.",
         "- cameras: " + "; ".join(f"{c}: {d.get('count_seen',0)} pkts, {d.get('rate_hz',0) or 0:.1f} Hz, {d.get('first_s',0):.2f}–{d.get('last_s',0):.2f} s" for c, d in cams.items()),
         "- problems: " + ("; ".join(problems) if problems else "none detected"),
         "- X/Y markers: " + ("none" if not any(k.endswith(("X", "Y")) for k in edges) else "PRESENT"),
         "- button edges (reference only, NOT used): " + (json.dumps(edges) if edges else "none"), "",
         "Reply with, per segment:  start_s / end_s / prompt / success   (MCAP-relative seconds as shown in the preview)"]
    (REPO / "reports" / f"{rid}_phase1.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
