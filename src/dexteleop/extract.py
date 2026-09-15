"""MCAP -> canonical synchronized episode.

Output layout  <output_root>/<episode_id>/<variant>/
  episode.npz        grid timestamps, proprio[K,19], action[K,19], aux[K,4], per-topic source index + age,
                     per-camera source packet index / log_time / pts / age
  meta.json          schema, dim names, config, MCAP identity, per-stream statistics, blockers
  frames/<cam>/<k>.jpg   one decoded frame per grid row per camera (selected eye, optional resize)
  frames_index.csv   row, cam, source packet index, source log_time_ns, source pts_us, age_s
  validation.md, qc/  human-readable report + plots + preview video (report.py)
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import schema as S
from .mcap_io import JointStateStream, VideoStream, mcap_info, read_topics
from .sync import Aligned, align, gap_report, gather, make_grid
from .video import PacketStreamDecoder

TOPIC_KEYS = {  # short keys used in npz / meta
    S.T_LEFT_ARM_STATE: "left_arm_state", S.T_RIGHT_ARM_STATE: "right_arm_state",
    S.T_LEFT_GRIP_STATE: "left_grip_state", S.T_RIGHT_GRIP_STATE: "right_grip_state", S.T_BASE_STATE: "base_state",
    S.T_LEFT_ARM_CMD: "left_arm_cmd", S.T_RIGHT_ARM_CMD: "right_arm_cmd",
    S.T_LEFT_GRIP_CMD: "left_grip_cmd", S.T_RIGHT_GRIP_CMD: "right_grip_cmd", S.T_BASE_CMD: "base_cmd",
}


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for k in ("episode_id", "mcap", "trim", "rate_hz", "align", "tolerances", "cameras", "output_root"):
        if k not in cfg:
            raise KeyError(f"config missing required key {k!r}")
    if "prompt" not in cfg:
        raise KeyError("config must contain `prompt` (may be null, but must be present)")
    return cfg


def variant_name(cfg: dict) -> str:
    return f"t{cfg['trim']['start_s']:g}-{cfg['trim']['end_s']:g}_hz{cfg['rate_hz']:g}"


def check_joint_names(streams: dict) -> None:
    exp = {S.T_LEFT_ARM_STATE: S.ARM_JOINT_NAMES["left"], S.T_RIGHT_ARM_STATE: S.ARM_JOINT_NAMES["right"],
           S.T_LEFT_ARM_CMD: S.ARM_JOINT_NAMES["left"], S.T_RIGHT_ARM_CMD: S.ARM_JOINT_NAMES["right"],
           S.T_LEFT_GRIP_STATE: [S.GRIPPER_JOINT_NAME["left"]], S.T_LEFT_GRIP_CMD: [S.GRIPPER_JOINT_NAME["left"]],
           S.T_RIGHT_GRIP_STATE: [S.GRIPPER_JOINT_NAME["right"]], S.T_RIGHT_GRIP_CMD: [S.GRIPPER_JOINT_NAME["right"]],
           S.T_BASE_STATE: S.BASE_STATE_NAMES, S.T_BASE_CMD: S.BASE_CMD_NAMES}
    for tp, names in exp.items():
        got = streams[tp].names
        if got != names:
            raise ValueError(f"{tp}: joint names {got} != expected {names}; refusing to index by position")


def assemble(dims: tuple[S.Dim, ...], streams: dict[str, JointStateStream], grid_ns: np.ndarray, mode: str
             ) -> tuple[np.ndarray, dict[str, Aligned]]:
    """Build [K, len(dims)] by aligning each source topic once and gathering the requested fields."""
    out = np.full((grid_ns.size, len(dims)), np.nan)
    aligned: dict[str, Aligned] = {}
    for tp, items in S.dims_by_topic(dims).items():
        st = streams[tp]
        al = aligned.setdefault(tp, align(st.log_time_ns, grid_ns, mode))
        for j, d in items:
            out[:, j] = gather(st.field(d.field)[:, d.index], al)
    return out, aligned


def select_eye(frame: np.ndarray, eye: str) -> np.ndarray:
    h, w = frame.shape[:2]
    if eye == "both":
        return frame
    half = w // 2
    return frame[:, :half] if eye == "left" else frame[:, half:]


def extract(cfg: dict, repo_root: Path, write_frames: bool = True, progress: bool = True) -> Path:
    t_start = time.time()
    mcap_path = (repo_root / cfg["mcap"]).resolve()
    info = mcap_info(mcap_path)
    t0 = info.message_start_ns
    start_ns = t0 + int(round(cfg["trim"]["start_s"] * 1e9))
    end_ns = t0 + int(round(cfg["trim"]["end_s"] * 1e9))
    if not (t0 <= start_ns < end_ns <= info.message_end_ns):
        raise ValueError("trim outside MCAP time span")
    grid = make_grid(start_ns, end_ns, float(cfg["rate_hz"]))
    K = grid.size

    streams = read_topics(mcap_path, S.topics_needed(), progress=progress)
    check_joint_names(streams)

    proprio, al_p = assemble(S.PROPRIO_DIMS, streams, grid, cfg["align"]["proprio"])
    action, al_a = assemble(S.ACTION_DIMS, streams, grid, cfg["align"]["action"])
    aux, _ = assemble(S.AUX_DIMS, streams, grid, cfg["align"]["proprio"])

    # ---- cameras -----------------------------------------------------------------------------------
    cam_al: dict[str, Aligned] = {}
    for cam, tp in S.CAMERA_TOPICS.items():
        cam_al[cam] = align(streams[tp].log_time_ns, grid, cfg["align"]["camera"])

    # ---- tolerance / dropout accounting -----------------------------------------------------------------
    tol = cfg["tolerances"]
    viol: dict[str, dict] = {}
    for tp, al in al_p.items():
        bad = np.isnan(al.age_s) | (al.age_s > tol["proprio_max_age_s"])
        viol[TOPIC_KEYS[tp]] = dict(kind="proprio", n_missing=int(np.isnan(al.age_s).sum()), n_stale=int(bad.sum()),
                                    max_age_s=float(np.nanmax(al.age_s)))
    for tp, al in al_a.items():
        bad = np.isnan(al.age_s) | (np.abs(al.age_s) > tol["action_max_lead_s"])
        viol[TOPIC_KEYS[tp]] = dict(kind="action", n_missing=int(np.isnan(al.age_s).sum()), n_stale=int(bad.sum()),
                                    max_abs_age_s=float(np.nanmax(np.abs(al.age_s))))
    for cam, al in cam_al.items():
        bad = np.isnan(al.age_s) | (al.age_s > tol["camera_max_age_s"])
        idx = al.index
        reused = int((np.diff(idx[idx >= 0]) == 0).sum())
        viol[f"cam_{cam}"] = dict(kind="camera", n_missing=int(np.isnan(al.age_s).sum()), n_stale=int(bad.sum()),
                                  max_age_s=float(np.nanmax(al.age_s)), n_reused_frames=reused)
    n_viol = sum(v["n_stale"] for v in viol.values())
    if n_viol and tol.get("fail_on_violation", False):
        raise RuntimeError(f"{n_viol} tolerance violations: {viol}")

    # in-window source-stream gap reports (dropouts INSIDE the trim, on the raw streams)
    win = {}
    for tp, st in streams.items():
        m = (st.log_time_ns >= start_ns - int(1e9)) & (st.log_time_ns <= end_ns + int(1e9))
        win[tp] = gap_report(st.log_time_ns[m])
        win[tp]["first_rel_s"] = float((st.log_time_ns[0] - t0) / 1e9)
        win[tp]["last_rel_s"] = float((st.log_time_ns[-1] - t0) / 1e9)

    # ---- output ----------------------------------------------------------------------------------------
    out = Path(cfg["output_root"]) / cfg["episode_id"] / variant_name(cfg)
    out.mkdir(parents=True, exist_ok=True)
    npz = dict(
        t_ns=grid, t_rel_s=(grid - t0) / 1e9, proprio=proprio.astype(np.float32), action=action.astype(np.float32),
        aux=aux.astype(np.float32), proprio_f64=proprio, action_f64=action,
    )
    for tp, al in {**al_p, **al_a}.items():
        k = TOPIC_KEYS[tp]
        npz[f"src_index/{k}"] = al.index; npz[f"src_age_s/{k}"] = al.age_s
        npz[f"src_log_time_ns/{k}"] = np.where(al.index >= 0, streams[tp].log_time_ns[np.clip(al.index, 0, None)], -1)
    for cam, al in cam_al.items():
        vs: VideoStream = streams[S.CAMERA_TOPICS[cam]]
        ok = al.index >= 0
        npz[f"frame_src_index/{cam}"] = al.index; npz[f"frame_age_s/{cam}"] = al.age_s
        npz[f"frame_src_log_time_ns/{cam}"] = np.where(ok, vs.log_time_ns[np.clip(al.index, 0, None)], -1)
        npz[f"frame_src_pts_us/{cam}"] = np.where(ok, vs.pts[np.clip(al.index, 0, None)].astype(np.int64), -1)
        npz[f"frame_src_header_stamp_ns/{cam}"] = np.where(ok, vs.header_stamp_ns[np.clip(al.index, 0, None)], -1)
    np.savez_compressed(out / "episode.npz", **npz)

    # ---- frames ---------------------------------------------------------------------------------------------
    frame_stats = {}
    if write_frames:
        rows = []
        for cam, tp in S.CAMERA_TOPICS.items():
            vs = streams[tp]; al = cam_al[cam]; ccfg = cfg["cameras"][cam]
            d = out / "frames" / cam; d.mkdir(parents=True, exist_ok=True)
            need = al.index[al.index >= 0]
            first, last = int(need.min()), int(need.max())
            want: dict[int, list[int]] = {}
            for k, i in enumerate(al.index):
                if i >= 0:
                    want.setdefault(int(i), []).append(k)
            dec = PacketStreamDecoder(vs.packets, vs.encoding)
            n_written, shape = 0, None
            tt = time.time()
            for i, fr in dec.decode_range(first, last):
                if i not in want:
                    continue
                img = select_eye(fr, ccfg["eye"])
                if ccfg.get("resize"):
                    img = cv2.resize(img, tuple(ccfg["resize"]), interpolation=cv2.INTER_AREA)
                shape = img.shape
                enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(cfg.get("jpeg_quality", 95))])[1].tobytes()
                for k in want[i]:
                    (d / f"{k:06d}.jpg").write_bytes(enc); n_written += 1
                    rows.append((k, cam, i, int(vs.log_time_ns[i]), int(vs.pts[i]), float(al.age_s[k])))
            missing_rows = [k for k in range(K) if not (d / f"{k:06d}.jpg").exists()]
            frame_stats[cam] = dict(n_written=n_written, n_rows_missing=len(missing_rows), shape=list(shape) if shape else None,
                                    packets_undecoded=getattr(dec, "undecoded", []), decode_s=time.time() - tt,
                                    src_resolution=[vs.width, vs.height], eye=ccfg["eye"], resize=ccfg.get("resize"),
                                    encoding=vs.encoding, keyframes_in_stream=int(dec.keyframe_indices().size))
            if missing_rows:
                raise RuntimeError(f"{cam}: {len(missing_rows)} grid rows have no decoded frame, e.g. {missing_rows[:5]}")
        rows.sort()
        with (out / "frames_index.csv").open("w") as f:
            f.write("row,cam,src_packet_index,src_log_time_ns,src_pts_us,age_s\n")
            for r in rows:
                f.write(",".join(map(str, r)) + "\n")

    # ---- meta --------------------------------------------------------------------------------------------------
    blockers = []
    if not cfg.get("prompt"):
        blockers.append("prompt is null: a natural-language task prompt is REQUIRED before any policy training")
    if n_viol:
        blockers.append(f"{n_viol} alignment tolerance violations (see stream_violations)")
    meta = dict(
        schema_version=S.SCHEMA_VERSION, episode_id=cfg["episode_id"], variant=variant_name(cfg), prompt=cfg.get("prompt"),
        config=cfg, mcap=dataclasses.asdict(info) | {"topic_counts": {t: c for t, c in info.topic_counts.items() if t in S.topics_needed()}},
        time_base="MCAP log_time (ns since epoch); t_rel_s = (log_time - message_start_ns)/1e9; header.stamp NOT used for alignment",
        grid=dict(K=K, rate_hz=cfg["rate_hz"], start_ns=int(start_ns), end_ns=int(end_ns), start_rel_s=cfg["trim"]["start_s"],
                  end_rel_s=cfg["trim"]["end_s"], duration_s=float((grid[-1] - grid[0]) / 1e9)),
        proprio_names=[d.name for d in S.PROPRIO_DIMS], action_names=[d.name for d in S.ACTION_DIMS],
        aux_names=[d.name for d in S.AUX_DIMS],
        proprio_sources=[dataclasses.asdict(d) for d in S.PROPRIO_DIMS], action_sources=[dataclasses.asdict(d) for d in S.ACTION_DIMS],
        align=cfg["align"], tolerances=tol, stream_violations=viol, source_stream_windows=win,
        frames=frame_stats, cameras={c: {"topic": t, **cfg["cameras"][c]} for c, t in S.CAMERA_TOPICS.items()},
        stats=dict(proprio_min=np.nanmin(proprio, 0).tolist(), proprio_max=np.nanmax(proprio, 0).tolist(),
                   proprio_mean=np.nanmean(proprio, 0).tolist(), proprio_std=np.nanstd(proprio, 0).tolist(),
                   action_min=np.nanmin(action, 0).tolist(), action_max=np.nanmax(action, 0).tolist(),
                   action_mean=np.nanmean(action, 0).tolist(), action_std=np.nanstd(action, 0).tolist(),
                   n_nan_proprio=int(np.isnan(proprio).sum()), n_nan_action=int(np.isnan(action).sum())),
        blockers=blockers, extract_seconds=time.time() - t_start,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out
