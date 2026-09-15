"""Decode one frame per camera at a given bag time and save PNGs (for deciding the camera layout)."""
import argparse, time
from pathlib import Path
import cv2, numpy as np
from dexteleop.mcap_io import read_topics, mcap_info
from dexteleop.schema import CAMERA_TOPICS
from dexteleop.video import PacketStreamDecoder

ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("--t", type=float, default=125.0); ap.add_argument("--out", required=True)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
info = mcap_info(a.mcap); t_ns = info.message_start_ns + int(a.t * 1e9)
streams = read_topics(a.mcap, set(CAMERA_TOPICS.values()))
for cam, tp in CAMERA_TOPICS.items():
    vs = streams[tp]; idx = int(np.searchsorted(vs.log_time_ns, t_ns, side="right") - 1)
    dec = PacketStreamDecoder(vs.packets, vs.encoding)
    keys = dec.keyframe_indices(); kdt = np.diff(keys)
    t0 = time.time(); fr = None
    for i, f in dec.decode_range(idx, idx):
        fr = f
    print(f"{cam}: {vs.encoding} {vs.width}x{vs.height} n={len(vs.packets)} keyframes={keys.size} "
          f"(interval median={np.median(kdt) if kdt.size else None}, first={keys[:3].tolist()}) "
          f"packet@{a.t}s idx={idx} start_key={dec._start_index(idx)} decode={time.time()-t0:.1f}s "
          f"frame={None if fr is None else fr.shape} pts_us={int(vs.pts[idx])} log_rel={(vs.log_time_ns[idx]-info.message_start_ns)/1e9:.3f}")
    if fr is not None:
        cv2.imwrite(str(out / f"{cam}_t{a.t:.0f}.png"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        small = cv2.resize(fr, (fr.shape[1] // 4, fr.shape[0] // 4), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{cam}_t{a.t:.0f}_small.png"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
