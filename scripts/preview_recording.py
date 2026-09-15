"""Full-recording 3-camera preview MP4 from the ORIGINAL MCAP streams, for choosing trim boundaries.

Clock: MCAP log_time. Burned-in `t = X.XX s` is (grid_tick_log_time - message_start_time) / 1e9, i.e. seconds
since the first message in the bag (== rosbag2 metadata `starting_time`). This is exactly the `start_s/end_s`
axis used by configs/episodes/*.yaml. Frame shown at tick t is the latest packet with log_time <= t (hold).
Gripper commands + base cmd (from joint_cmd topics, hold-aligned) are overlaid to help find task boundaries.
"""
import argparse, json
from pathlib import Path
import av, cv2, numpy as np
from dexteleop import schema as S
from dexteleop.mcap_io import mcap_info, read_topics
from dexteleop.sync import align, gather, make_grid
from dexteleop.video import PacketStreamDecoder

ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--tile", type=int, default=480)
ap.add_argument("--tmp", default=None, help="tile scratch dir (default: <out>_tiles); tiles are streamed to disk to keep RAM low")
a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
tmp = Path(a.tmp or (str(out) + "_tiles")); tmp.mkdir(parents=True, exist_ok=True)
info = mcap_info(a.mcap); t0 = info.message_start_ns
grid = make_grid(t0, info.message_end_ns, a.fps); K = grid.size; T = a.tile
streams = read_topics(a.mcap, set(S.CAMERA_TOPICS.values()) | {S.T_LEFT_GRIP_CMD, S.T_RIGHT_GRIP_CMD, S.T_BASE_CMD}, progress=False)
gl = gather(streams[S.T_LEFT_GRIP_CMD].effort[:, 0:1], align(streams[S.T_LEFT_GRIP_CMD].log_time_ns, grid, "hold"))[:, 0]
gr = gather(streams[S.T_RIGHT_GRIP_CMD].effort[:, 0:1], align(streams[S.T_RIGHT_GRIP_CMD].log_time_ns, grid, "hold"))[:, 0]
bv = gather(streams[S.T_BASE_CMD].velocity, align(streams[S.T_BASE_CMD].log_time_ns, grid, "hold"))
# decode every camera once, keep only needed packets, downscaled tiles (left eye of the stereo pair)
tiles = {}
for cam, tp in S.CAMERA_TOPICS.items():
    vs = streams[tp]; al = align(vs.log_time_ns, grid, "hold"); idx = al.index
    want = {}
    for k, i in enumerate(idx):
        if i >= 0: want.setdefault(int(i), []).append(k)
    have = [False] * K; d = tmp / cam; d.mkdir(exist_ok=True)
    if want:
        dec = PacketStreamDecoder(vs.packets, vs.encoding)
        first_key = int(dec.keyframe_indices()[0])          # packets before the first IRAP are undecodable -> NO FRAME
        for i, fr in dec.decode_range(max(min(want), first_key), max(want)):
            if i in want:
                eye = fr[:, : fr.shape[1] // 2]; sm = cv2.resize(eye, (T, T), interpolation=cv2.INTER_AREA)
                enc = cv2.imencode(".jpg", cv2.cvtColor(sm, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                for k in want[i]: (d / f"{k:06d}.jpg").write_bytes(enc); have[k] = True
    del streams[tp].packets[:]                              # free the HEVC packets of this camera
    tiles[cam] = (have, al.age_s, d)
    print(f"{cam}: packets={len(vs.log_time_ns)} span={(vs.log_time_ns[0]-t0)/1e9:.2f}-{(vs.log_time_ns[-1]-t0)/1e9:.2f}s ticks_with_frame={sum(have)}/{K} first_keyframe_packet={first_key if want else None}")
cont = av.open(str(out), "w"); st = cont.add_stream("libx264", rate=int(round(a.fps))); st.width, st.height = 3 * T, T + 40; st.pix_fmt = "yuv420p"; st.options = {"crf": "22"}
for k in range(K):
    row = []
    for cam in ("head", "left", "right"):
        have, age, d = tiles[cam]
        im = cv2.cvtColor(cv2.imread(str(d / f"{k:06d}.jpg")), cv2.COLOR_BGR2RGB) if have[k] else np.zeros((T, T, 3), np.uint8)
        lab = f"{cam}" + ("" if have[k] else "  NO FRAME") + (f"  age {age[k]*1e3:.0f}ms" if np.isfinite(age[k]) else "")
        cv2.putText(im, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); row.append(im)
    fr = np.vstack([np.hstack(row), np.zeros((40, 3 * T, 3), np.uint8)])
    t = (grid[k] - t0) / 1e9
    cv2.putText(fr, f"t = {t:7.2f} s   (MCAP log_time - bag start)   gripL={gl[k]:+.2f} gripR={gr[k]:+.2f} base={np.round(bv[k],2)}", (8, T + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")): cont.mux(p)
for p in st.encode(): cont.mux(p)
cont.close()
import shutil; shutil.rmtree(tmp, ignore_errors=True)
print(f"wrote {out} K={K} fps={a.fps} duration={(grid[-1]-t0)/1e9:.2f}s size={out.stat().st_size/2**20:.1f}MB")
