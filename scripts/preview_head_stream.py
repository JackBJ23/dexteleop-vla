"""Streaming HEAD-camera preview MP4 of a full recording, for choosing trim boundaries. O(1) memory.

- Reads the head FFMPEGPacket stream straight from the MCAP (no packet/frame accumulation), decodes packet by
  packet with PyAV, keeps only the latest decoded frame, selects the configured stereo eye, downscales, burns
  `t = X.XX s`, and pipes raw RGB frames into ffmpeg's stdin (libx264). Nothing is buffered per frame.
- Clock: MCAP log_time. Burned-in t = (tick_log_time - message_start_time)/1e9, seconds since the first message
  of the bag (t = 0.00 == rosbag2 `starting_time`). Output tick k is at t = k/fps exactly; the frame shown is the
  latest head packet with log_time <= tick (hold). Ticks before the first decodable keyframe show NO FRAME.
- Preview fps/resolution never touch the canonical data (extract.py works from the original streams).
"""
import argparse, subprocess
from pathlib import Path
import av, cv2, numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from dexteleop.mcap_io import mcap_info
from dexteleop.schema import CAMERA_TOPICS
from dexteleop.video import codec_name

ap = argparse.ArgumentParser(); ap.add_argument("mcap"); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=10.0); ap.add_argument("--size", type=int, default=640)
ap.add_argument("--eye", default="left", choices=["left", "right"]); ap.add_argument("--crf", type=int, default=23)
a = ap.parse_args(); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
info = mcap_info(a.mcap); t0 = info.message_start_ns; t_end = info.message_end_ns
period = 1e9 / a.fps; n_ticks = int(np.floor((t_end - t0) / period + 1e-9)) + 1
tick_ns = lambda k: t0 + int(round(k * period))
S = a.size; topic = CAMERA_TOPICS["head"]
import imageio_ffmpeg
ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{S}x{S+36}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-crf", str(a.crf), "-preset", "veryfast", str(out)], stdin=subprocess.PIPE)
k = 0; current = None; cur_t = None; ctx = None; n_pk = n_fr = 0; first_key_t = None; last_t = None
def emit_until(t_limit_ns):
    """Emit output ticks with tick <= t_limit using the current frame (hold)."""
    global k
    while k < n_ticks and tick_ns(k) <= t_limit_ns:
        t = (tick_ns(k) - t0) / 1e9
        if current is None:
            fr = np.zeros((S, S, 3), np.uint8); cv2.putText(fr, "NO FRAME (before first keyframe)", (10, S // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            fr = current
        canvas = np.zeros((S + 36, S, 3), np.uint8); canvas[:S] = fr
        cv2.putText(canvas, f"t = {t:7.2f} s   (MCAP log_time - bag start)   head/{a.eye} eye", (8, S + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ff.stdin.write(canvas.tobytes()); k += 1
with open(a.mcap, "rb") as f:
    reader = make_reader(f, decoder_factories=[DecoderFactory()])
    for d in reader.iter_decoded_messages(topics=[topic], log_time_order=True):
        m = d.decoded_message; t_pk = d.message.log_time; n_pk += 1; last_t = t_pk
        if ctx is None:
            ctx = av.CodecContext.create(codec_name(str(m.encoding)), "r"); ctx.thread_type = "AUTO"; ctx.thread_count = 4
        for fr in ctx.decode(av.Packet(bytes(m.data))):     # <=1 frame per packet; nothing decoded before an IRAP
            n_fr += 1
            if first_key_t is None: first_key_t = t_pk
            emit_until(t_pk - 1)                             # ticks strictly before this packet use the previous frame
            img = fr.to_ndarray(format="rgb24"); half = img.shape[1] // 2
            eye = img[:, :half] if a.eye == "left" else img[:, half:]
            current = cv2.resize(eye, (S, S), interpolation=cv2.INTER_AREA); cur_t = t_pk
            del img, eye
    for fr in ctx.decode(None):
        pass
emit_until(t_end)
ff.stdin.close(); ff.wait()
print(f"wrote {out} ({out.stat().st_size/2**20:.1f} MB): ticks={k} @ {a.fps} fps, duration={(t_end-t0)/1e9:.2f}s, "
      f"head packets={n_pk} decoded={n_fr}, first decodable frame at t={(first_key_t-t0)/1e9 if first_key_t else None}, last packet t={(last_t-t0)/1e9:.2f}s, ffmpeg rc={ff.returncode}")
