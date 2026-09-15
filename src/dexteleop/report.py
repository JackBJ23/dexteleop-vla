"""Validation report + visual QC for a canonical episode directory produced by extract.py."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import schema as S


def _q(a, ps=(50, 95, 99, 100)):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return {f"p{p}": float(np.percentile(a, p)) for p in ps} if a.size else {}


def _du(path: Path) -> str:
    try:
        return subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "?"


def build_report(ep_dir: Path, legacy_npz: Path | None = None) -> Path:
    ep_dir = Path(ep_dir)
    meta = json.loads((ep_dir / "meta.json").read_text())
    z = np.load(ep_dir / "episode.npz")
    qc = ep_dir / "qc"; qc.mkdir(exist_ok=True)
    t = z["t_rel_s"]; P = z["proprio_f64"]; A = z["action_f64"]; X = z["aux"]
    K = t.size
    L = [f"# Validation report: {meta['episode_id']} / {meta['variant']}", "",
         f"- schema: `{meta['schema_version']}`  | grid: K={K}, {meta['grid']['rate_hz']} Hz, "
         f"{meta['grid']['start_rel_s']}–{meta['grid']['end_rel_s']} s (duration {meta['grid']['duration_s']:.3f} s)",
         f"- MCAP: `{meta['mcap']['path']}` ({meta['mcap']['size_bytes']/2**20:.1f} MiB, id `{meta['mcap']['sha256_head_1mb'][:16]}`)",
         f"- time base: {meta['time_base']}", f"- alignment: {meta['align']}", f"- prompt: **{meta['prompt']!r}**", ""]
    if meta["blockers"]:
        L += ["## BLOCKERS", ""] + [f"- **{b}**" for b in meta["blockers"]] + [""]

    # ---- alignment / staleness ------------------------------------------------------------------------------
    L += ["## Alignment statistics per source stream (age = t_grid − t_source, s)", "",
          "| stream | kind | n_missing | n_out_of_tol | age p50 | p95 | p99 | max | notes |", "|---|---|---|---|---|---|---|---|---|"]
    for key, v in meta["stream_violations"].items():
        arr = z[f"src_age_s/{key}"] if key in [kk for kk in meta["stream_violations"] if not kk.startswith("cam_")] and f"src_age_s/{key}" in z else z[f"frame_age_s/{key[4:]}"]
        q = _q(np.abs(arr))
        note = f"reused frames={v['n_reused_frames']}" if "n_reused_frames" in v else ""
        L.append(f"| {key} | {v['kind']} | {v['n_missing']} | {v['n_stale']} | {q.get('p50',0)*1e3:.1f} ms | {q.get('p95',0)*1e3:.1f} ms | "
                 f"{q.get('p99',0)*1e3:.1f} ms | {q.get('p100',0)*1e3:.1f} ms | {note} |")
    L += ["", "## Raw source-stream gaps inside the trim (±1 s)", "",
          "| topic | first s | last s | n | rate Hz | dt median | dt p99 | dt max | n gaps >2× | largest gaps (t_rel, dt) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for tp, g in meta["source_stream_windows"].items():
        if g.get("n", 0) < 2:
            L.append(f"| `{tp}` | | | {g.get('n')} | | | | | | |"); continue
        gaps = ", ".join(f"({meta['grid']['start_rel_s']-1+a:.2f}s,{b*1e3:.0f}ms)" for a, b in g["gaps"][:4])
        L.append(f"| `{tp}` | {g['first_rel_s']:.1f} | {g['last_rel_s']:.1f} | {g['n']} | {g['rate_hz']:.1f} | {g['dt_median_s']*1e3:.1f} ms | "
                 f"{g['dt_p99_s']*1e3:.1f} ms | {g['dt_max_s']*1e3:.1f} ms | {g['n_gaps']} | {gaps} |")

    # ---- camera frame age -----------------------------------------------------------------------------------------
    L += ["", "## Camera frame age (hold alignment; log_time based) + sender pts cross-check", ""]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
    for cam in S.CAMERA_TOPICS:
        age = z[f"frame_age_s/{cam}"] * 1e3
        pts = z[f"frame_src_pts_us/{cam}"].astype(np.float64) / 1e6
        lt = (z[f"frame_src_log_time_ns/{cam}"] - meta["mcap"]["message_start_ns"]) / 1e9
        d = (lt - pts) * 1e3; d = d[np.isfinite(d)]
        fs = meta["frames"].get(cam, {})
        L.append(f"- **{cam}**: age p50/p95/max = {np.nanpercentile(age,50):.1f}/{np.nanpercentile(age,95):.1f}/{np.nanmax(age):.1f} ms; "
                 f"reused frames = {meta['stream_violations']['cam_'+cam]['n_reused_frames']}; missing = {meta['stream_violations']['cam_'+cam]['n_missing']}; "
                 f"log_time − pts offset: mean {d.mean():.1f} ms, std {d.std():.1f} ms, range [{d.min():.1f}, {d.max():.1f}] ms; "
                 f"frames written {fs.get('n_written')} shape {fs.get('shape')} (src {fs.get('src_resolution')}, eye {fs.get('eye')}, resize {fs.get('resize')}), undecoded packets {fs.get('packets_undecoded')}")
        ax[0].hist(age[np.isfinite(age)], bins=40, alpha=0.6, label=cam); ax[1].plot(t, age, lw=0.6, label=cam)
    ax[0].set_xlabel("frame age (ms)"); ax[0].legend(); ax[1].set_xlabel("t_rel (s)"); ax[1].set_ylabel("frame age (ms)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(qc / "frame_age.png", dpi=110); plt.close(fig)

    # ---- data stats ------------------------------------------------------------------------------------------------
    L += ["", "## Proprio (19-D, measured) and action (19-D, commanded) statistics", "",
          "| i | proprio | min | max | std | nan | action | min | max | std | nan |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i in range(19):
        L.append(f"| {i} | {meta['proprio_names'][i]} | {np.nanmin(P[:,i]):.4f} | {np.nanmax(P[:,i]):.4f} | {np.nanstd(P[:,i]):.4f} | {int(np.isnan(P[:,i]).sum())} "
                 f"| {meta['action_names'][i]} | {np.nanmin(A[:,i]):.4f} | {np.nanmax(A[:,i]):.4f} | {np.nanstd(A[:,i]):.4f} | {int(np.isnan(A[:,i]).sum())} |")
    L += ["", "aux (not in canonical proprio): " + ", ".join(f"{n}: [{np.nanmin(X[:,i]):.3f}, {np.nanmax(X[:,i]):.3f}] std {np.nanstd(X[:,i]):.3f}"
                                                            for i, n in enumerate(meta["aux_names"]))]

    # ---- gripper semantics evidence ------------------------------------------------------------------------------------
    L += ["", "## Gripper semantics evidence (measured joint_states vs commanded joint_cmd.effort)", ""]
    fig, axs = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for r, (side, gi, ai) in enumerate((("left", 14, 14), ("right", 15, 15))):
        pos, cmd = P[:, gi], A[:, ai]
        vel, eff = X[:, 2 + r], X[:, r]
        ok = np.isfinite(pos) & np.isfinite(cmd)
        c_pc = np.corrcoef(pos[ok], cmd[ok])[0, 1]
        # lagged correlation: does position follow the command? best lag in grid steps
        lags = range(0, int(meta["grid"]["rate_hz"] * 1.0) + 1); best = max(lags, key=lambda l: np.corrcoef(cmd[:K-l][ok[:K-l] & ok[l:]], pos[l:][ok[:K-l] & ok[l:]])[0, 1] if l else c_pc)
        c_best = np.corrcoef(cmd[:K-best][ok[:K-best] & ok[best:]], pos[best:][ok[:K-best] & ok[best:]])[0, 1] if best else c_pc
        i_min = int(np.nanargmin(cmd)); i_pmin = int(np.nanargmin(pos))
        L.append(f"- **{side}**: corr(cmd, pos) same-time = {c_pc:+.3f}; best lag = {best/meta['grid']['rate_hz']:.2f} s (corr {c_best:+.3f}). "
                 f"cmd min {np.nanmin(cmd):.3f} @ t={t[i_min]:.2f}s; pos min {np.nanmin(pos):.3f} @ t={t[i_pmin]:.2f}s; "
                 f"pos when cmd==+2.0: mean {np.nanmean(pos[cmd>=1.99]):.3f}; pos when cmd<0: mean {np.nanmean(pos[cmd<0]) if (cmd<0).any() else float('nan'):.3f}; "
                 f"measured effort when cmd<0: mean {np.nanmean(eff[cmd<0]) if (cmd<0).any() else float('nan'):.3f} vs cmd>=1.99: {np.nanmean(eff[cmd>=1.99]):.3f}")
        ax = axs[r]; ax.plot(t, cmd, label="joint_cmd.effort (command)", lw=1.2); ax.plot(t, pos, label="joint_states.position", lw=1.2)
        ax.plot(t, eff, label="joint_states.effort (measured)", lw=0.7, alpha=0.7); ax.plot(t, vel, label="joint_states.velocity", lw=0.5, alpha=0.5)
        ax.set_ylabel(f"{side} gripper"); ax.legend(loc="lower left", fontsize=7); ax.grid(alpha=0.3)
    axs[-1].set_xlabel("t_rel (s)"); fig.tight_layout(); fig.savefig(qc / "grippers.png", dpi=110); plt.close(fig)

    # ---- events -------------------------------------------------------------------------------------------------
    bv = A[:, 16:19]; bm = P[:, 16:19]
    fwd = bv.mean(1)  # crude: mean motor velocity (sign convention unknown) -- report extremes only
    ts0 = meta["grid"]["start_rel_s"]
    def _ev(tt): return f"t={tt:.2f} s MCAP (= {tt - ts0:.2f} s after trim start)"
    L += ["", "## Event check (canonical data; MCAP log_time, and seconds after trim start)", "",
          f"- right gripper cmd minimum {np.nanmin(A[:,15]):.2f} at {_ev(t[int(np.nanargmin(A[:,15]))])}",
          f"- left gripper cmd minimum {np.nanmin(A[:,14]):.2f} at {_ev(t[int(np.nanargmin(A[:,14]))])}",
          f"- base cmd |v| max {np.nanmax(np.abs(bv)):.2f} at {_ev(t[int(np.nanargmax(np.abs(bv).max(1)))])}",
          f"- base activity (|v|>0.2 on any motor) intervals, MCAP s: {_intervals(t, (np.abs(bv) > 0.2).any(1))}",
          f"- corr(base cmd, base measured) per motor: {[f'{np.corrcoef(bv[:,i][np.isfinite(bm[:,i])], bm[:,i][np.isfinite(bm[:,i])])[0,1]:+.3f}' for i in range(3)]}"]

    # ---- arm / base plots --------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(7, 2, figsize=(14, 14), sharex=True)
    for j in range(7):
        for c, (side, off) in enumerate((("left", 0), ("right", 7))):
            ax = axs[j, c]; ax.plot(t, A[:, off + j], lw=0.9, label="cmd"); ax.plot(t, P[:, off + j], lw=0.9, label="measured")
            ax.set_ylabel(f"{side} j{j+1}", fontsize=8); ax.grid(alpha=0.3)
            if j == 0: ax.legend(fontsize=7)
    axs[-1, 0].set_xlabel("t_rel (s)"); axs[-1, 1].set_xlabel("t_rel (s)"); fig.tight_layout(); fig.savefig(qc / "arms.png", dpi=100); plt.close(fig)
    fig, axs = plt.subplots(3, 1, figsize=(13, 6), sharex=True)
    for i in range(3):
        axs[i].plot(t, bv[:, i], label="cmd vel"); axs[i].plot(t, bm[:, i], label="measured vel", alpha=0.8); axs[i].set_ylabel(f"base m{i+1}"); axs[i].grid(alpha=0.3); axs[i].legend(fontsize=7)
    axs[-1].set_xlabel("t_rel (s)"); fig.tight_layout(); fig.savefig(qc / "base.png", dpi=110); plt.close(fig)
    # tracking error cmd vs measured (arms)
    err = np.abs(A[:, :14] - P[:, :14])
    L += ["", f"- arm |cmd − measured| joint error: mean {np.nanmean(err):.4f} rad, p95 {np.nanpercentile(err,95):.4f}, max {np.nanmax(err):.4f} "
          f"(per-joint mean: {np.round(np.nanmean(err,0),3).tolist()})"]

    # ---- montage + preview video ---------------------------------------------------------------------------------------
    picks = np.linspace(0, K - 1, 8).astype(int)
    tiles = []
    for k in picks:
        col = []
        for cam in S.CAMERA_TOPICS:
            im = cv2.imread(str(ep_dir / "frames" / cam / f"{k:06d}.jpg"))
            im = cv2.resize(im, (320, 200)) if cam != "head" else cv2.resize(im, (320, 320))
            cv2.putText(im, f"{cam} t={t[k]:.1f}s", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            col.append(im)
        tiles.append(np.vstack(col))
    cv2.imwrite(str(qc / "montage.png"), np.hstack(tiles))
    _preview_video(ep_dir, qc / "preview.mp4", t, A, meta)
    L += ["", "## Visual QC", "", f"- `{qc}/montage.png`, `frame_age.png`, `grippers.png`, `arms.png`, `base.png`, `preview.mp4`"]

    # ---- legacy cross-check -----------------------------------------------------------------------------------------------
    if legacy_npz is not None and Path(legacy_npz).exists():
        L += ["", "## Cross-check vs legacy Mac-side npz (regression only; NOT source of truth)", ""] + legacy_crosscheck(ep_dir, Path(legacy_npz), meta, z)

    L += ["", "## Storage", "", f"- `{ep_dir}`: {_du(ep_dir)} (frames: {_du(ep_dir/'frames')}, episode.npz: {(ep_dir/'episode.npz').stat().st_size/2**20:.1f} MiB)"]
    (ep_dir / "validation.md").write_text("\n".join(L))
    return ep_dir / "validation.md"


def _intervals(t, mask, min_len_s=0.3):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if (not m or i == len(mask) - 1) and start is not None:
            if t[i] - t[start] >= min_len_s: out.append((round(float(t[start]), 1), round(float(t[i]), 1)))
            start = None
    return out


def _preview_video(ep_dir: Path, out: Path, t, A, meta, fps_cap=15.0):
    import av
    K = t.size; step = max(1, int(round(meta["grid"]["rate_hz"] / fps_cap)))
    fps = meta["grid"]["rate_hz"] / step
    codec = "libx264" if any(c == "libx264" for c in av.codecs_available) else "mpeg4"
    cont = av.open(str(out), "w"); st = cont.add_stream(codec, rate=int(round(fps))); st.pix_fmt = "yuv420p"
    W = 960; st.width, st.height = W, 320
    for k in range(0, K, step):
        tiles = []
        for cam in S.CAMERA_TOPICS:
            im = cv2.imread(str(ep_dir / "frames" / cam / f"{k:06d}.jpg")); im = cv2.resize(im, (320, 320))
            tiles.append(im)
        fr = np.hstack(tiles)
        cv2.putText(fr, f"t={t[k]:7.2f}s  gripL={A[k,14]:+.2f} gripR={A[k,15]:+.2f} base={np.round(A[k,16:19],2)}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        vf = av.VideoFrame.from_ndarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), format="rgb24")
        for pkt in st.encode(vf): cont.mux(pkt)
    for pkt in st.encode(): cont.mux(pkt)
    cont.close()


def legacy_crosscheck(ep_dir, legacy_npz, meta, z):
    """Compare our hold-aligned proprio/action against the legacy npz at ITS bag timestamps."""
    from .mcap_io import read_topics
    from .extract import assemble
    lz = np.load(legacy_npz, allow_pickle=True)
    t0 = meta["mcap"]["message_start_ns"]
    ts = lz["state_timestamp_bag_s"]; ta = lz["action_timestamp_bag_s"]
    lo, hi = meta["grid"]["start_rel_s"], meta["grid"]["end_rel_s"]
    m = (ts >= lo) & (ts <= hi)
    grid_s = (t0 + np.round(ts[m] * 1e9)).astype(np.int64); grid_a = (t0 + np.round(ta[m] * 1e9)).astype(np.int64)
    streams = read_topics(meta["mcap"]["path"], {d.topic for d in S.PROPRIO_DIMS + S.ACTION_DIMS}, progress=False)
    P, _ = assemble(S.PROPRIO_DIMS, streams, grid_s, "hold"); A, _ = assemble(S.ACTION_DIMS, streams, grid_a, "hold")
    LP = lz["observation_state_72d"][m][:, S.LEGACY72_PROPRIO_INDEX]; LA = lz["action_72d"][m][:, S.LEGACY72_ACTION_INDEX]
    out = [f"- legacy rows in window: {m.sum()} (legacy 'sync start' offset means its t=125.0 is bag {ts[m][0]:.4f} s)",
           f"- legacy timestamp columns: state_timestamp_bag_s first={ts[m][0]:.4f} last={ts[m][-1]:.4f}; action − state = {np.median(ta[m]-ts[m])*1e3:.2f} ms"]
    for name, ours, legacy in (("proprio", P, LP), ("action", A, LA)):
        d = np.abs(ours - legacy); ok = np.isfinite(d)
        exact = (d[ok] < 1e-6).mean()
        worst = np.nanmax(d, 0)
        out.append(f"- **{name}**: fraction of entries equal (<1e-6): {exact:.4f}; per-dim max |diff|: {np.round(worst, 4).tolist()}")
    return out
