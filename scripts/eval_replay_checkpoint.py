"""Open-loop replay check: feed a trained pi0.5 checkpoint the canonical episode's own observations and compare
the predicted action chunks with the recorded actions. Exercises the full inference path (DexTeleopInputs ->
model -> Unnormalize -> DexTeleopOutputs). Run under openpi's `uv run` on a GPU node.

    uv run python <repo>/scripts/eval_replay_checkpoint.py --config pi05_dexteleop_overfit1 --ckpt <dir>/500 \
        --canonical <canonical episode dir> --out <dir> [--stride 5]
"""
import argparse, json, time
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--canonical", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--max-rows", type=int, default=0)
    a = ap.parse_args()
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    can = Path(a.canonical); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((can / "meta.json").read_text()); z = np.load(can / "episode.npz")
    A, P, t = z["action"].astype(np.float32), z["proprio"].astype(np.float32), z["t_rel_s"]
    prompt = meta["prompt"]; K = A.shape[0]
    cfg = _config.get_config(a.config)
    H = cfg.model.action_horizon
    t0 = time.time(); policy = _pc.create_trained_policy(cfg, a.ckpt); print(f"policy loaded in {time.time()-t0:.1f}s; horizon {H}")

    def img(cam, k):
        im = cv2.imread(str(can / "frames" / cam / f"{k:06d}.jpg")); return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    rows = list(range(0, K - H, a.stride))
    if a.max_rows: rows = rows[: a.max_rows]
    pred = np.zeros((len(rows), H, A.shape[1]), np.float32); gt = np.zeros_like(pred); lat = []
    for i, k in enumerate(rows):
        obs = {"observation/head_image": img("head", k), "observation/left_image": img("left", k),
               "observation/right_image": img("right", k), "observation/state": P[k], "prompt": prompt}
        t1 = time.time(); r = policy.infer(obs); lat.append(time.time() - t1)
        pred[i] = np.asarray(r["actions"])[:H, : A.shape[1]]; gt[i] = A[k:k + H]
        if i % 20 == 0: print(f"  row {k}/{K}  infer {lat[-1]*1e3:.0f} ms")
    err = np.abs(pred - gt); std = A.std(0) + 1e-6
    names = meta["action_names"]
    rep = dict(config=a.config, ckpt=a.ckpt, canonical=str(can), prompt=prompt, n_rows=len(rows), stride=a.stride, horizon=H,
               infer_ms_first=lat[0] * 1e3, infer_ms_median=float(np.median(lat[1:]) * 1e3),
               mae_per_dim=dict(zip(names, err.mean((0, 1)).round(4).tolist())),
               mae_over_std_per_dim=dict(zip(names, (err.mean((0, 1)) / std).round(3).tolist())),
               mae_first_step=float(err[:, 0].mean()), mae_last_step=float(err[:, -1].mean()),
               corr_per_dim=dict(zip(names, [float(np.corrcoef(pred[:, 0, d], gt[:, 0, d])[0, 1]) if gt[:, 0, d].std() > 1e-6 else None for d in range(A.shape[1])])),
               action_std_per_dim=dict(zip(names, std.round(4).tolist())))
    (out / "replay_metrics.json").write_text(json.dumps(rep, indent=1))
    np.savez_compressed(out / "replay_pred.npz", rows=np.array(rows), pred=pred, gt=gt, t=t[rows])
    # plot: first-step prediction vs GT for every dim
    fig, axs = plt.subplots(19, 1, figsize=(13, 30), sharex=True)
    for d in range(19):
        axs[d].plot(t[rows], gt[:, 0, d], lw=1.0, label="recorded"); axs[d].plot(t[rows], pred[:, 0, d], lw=0.9, alpha=0.8, label="predicted (chunk step 0)")
        axs[d].set_ylabel(names[d], fontsize=7); axs[d].grid(alpha=0.3)
    axs[0].legend(fontsize=8); axs[-1].set_xlabel("t_rel (s)"); fig.tight_layout(); fig.savefig(out / "replay_actions.png", dpi=90)
    # plot: full chunks at a few rows for the grippers/base
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, d in zip(axs, (15, 14, 16)):
        ax.plot(t, A[:, d], color="k", lw=1.0, label="recorded")
        for i in range(0, len(rows), max(1, len(rows) // 40)):
            ax.plot(t[rows[i]:rows[i] + H], pred[i, :, d], color="tab:red", lw=0.8, alpha=0.7)
        ax.set_ylabel(names[d]); ax.grid(alpha=0.3)
    axs[0].legend(); axs[-1].set_xlabel("t_rel (s)"); fig.tight_layout(); fig.savefig(out / "replay_chunks.png", dpi=100)
    print(json.dumps({k: rep[k] for k in ("n_rows", "infer_ms_first", "infer_ms_median", "mae_first_step", "mae_last_step")}, indent=1))
    print("mae/std per dim:", rep["mae_over_std_per_dim"]); print("->", out)


if __name__ == "__main__":
    main()
