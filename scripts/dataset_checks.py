"""Post-build checks for a DexTeleop LeRobot dataset + OpenPI config (run under openpi `uv run`, GPU node ok):
 1. dataset validation vs the canonical episodes (counts, prompts, images, shapes, NaNs, boundaries)
 2. norm-stat analysis: per-dim ranges, degenerate quantile dims, normalized min/max with and without the guard
 3. dataloader smoke test through the real TrainConfig (shapes, masks, prompts) + sample images to PNG
"""
import argparse, json
from pathlib import Path
import numpy as np, yaml

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from openpi.training import config as _config, data_loader as _dl
    from openpi import transforms as T
    import openpi.models.model as _model
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    cfg = _config.get_config(a.config); repo = cfg.data.repo_id; R = {}

    # ---- 1. dataset validation ---------------------------------------------------------------------------------------
    eps = [json.loads((Path(e["canonical"]) / "meta.json").read_text()) | {"dir": e["canonical"]} for e in yaml.safe_load(Path(a.manifest).read_text())]
    meta = LeRobotDatasetMetadata(repo); ds = LeRobotDataset(repo)
    hf = ds.hf_dataset
    ep_idx = np.asarray(hf["episode_index"]); fr_idx = np.asarray(hf["frame_index"]); task_idx = np.asarray(hf["task_index"])
    lengths = [int((ep_idx == i).sum()) for i in range(meta.total_episodes)]
    S = np.stack([np.asarray(x) for x in hf["state"]]); A = np.stack([np.asarray(x) for x in hf["actions"]])
    val = dict(repo_id=repo, root=str(ds.root), fps=meta.fps, total_episodes=meta.total_episodes, total_frames=meta.total_frames, tasks=meta.tasks,
               episode_lengths=lengths, canonical_K=[m["grid"]["K"] for m in eps], state_shape=list(S.shape), action_shape=list(A.shape),
               nan_state=int(np.isnan(S).sum()), nan_action=int(np.isnan(A).sum()), image_keys=[k for k in meta.features if meta.features[k]["dtype"] in ("image", "video")],
               image_shapes={k: meta.features[k]["shape"] for k in meta.features if meta.features[k]["dtype"] in ("image", "video")})
    val["boundaries_ok"] = all(fr_idx[ep_idx == i].tolist() == list(range(lengths[i])) for i in range(meta.total_episodes))
    val["prompt_match"] = []
    for i, m in enumerate(eps):
        tid = set(task_idx[ep_idx == i].tolist()); prompt = meta.tasks[tid.pop()] if len(tid) == 1 else None
        z = np.load(Path(m["dir"]) / "episode.npz")
        same = bool(lengths[i] == m["grid"]["K"] and np.allclose(S[ep_idx == i], z["proprio"]) and np.allclose(A[ep_idx == i], z["action"]))
        val["prompt_match"].append(dict(episode=i, canonical=Path(m["dir"]).parent.name + "/" + Path(m["dir"]).name, K=lengths[i], prompt_dataset=prompt, prompt_meta=m["prompt"], prompt_ok=prompt == m["prompt"], rows_identical_to_canonical=same))
    exp_frames = sum(m["grid"]["K"] for m in eps)
    val["expected"] = dict(episodes=len(eps), frames=exp_frames)
    val["all_ok"] = (meta.total_episodes == len(eps) and meta.total_frames == exp_frames == sum(lengths) and val["boundaries_ok"] and val["nan_state"] == 0 and val["nan_action"] == 0
                     and all(p["prompt_ok"] and p["rows_identical_to_canonical"] for p in val["prompt_match"]) and S.shape[1] == 19 and A.shape[1] == 19 and len(val["image_keys"]) == 3)
    R["dataset"] = val; print("DATASET all_ok =", val["all_ok"])

    # ---- 2. norm stats --------------------------------------------------------------------------------------------------
    dc = cfg.data.create(cfg.assets_dirs, cfg.model); ns = dc.norm_stats; mr = dc.quantile_min_range
    names = {"state": eps[0]["proprio_names"], "actions": eps[0]["action_names"]}; raw = {"state": S, "actions": A}
    nsum = {}
    for key in ("state", "actions"):
        st = ns[key]; q01, q99 = st.q01, st.q99; rng = q99 - q01; deg = rng < T.QUANTILE_DEGENERATE_EPS
        xn_raw = (raw[key] - q01) / (rng + 1e-6) * 2 - 1
        e01, e99 = T._effective_quantiles(st, 19, mr); xn = (raw[key] - e01) / (e99 - e01 + 1e-6) * 2 - 1
        rows = []
        for d in range(19):
            rows.append(dict(dim=d, name=names[key][d], mean=float(st.mean[d]), std=float(st.std[d]), q01=float(q01[d]), q99=float(q99[d]), q_range=float(rng[d]),
                             degenerate=bool(deg[d]), uses_fallback=bool(deg[d] and mr is not None), eff_q01=float(e01[d]), eff_q99=float(e99[d]),
                             norm_min_raw=float(xn_raw[:, d].min()), norm_max_raw=float(xn_raw[:, d].max()), norm_min=float(xn[:, d].min()), norm_max=float(xn[:, d].max())))
        nsum[key] = rows
    R["norm"] = dict(quantile_min_range=mr, use_quantile_norm=dc.use_quantile_norm, dims=nsum)

    # ---- 3. dataloader smoke ----------------------------------------------------------------------------------------------
    loader = _dl.create_data_loader(cfg, shuffle=True, num_batches=1)
    obs, act = next(iter(loader))
    smoke = dict(batch_size=cfg.batch_size, images={k: list(v.shape) + [str(v.dtype)] for k, v in obs.images.items()}, image_masks={k: [list(v.shape), bool(np.asarray(v).all())] for k, v in obs.image_masks.items()},
                 state=list(obs.state.shape), state_nonzero_dims=int((np.abs(np.asarray(obs.state)).max(0) > 0).sum()), actions=list(act.shape), action_nonzero_dims=int((np.abs(np.asarray(act)).max((0, 1)) > 0).sum()),
                 tokenized_prompt=list(obs.tokenized_prompt.shape), prompt_tokens_used=int(np.asarray(obs.tokenized_prompt_mask).sum(1).max()),
                 image_value_range=[float(np.asarray(obs.images["base_0_rgb"]).min()), float(np.asarray(obs.images["base_0_rgb"]).max())],
                 state_norm_range=[float(np.asarray(obs.state)[:, :19].min()), float(np.asarray(obs.state)[:, :19].max())], action_norm_range=[float(np.asarray(act)[..., :19].min()), float(np.asarray(act)[..., :19].max())],
                 padded_dims_all_zero=bool((np.asarray(obs.state)[:, 19:] == 0).all() and (np.asarray(act)[..., 19:] == 0).all()),
                 action_dim_mapping={i: n for i, n in enumerate(names["actions"])} | {"19..31": "zero padding (PadStatesAndActions to 32)"},
                 image_slot_mapping={"base_0_rgb": "head_image", "left_wrist_0_rgb": "left_image", "right_wrist_0_rgb": "right_image"})
    R["dataloader"] = smoke
    (out / "dataset_checks.json").write_text(json.dumps(R, indent=1, default=str))   # save before the sample step
    # decoded prompts + sample images (untransformed dataset -> our transforms, specific rows from old + new recordings)
    tds = _dl.transform_dataset(_dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model), dc, skip_norm_stats=True)
    import cv2
    starts = np.cumsum([0] + lengths[:-1])
    pick_eps = sorted(set([0, len(lengths) // 2, len(lengths) - 1] + list(range(0, len(lengths), max(1, len(lengths) // 4)))))[:6]
    picks = [(e, lengths[e] // 2) for e in pick_eps]          # middle frame of a spread of episodes
    samples = []
    for e, off in picks:
        x = tds[int(starts[e] + off)]; ims = x["image"]
        def u8(im):
            im = np.asarray(im)
            if np.issubdtype(im.dtype, np.floating):
                im = ((im + 1.0) * 127.5 if im.min() < 0 else im * (255.0 if im.max() <= 1.0 else 1.0))
            return np.ascontiguousarray(np.clip(im, 0, 255).astype(np.uint8))
        row = np.ascontiguousarray(np.hstack([u8(ims[k]) for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")]))
        ptxt = meta.tasks[int(task_idx[int(starts[e] + off)])]   # model transforms already tokenized the prompt
        cv2.putText(row, f"ep{e} row{off}  {ptxt}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        fn = out / f"sample_ep{e}_row{off}.png"; cv2.imwrite(str(fn), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
        samples.append(dict(episode=e, row=off, prompt=ptxt, file=str(fn), image_shape=list(np.asarray(ims["base_0_rgb"]).shape), state19=np.round(np.asarray(x["state"])[:19], 3).tolist(),
                            tokenized_prompt_len=int(np.asarray(x["tokenized_prompt_mask"]).sum()) if "tokenized_prompt_mask" in x else None))
    R["samples"] = samples
    (out / "dataset_checks.json").write_text(json.dumps(R, indent=1, default=str)); print("->", out / "dataset_checks.json")


if __name__ == "__main__":   # required: the torch DataLoader spawns workers that re-import this file
    main()
