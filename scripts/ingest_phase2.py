#!/usr/bin/env python
"""PHASE 2: turn the user's trim/prompt/success answers into canonical episodes.

    $PY scripts/ingest_phase2.py <record_id> --date 2026-09-14 \
        --seg 15 30 "Stack the red cylinder on the green cylinder." true \
        --seg 82 93 "Stack the red cylinder on the green cylinder." true  ...
    (or --segments-file segs.yaml with a list of {start_s, end_s, prompt, success})

Per segment: writes configs/episodes/<record_id>_segNN.yaml (canonical 19-D schema, 15 Hz, 3 cameras), extracts +
validates (validation.md, QC montage, preview.mp4), then updates configs/episodes/registry.yaml and the
per-recording manifest configs/lerobot/<record_id>.yaml (success-only). Runs pytest. Does NOT convert/train.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, yaml

REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
OUT_ROOT = "/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical"
REGISTRY = REPO / "configs/episodes/registry.yaml"

CFG_TEMPLATE = """# Canonical-extraction config: segment {n} of {N} of recording {rid}. Times = seconds of MCAP log_time
# relative to the first message of the bag (the clock burned into the phase-1 preview MP4).
episode_id: {rid}
record_id: {rid}
segment: {n}
actual_date: "{date}"
mcap: data/raw/{rid}/{mcap}

# Supplied by the user (not inferred).
prompt: {prompt}
success: {success}

trim:
  start_s: {s}
  end_s: {e}

rate_hz: 15

align: {{proprio: hold, action: next, camera: hold}}
tolerances: {{proprio_max_age_s: 0.05, action_max_lead_s: 0.10, camera_max_age_s: 0.06, fail_on_violation: false}}
cameras:
  left:  {{eye: right, resize: null}}
  right: {{eye: left, resize: null}}
  head:  {{eye: left, resize: [960, 960]}}
jpeg_quality: 95
output_root: {out_root}
"""


def summarize(meta_path: Path) -> dict:
    m = json.loads(meta_path.read_text()); sv = m["stream_violations"]; z = np.load(meta_path.parent / "episode.npz"); A = z["action"]
    return dict(K=m["grid"]["K"], variant=m["variant"], dir=str(meta_path.parent), blockers=m["blockers"],
                n_missing=sum(x["n_missing"] for x in sv.values()), n_out_of_tol=sum(x["n_stale"] for x in sv.values()),
                reused_frames=sum(x.get("n_reused_frames", 0) for x in sv.values()), nan=m["stats"]["n_nan_proprio"] + m["stats"]["n_nan_action"],
                max_age_ms=dict(proprio=round(max(x["max_age_s"] for x in sv.values() if x["kind"] == "proprio") * 1e3, 1),
                                action=round(max(x["max_abs_age_s"] for x in sv.values() if x["kind"] == "action") * 1e3, 1),
                                camera=round(max(x["max_age_s"] for x in sv.values() if x["kind"] == "camera") * 1e3, 1)),
                frames={c: f["n_written"] for c, f in m["frames"].items()},
                left_gripper_min=float(A[:, 14].min()), right_gripper_min=float(A[:, 15].min()), base_active_frac=float((np.abs(A[:, 16:19]) > 0.2).any(1).mean()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("record_id"); ap.add_argument("--date", required=True)
    ap.add_argument("--seg", nargs=4, action="append", metavar=("START", "END", "PROMPT", "SUCCESS"), default=[])
    ap.add_argument("--segments-file"); ap.add_argument("--no-frames", action="store_true"); ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(); rid = a.record_id
    segs = [dict(start_s=float(s), end_s=float(e), prompt=p, success=str(ok).lower() in ("true", "1", "yes")) for s, e, p, ok in a.seg]
    if a.segments_file: segs += yaml.safe_load(Path(a.segments_file).read_text())
    if not segs: sys.exit("no segments given")
    raw = REPO / "data/raw" / rid; mcap = sorted(raw.glob("*.mcap"))
    if len(mcap) != 1: sys.exit(f"{raw}: expected one .mcap")
    from dexteleop.extract import extract, load_config
    from dexteleop.report import build_report
    results = []
    for n, sg in enumerate(segs, 1):
        cfg_path = REPO / "configs/episodes" / f"{rid}_seg{n:02d}.yaml"
        cfg_path.write_text(CFG_TEMPLATE.format(n=n, N=len(segs), rid=rid, date=a.date, mcap=mcap[0].name, prompt=json.dumps(sg["prompt"]), success=str(sg["success"]).lower(),
                                                s=f"{sg['start_s']:g}", e=f"{sg['end_s']:g}", out_root=OUT_ROOT))
        print(f"=== seg{n:02d}: {sg['start_s']:g}-{sg['end_s']:g} s  {sg['prompt']!r}  success={sg['success']} ===")
        out = extract(load_config(cfg_path), REPO, write_frames=not a.no_frames, progress=False)
        if not a.no_frames: build_report(out)
        r = summarize(out / "meta.json") | dict(segment=n, config=str(cfg_path.relative_to(REPO)), **sg); results.append(r)
        print(f"   K={r['K']} missing={r['n_missing']} out_of_tol={r['n_out_of_tol']} nan={r['nan']} reused={r['reused_frames']} max_age={r['max_age_ms']} -> {out}")
    # ---- registry + per-recording manifest ------------------------------------------------------------------------------
    reg = yaml.safe_load(REGISTRY.read_text()) if REGISTRY.exists() else {}
    reg[rid] = dict(record_id=rid, actual_date=a.date, mcap=str(mcap[0]), segments=[{k: v for k, v in r.items() if k != "dir"} | {"canonical": r["dir"]} for r in results])
    REGISTRY.write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True))
    man = REPO / "configs/lerobot" / f"{rid}.yaml"
    man.write_text(f"# success-only canonical segments of {rid} (generated by ingest_phase2.py)\n" + "".join(f"- canonical: {r['dir']}\n" for r in results if r["success"]))
    tests = "skipped" if a.skip_tests else subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests")], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    print("\n=== PHASE 2 REPORT ===")
    for r in results:
        print(f"seg{r['segment']:02d} {r['start_s']:g}-{r['end_s']:g}s K={r['K']} success={r['success']} prompt={r['prompt']!r}\n      {r['dir']}\n      missing={r['n_missing']} out_of_tol={r['n_out_of_tol']} nan={r['nan']} reused={r['reused_frames']} "
              f"max_age_ms={r['max_age_ms']} frames={r['frames']} gripL_min={r['left_gripper_min']:+.2f} gripR_min={r['right_gripper_min']:+.2f} base_active={r['base_active_frac']*100:.0f}% blockers={r['blockers']}")
    print(f"registry: {REGISTRY.relative_to(REPO)}   manifest: {man.relative_to(REPO)} ({sum(r['success'] for r in results)} success segments)   pytest: {tests}")


if __name__ == "__main__":
    main()
