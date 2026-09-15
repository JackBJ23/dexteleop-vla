#!/usr/bin/env python
"""Build the TRAINING dataset from named recordings (success-only segments from configs/episodes/registry.yaml).

    $PY scripts/build_dataset.py --recordings rec_20260821_052219_d4f4fc76 [rec_...] [--name dexteleop_train] [--config pi05_dexteleop_v1]
                                 [--prompt "Stack the red cylinder on the green cylinder."]   # keep only segments with EXACTLY this prompt

Writes configs/lerobot/<name>.yaml (+ a dated copy), then submits sbatch/dt01_build_overfit1.sbatch which converts to
LeRobot repo_id local/<name> (node-local -> netscratch), computes OpenPI norm stats for --config, and runs the
dataset checks + dataloader smoke (POST_CHECKS=1). Training is NOT launched.
"""
import argparse, datetime, subprocess, sys
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[1]; REGISTRY = REPO / "configs/episodes/registry.yaml"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recordings", nargs="+", required=True); ap.add_argument("--name", default="dexteleop_train")
    ap.add_argument("--config", default="pi05_dexteleop_v1"); ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], metavar="RECORD:SEG", help="hold out segments, e.g. rec_20260821_052219_d4f4fc76:6 (not written to the manifest)")
    ap.add_argument("--prompt", default=None, help="if given, keep only success segments whose prompt equals this string exactly (case/whitespace-insensitive)")
    a = ap.parse_args(); reg = yaml.safe_load(REGISTRY.read_text())
    missing = [r for r in a.recordings if r not in reg]
    if missing: sys.exit(f"not in registry (run ingest_phase2 first): {missing}")
    norm = lambda p: " ".join(str(p).split()).strip().lower()
    excl = {(e.split(":")[0], int(e.split(":")[1])) for e in a.exclude}
    rows = [(rid, sg) for rid in a.recordings for sg in reg[rid]["segments"] if sg["success"] and (a.prompt is None or norm(sg["prompt"]) == norm(a.prompt)) and (rid, sg["segment"]) not in excl]
    if excl: print(f"held out (excluded): {sorted(excl)}")
    if a.prompt is not None:
        avail = sorted({sg["prompt"] for rid in a.recordings for sg in reg[rid]["segments"] if sg["success"]})
        if not rows: sys.exit(f"no success segments with prompt {a.prompt!r}; prompts available in these recordings: {avail}")
        print(f"prompt filter {a.prompt!r}: kept {len(rows)} segments (available prompts: {avail})")
    lines = [f"# training dataset '{a.name}' -- generated {datetime.date.today()} by build_dataset.py from recordings: {' '.join(a.recordings)}"
             + (f"   prompt filter: {a.prompt!r}" if a.prompt else "") + (f"   held out: {sorted(excl)}" if excl else ""),
             f"# {len(rows)} success segments, {sum(sg['K'] for _, sg in rows)} rows"]
    for rid, sg in rows: lines.append(f"- canonical: {sg['canonical']}   # {rid} seg{sg['segment']:02d} {sg['start_s']:g}-{sg['end_s']:g}s {sg['prompt']!r}")
    man = REPO / "configs/lerobot" / f"{a.name}.yaml"; man.write_text("\n".join(lines) + "\n")
    (REPO / "configs/lerobot" / f"{a.name}_{datetime.date.today():%Y%m%d}.yaml").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\nmanifest -> {man}")
    if a.no_submit: return
    env = dict(MANIFEST=str(man), REPO_ID=f"local/{a.name}", CONFIG=a.config, EXP=f"{a.name}_lora", POST_CHECKS="1")
    cmd = ["sbatch", "--parsable", str(REPO / "sbatch/dt01_build_overfit1.sbatch")]
    jid = subprocess.run(cmd, env={**__import__("os").environ, **env}, capture_output=True, text=True, check=True).stdout.strip()
    (REPO / "sbatch" / f".build_{a.name}_jid").write_text(jid); print(f"submitted build job {jid}  ({' '.join(f'{k}={v}' for k, v in env.items())})")


if __name__ == "__main__":
    main()
