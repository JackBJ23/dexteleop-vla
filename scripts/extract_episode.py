#!/usr/bin/env python
"""Run the canonical extraction for one episode config, then build the validation report + QC.

    PYTHONPATH=src python scripts/extract_episode.py configs/episodes/<ep>.yaml [--rate 45] [--no-frames] [--legacy-npz PATH]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dexteleop.extract import extract, load_config
from dexteleop.report import build_report

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--rate", type=float, help="override rate_hz")
ap.add_argument("--start", type=float); ap.add_argument("--end", type=float)
ap.add_argument("--no-frames", action="store_true"); ap.add_argument("--legacy-npz", default=None)
ap.add_argument("--no-report", action="store_true")
a = ap.parse_args()
repo = Path(__file__).resolve().parents[1]
cfg = load_config(a.config)
if a.rate: cfg["rate_hz"] = a.rate
if a.start is not None: cfg["trim"]["start_s"] = a.start
if a.end is not None: cfg["trim"]["end_s"] = a.end
out = extract(cfg, repo, write_frames=not a.no_frames)
print("canonical episode ->", out)
if not a.no_report and not a.no_frames:
    print("report ->", build_report(out, Path(a.legacy_npz) if a.legacy_npz else None))
