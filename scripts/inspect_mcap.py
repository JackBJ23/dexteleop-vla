#!/usr/bin/env python
"""Full-pass inspection of an original DexTeleop MCAP: schemas, per-topic timing, per-field statistics.

Read-only. Decodes every message with the schemas embedded in the MCAP (mcap-ros2-support), so no ROS
install is needed. Writes <out>/<episode>_inspection.json + .md.  Run under conda env `dexteleop-data`.

    python scripts/inspect_mcap.py data/raw/<ep>/<ep>_0.mcap --out reports/
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

SKIP_DECODE = {"/tf"}                      # 124k TransformStamped[] -- timing only
ARRAY_FIELDS = ("position", "velocity", "effort", "axes", "buttons", "data")
POSE_FIELDS = ("position.x", "position.y", "position.z",
               "orientation.x", "orientation.y", "orientation.z", "orientation.w")


def _get(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _stamp_ns(msg):
    h = getattr(msg, "header", None)
    if h is None:
        return None
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def summarize(vals):
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return None
    return dict(n=int(a.size), min=float(np.nanmin(a)), max=float(np.nanmax(a)), mean=float(np.nanmean(a)),
                std=float(np.nanstd(a)), n_unique=int(np.unique(np.round(a[~np.isnan(a)], 6)).size),
                n_nan=int(np.isnan(a).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--max-per-topic", type=int, default=0, help="debug: stop per topic after N msgs")
    args = ap.parse_args()
    path = Path(args.mcap)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ep = path.stem.rsplit("_", 1)[0]

    t0 = time.time()
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summ = reader.get_summary()
        stats = summ.statistics
        chan_by_id = summ.channels
        schema_by_id = summ.schemas
        bag_start_ns, bag_end_ns = stats.message_start_time, stats.message_end_time

        topics = {}
        for ch in chan_by_id.values():
            sc = schema_by_id[ch.schema_id]
            topics[ch.topic] = dict(
                topic=ch.topic, schema=sc.name, encoding=sc.encoding, msg_encoding=ch.message_encoding,
                count_index=stats.channel_message_counts.get(ch.id, 0),
                schema_text=sc.data.decode("utf-8", "replace"))

        log_t = defaultdict(list); pub_t = defaultdict(list); hdr_t = defaultdict(list)
        fields = defaultdict(lambda: defaultdict(list))      # topic -> field -> values
        arr_len = defaultdict(lambda: defaultdict(set))       # topic -> field -> set(len)
        names = {}                                            # topic -> JointState.name / layout labels
        vid = defaultdict(lambda: defaultdict(list))          # video packet meta
        n_seen = defaultdict(int)

        for it in reader.iter_decoded_messages(log_time_order=True):
            ch, msg, m = it.channel, it.decoded_message, it.message
            tp = ch.topic
            n_seen[tp] += 1
            if args.max_per_topic and n_seen[tp] > args.max_per_topic:
                continue
            log_t[tp].append(m.log_time); pub_t[tp].append(m.publish_time)
            if tp in SKIP_DECODE:
                continue
            hs = _stamp_ns(msg)
            if hs is not None:
                hdr_t[tp].append(hs)
            sc = topics[tp]["schema"]
            if sc == "sensor_msgs/msg/JointState":
                if tp not in names:
                    names[tp] = list(msg.name)
                for fld in ("position", "velocity", "effort"):
                    v = list(getattr(msg, fld))
                    arr_len[tp][fld].add(len(v))
                    for i, x in enumerate(v):
                        fields[tp][f"{fld}[{i}]"].append(x)
            elif sc == "sensor_msgs/msg/Joy":
                for fld in ("axes", "buttons"):
                    v = list(getattr(msg, fld)); arr_len[tp][fld].add(len(v))
                    for i, x in enumerate(v):
                        fields[tp][f"{fld}[{i}]"].append(x)
            elif sc in ("geometry_msgs/msg/Pose", "geometry_msgs/msg/PoseStamped"):
                base = msg.pose if sc.endswith("Stamped") else msg
                for fld in POSE_FIELDS:
                    fields[tp][fld].append(_get(base, fld))
            elif sc == "std_msgs/msg/Float32MultiArray":
                v = list(msg.data); arr_len[tp]["data"].add(len(v))
                if tp not in names:
                    names[tp] = [f"{d.label}:{d.size}:{d.stride}" for d in msg.layout.dim]
                for i, x in enumerate(v):
                    fields[tp][f"data[{i}]"].append(x)
            elif sc in ("std_msgs/msg/Float64", "std_msgs/msg/Int32", "std_msgs/msg/Bool"):
                fields[tp]["data"].append(float(msg.data))
            elif sc == "std_msgs/msg/String":
                fields[tp]["data_str"].append(str(msg.data))
            elif sc == "ffmpeg_image_transport_msgs/msg/FFMPEGPacket":
                d = vid[tp]
                d["pts"].append(int(msg.pts)); d["flags"].append(int(msg.flags)); d["size"].append(len(msg.data))
                d["width"].append(int(msg.width)); d["height"].append(int(msg.height))
                if "encoding" not in d or msg.encoding not in d["encoding"]:
                    d.setdefault("encoding", []).append(str(msg.encoding))
                d["frame_id"].append(str(msg.header.frame_id))
            elif sc.endswith("ErrorCode"):
                for k in msg.__slots__ if hasattr(msg, "__slots__") else vars(msg):
                    k = k.lstrip("_")
                    if k == "header":
                        continue
                    val = getattr(msg, k, None)
                    if isinstance(val, (int, float, bool)):
                        fields[tp][k].append(float(val))
                    elif isinstance(val, str):
                        fields[tp][k + "_str"].append(val)
                    elif hasattr(val, "__len__"):
                        for i, x in enumerate(list(val)):
                            if isinstance(x, (int, float, bool)):
                                fields[tp][f"{k}[{i}]"].append(float(x))
    elapsed = time.time() - t0

    # ---- per-topic report -------------------------------------------------------------------
    rep = dict(mcap=str(path), size_bytes=path.stat().st_size, episode=ep, decode_seconds=elapsed,
               bag_start_ns=bag_start_ns, bag_end_ns=bag_end_ns, duration_s=(bag_end_ns - bag_start_ns) / 1e9,
               message_count=stats.message_count, topics={})
    for tp, info in sorted(topics.items()):
        lt = np.asarray(log_t.get(tp, []), dtype=np.int64)
        d = dict(info); d.pop("schema_text")
        d["count_seen"] = int(lt.size)
        if lt.size:
            rel = (lt - bag_start_ns) / 1e9
            d["first_s"], d["last_s"] = float(rel[0]), float(rel[-1])
            if lt.size > 1:
                dt = np.diff(lt) / 1e9
                d["rate_hz"] = float((lt.size - 1) / (rel[-1] - rel[0])) if rel[-1] > rel[0] else None
                d["dt_ms"] = dict(median=float(np.median(dt) * 1e3), p95=float(np.percentile(dt, 95) * 1e3),
                                  p99=float(np.percentile(dt, 99) * 1e3), max=float(dt.max() * 1e3),
                                  n_gt_2x_median=int((dt > 2 * np.median(dt)).sum()),
                                  n_nonmonotonic=int((dt < 0).sum()))
            pt = np.asarray(pub_t[tp], dtype=np.int64)
            d["pub_minus_log_ms"] = summarize((pt - lt) / 1e6)
            if hdr_t.get(tp):
                ht = np.asarray(hdr_t[tp], dtype=np.int64)
                d["header_minus_log_ms"] = summarize((ht - lt) / 1e6)
                d["header_all_zero"] = bool((ht == 0).all())
                hdt = np.diff(ht) / 1e9
                d["header_dt_nonmonotonic"] = int((hdt < 0).sum()) if hdt.size else 0
        if tp in names:
            d["names"] = names[tp]
        if tp in arr_len:
            d["array_lengths"] = {k: sorted(v) for k, v in arr_len[tp].items()}
        if tp in fields:
            d["fields"] = {}
            for k, v in fields[tp].items():
                if k.endswith("_str"):
                    u, c = np.unique(np.asarray(v), return_counts=True)
                    d["fields"][k] = {"unique": {str(a): int(b) for a, b in zip(u, c)}} if u.size <= 30 else {"n_unique": int(u.size)}
                else:
                    d["fields"][k] = summarize(v)
        if tp in vid:
            v = vid[tp]
            fl = np.asarray(v["flags"]); pts = np.asarray(v["pts"], dtype=np.int64)
            d["video"] = dict(encoding=v["encoding"], width=sorted(set(v["width"])), height=sorted(set(v["height"])),
                              frame_id=sorted(set(v["frame_id"])), n_packets=int(fl.size),
                              n_keyframes=int((fl & 1).sum()), flags_unique=sorted(set(int(x) for x in fl)),
                              pts_first=int(pts[0]), pts_last=int(pts[-1]),
                              pts_dt_unique=sorted(set(int(x) for x in np.unique(np.diff(pts))))[:20],
                              pts_nonmonotonic=int((np.diff(pts) <= 0).sum()),
                              bytes_mean=float(np.mean(v["size"])), bytes_max=int(np.max(v["size"])),
                              keyframe_interval_packets=int(np.median(np.diff(np.flatnonzero(fl & 1)))) if (fl & 1).sum() > 1 else None)
        rep["topics"][tp] = d

    (out / f"{ep}_inspection.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / f"{ep}_schemas.txt").write_text("\n\n".join(
        f"##### {t}  ({i['schema']})\n{i['schema_text']}" for t, i in sorted(topics.items())))

    # ---- markdown ------------------------------------------------------------------------------
    L = [f"# MCAP inspection: {ep}", "", f"- file: `{path}` ({path.stat().st_size/2**20:.1f} MiB)",
         f"- bag start (ns since epoch): {bag_start_ns}", f"- duration: {rep['duration_s']:.3f} s, messages: {stats.message_count}",
         f"- decode pass: {elapsed:.0f} s", "", "## Topics (timing from MCAP log_time)", "",
         "| topic | schema | n | first s | last s | Hz | dt med/p99/max ms | gaps>2x | hdr−log ms (mean) |", "|---|---|---|---|---|---|---|---|---|"]
    for tp, d in rep["topics"].items():
        if d["count_seen"] == 0:
            L.append(f"| `{tp}` | {d['schema'].split('/')[-1]} | 0 | | | | | | |"); continue
        dt = d.get("dt_ms", {}); hl = d.get("header_minus_log_ms")
        L.append(f"| `{tp}` | {d['schema'].split('/')[-1]} | {d['count_seen']} | {d['first_s']:.3f} | {d['last_s']:.3f} | "
                 f"{d.get('rate_hz') or 0:.1f} | {dt.get('median',0):.1f}/{dt.get('p99',0):.1f}/{dt.get('max',0):.0f} | "
                 f"{dt.get('n_gt_2x_median','')} | {'' if hl is None else f'{hl[chr(109)+chr(101)+chr(97)+chr(110)]:.1f}'} |")
    L += ["", "## JointState topics: names + per-field stats", ""]
    for tp, d in rep["topics"].items():
        if d["schema"] != "sensor_msgs/msg/JointState" or d["count_seen"] == 0:
            continue
        L += [f"### `{tp}`", f"- names: {d.get('names')}", f"- array lengths: {d.get('array_lengths')}", "",
              "| field | min | max | mean | std | n_unique |", "|---|---|---|---|---|---|"]
        for k, s in d["fields"].items():
            if s: L.append(f"| {k} | {s['min']:.4f} | {s['max']:.4f} | {s['mean']:.4f} | {s['std']:.4f} | {s['n_unique']} |")
        L.append("")
    L += ["## Video topics", ""]
    for tp, d in rep["topics"].items():
        if "video" in d:
            L.append(f"- `{tp}`: {json.dumps(d['video'])}")
    L += ["", "## Other decoded topics", ""]
    for tp, d in rep["topics"].items():
        if d["schema"] in ("sensor_msgs/msg/JointState",) or "video" in d or "fields" not in d:
            continue
        L.append(f"### `{tp}` ({d['schema']})" + (f"  names={d['names']}" if d.get("names") else ""))
        for k, s in d["fields"].items():
            if isinstance(s, dict) and "min" in s:
                L.append(f"- {k}: min={s['min']:.4f} max={s['max']:.4f} std={s['std']:.4f} n_unique={s['n_unique']}")
            else:
                L.append(f"- {k}: {s}")
        L.append("")
    (out / f"{ep}_inspection.md").write_text("\n".join(L))
    print(f"wrote {out}/{ep}_inspection.{{json,md}} + _schemas.txt in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
