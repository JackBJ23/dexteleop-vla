"""Read the topics the canonical pipeline needs from an original rosbag2 MCAP, preserving source
timestamps (log_time, publish_time, header.stamp) and the 0-based per-topic message index.

Uses the schemas embedded in the MCAP (mcap-ros2-support) -- no ROS installation required.
The MCAP is opened read-only and never modified.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


@dataclasses.dataclass
class JointStateStream:
    topic: str
    names: list[str]
    log_time_ns: np.ndarray      # int64 [N]  (sorted, MCAP log_time)
    publish_time_ns: np.ndarray  # int64 [N]
    header_stamp_ns: np.ndarray  # int64 [N]  (whatever clock the publisher used; may be unrelated)
    position: np.ndarray         # float64 [N, J]
    velocity: np.ndarray
    effort: np.ndarray

    def field(self, name: str) -> np.ndarray:
        return getattr(self, name)


@dataclasses.dataclass
class VideoStream:
    topic: str
    encoding: str
    width: int
    height: int
    log_time_ns: np.ndarray      # int64 [N]
    publish_time_ns: np.ndarray
    header_stamp_ns: np.ndarray
    pts: np.ndarray              # uint64 [N]  sender-side pts (microseconds in this robot)
    flags: np.ndarray            # uint8 [N]
    packets: list[bytes]


@dataclasses.dataclass
class ScalarStream:
    topic: str
    log_time_ns: np.ndarray
    value: np.ndarray


@dataclasses.dataclass
class McapInfo:
    path: str
    size_bytes: int
    sha256_head_1mb: str         # cheap identity: sha256 of the first 1 MiB + size (full hash of 855 MB is slow)
    message_start_ns: int
    message_end_ns: int
    message_count: int
    topic_counts: dict[str, int]
    schemas: dict[str, str]      # topic -> schema name


def _stamp_ns(msg) -> int:
    h = getattr(msg, "header", None)
    if h is None:
        return -1
    return int(h.stamp.sec) * 1_000_000_000 + int(h.stamp.nanosec)


def mcap_info(path: str | Path) -> McapInfo:
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(1 << 20)
        f.seek(0)
        reader = make_reader(f)
        s = reader.get_summary()
        chans = s.channels
        return McapInfo(str(path), path.stat().st_size,
                        hashlib.sha256(head + str(path.stat().st_size).encode()).hexdigest(),
                        s.statistics.message_start_time, s.statistics.message_end_time, s.statistics.message_count,
                        {c.topic: s.statistics.channel_message_counts.get(cid, 0) for cid, c in chans.items()},
                        {c.topic: s.schemas[c.schema_id].name for c in chans.values()})


def read_topics(path: str | Path, topics: set[str], progress: bool = True) -> dict[str, object]:
    """One indexed pass over `topics`. Returns {topic: JointStateStream | VideoStream | ScalarStream}."""
    path = Path(path)
    acc: dict[str, dict] = {t: dict(log=[], pub=[], hdr=[], rows=[]) for t in topics}
    with path.open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        schema_of = {c.topic: reader.get_summary().schemas[c.schema_id].name for c in reader.get_summary().channels.values()}
        missing = topics - set(schema_of)
        if missing:
            raise KeyError(f"topics not in MCAP: {sorted(missing)}")
        it = reader.iter_decoded_messages(topics=sorted(topics), log_time_order=True)
        if progress:
            from tqdm import tqdm
            total = sum(reader.get_summary().statistics.channel_message_counts.get(cid, 0)
                        for cid, c in reader.get_summary().channels.items() if c.topic in topics)
            it = tqdm(it, total=total, unit="msg", mininterval=2.0, desc="mcap")
        for d in it:
            tp, msg, m = d.channel.topic, d.decoded_message, d.message
            a = acc[tp]
            a["log"].append(m.log_time); a["pub"].append(m.publish_time); a["hdr"].append(_stamp_ns(msg))
            sc = schema_of[tp]
            if sc == "sensor_msgs/msg/JointState":
                if "names" not in a:
                    a["names"] = list(msg.name)
                elif list(msg.name) != a["names"]:
                    raise ValueError(f"{tp}: joint name order changed mid-stream")
                a["rows"].append((list(msg.position), list(msg.velocity), list(msg.effort)))
            elif sc == "ffmpeg_image_transport_msgs/msg/FFMPEGPacket":
                if "enc" not in a:
                    a["enc"], a["w"], a["h"] = str(msg.encoding), int(msg.width), int(msg.height)
                elif (str(msg.encoding), int(msg.width), int(msg.height)) != (a["enc"], a["w"], a["h"]):
                    raise ValueError(f"{tp}: encoding/resolution changed mid-stream")
                a["rows"].append((bytes(msg.data), int(msg.pts), int(msg.flags)))
            elif sc in ("std_msgs/msg/Float64", "std_msgs/msg/Int32", "std_msgs/msg/Bool"):
                a["rows"].append(float(msg.data))
            else:
                raise NotImplementedError(f"{tp}: no loader for {sc}")
    out = {}
    for tp, a in acc.items():
        log = np.asarray(a["log"], np.int64); pub = np.asarray(a["pub"], np.int64); hdr = np.asarray(a["hdr"], np.int64)
        order = np.argsort(log, kind="stable")
        if not np.array_equal(order, np.arange(log.size)):
            raise ValueError(f"{tp}: iterator not in log_time order")   # keep source index == arrival index
        sc = schema_of[tp]
        if sc == "sensor_msgs/msg/JointState":
            J = len(a["names"])
            def col(k):
                arr = np.full((log.size, J), np.nan)
                for i, r in enumerate(a["rows"]):
                    v = r[k]
                    if len(v) == J:
                        arr[i] = v
                    elif len(v) != 0:
                        raise ValueError(f"{tp}: {['position','velocity','effort'][k]} has length {len(v)} != {J}")
                return arr
            out[tp] = JointStateStream(tp, a["names"], log, pub, hdr, col(0), col(1), col(2))
        elif sc == "ffmpeg_image_transport_msgs/msg/FFMPEGPacket":
            out[tp] = VideoStream(tp, a["enc"], a["w"], a["h"], log, pub, hdr,
                                  np.asarray([r[1] for r in a["rows"]], np.uint64),
                                  np.asarray([r[2] for r in a["rows"]], np.uint8), [r[0] for r in a["rows"]])
        else:
            out[tp] = ScalarStream(tp, log, np.asarray(a["rows"], np.float64))
    return out
