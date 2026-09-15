"""Decode `ffmpeg_image_transport_msgs/FFMPEGPacket` streams (HEVC in this robot) with PyAV.

Frames are addressed by SOURCE PACKET INDEX in the MCAP topic (0-based, log_time order); the timestamp
of a decoded frame is the MCAP log_time / header.stamp of the packet that produced it -- never the
container/pts time of any re-muxed file.

HEVC needs decoding to start at an IRAP (IDR/CRA/BLA) picture. The FFMPEGPacket `flags` field is 0 on
every packet in this recording, so keyframes are found by parsing NAL unit types from the bitstream.
"""
from __future__ import annotations

import dataclasses
from typing import Iterator, Sequence

import av
import numpy as np

HEVC_IRAP_TYPES = set(range(16, 24))   # BLA_W_LP..RSV_IRAP_VCL23 (IDR_W_RADL=19, IDR_N_LP=20, CRA=21)
H264_IDR = 5


def _nal_starts(buf: bytes) -> Iterator[int]:
    """Yield byte offsets just after each Annex-B start code (00 00 01 / 00 00 00 01)."""
    i, n = 0, len(buf)
    while True:
        j = buf.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= n:
            return
        yield j + 3
        i = j + 3


def is_keyframe(data: bytes, encoding: str) -> bool:
    enc = encoding.lower()
    for off in _nal_starts(data):
        b = data[off]
        if "hevc" in enc or "265" in enc:
            if ((b >> 1) & 0x3F) in HEVC_IRAP_TYPES:
                return True
        else:  # h264
            if (b & 0x1F) == H264_IDR:
                return True
    return False


def codec_name(encoding: str) -> str:
    e = encoding.lower()
    if "hevc" in e or "265" in e:
        return "hevc"
    if "264" in e or "avc" in e:
        return "h264"
    raise ValueError(f"unsupported FFMPEGPacket encoding {encoding!r}")


@dataclasses.dataclass
class PacketStreamDecoder:
    """Sequential decoder over a list of raw packets. `decode_range(first_idx, last_idx)` yields
    (packet_index, frame_rgb_uint8) for every packet index in [first_idx, last_idx] that produced a frame,
    automatically starting from the nearest preceding keyframe so that the first requested frame is valid.
    Packets before the requested range that are only needed for warm-up are decoded but not yielded.
    """
    packets: Sequence[bytes]
    encoding: str
    threads: int = 4

    def __post_init__(self):
        self._is_key = None

    def keyframe_indices(self) -> np.ndarray:
        if self._is_key is None:
            self._is_key = np.fromiter((is_keyframe(p, self.encoding) for p in self.packets), dtype=bool,
                                       count=len(self.packets))
        return np.flatnonzero(self._is_key)

    def _start_index(self, first_idx: int) -> int:
        keys = self.keyframe_indices()
        prior = keys[keys <= first_idx]
        if prior.size == 0:
            raise RuntimeError(f"no keyframe at or before packet {first_idx} (first keyframe at {keys[:1]})")
        return int(prior[-1])

    def decode_range(self, first_idx: int, last_idx: int) -> Iterator[tuple[int, np.ndarray]]:
        start = self._start_index(first_idx)
        ctx = av.CodecContext.create(codec_name(self.encoding), "r")
        ctx.thread_type = "AUTO"
        ctx.thread_count = self.threads
        # One FFMPEGPacket == one access unit, so each fed packet yields <=1 frame after the decoder's
        # own delay; we tag output frames with the packet index they came from by keeping a FIFO of
        # fed indices (the decoder has no B-frame reordering in a low-latency robot stream, and we
        # verify monotonic pts to catch it if it ever does).
        fed: list[int] = []
        last_pts = None
        def _emit(frames):
            nonlocal last_pts
            for fr in frames:
                idx = fed.pop(0)
                if fr.pts is not None and last_pts is not None and fr.pts < last_pts:
                    raise RuntimeError("decoder reordered frames (pts went backwards); packet<->frame mapping unsafe")
                last_pts = fr.pts
                if idx >= first_idx:
                    yield idx, fr.to_ndarray(format="rgb24")
        for idx in range(start, last_idx + 1):
            pkt = av.Packet(self.packets[idx])
            fed.append(idx)
            yield from _emit(ctx.decode(pkt))
        yield from _emit(ctx.decode(None))  # flush
        if fed:
            # packets that produced no picture (should be none for an all-frame stream) -- report, don't hide
            self.undecoded = list(fed)
        else:
            self.undecoded = []
