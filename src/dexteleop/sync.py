"""Deterministic time alignment of asynchronous ROS streams onto a fixed-rate grid.

Time base: MCAP `log_time` in nanoseconds since epoch (available for every message, including
header-less geometry_msgs/Pose). `header.stamp` is carried alongside where present and its offset
to log_time is reported, but grid alignment uses log_time.

Alignment modes (all deterministic, all preserve the source index of the chosen message):
  hold     : latest message with t_msg <= t_grid          (zero-order hold; staleness >= 0)
  nearest  : message minimising |t_msg - t_grid|
  next     : earliest message with t_msg >= t_grid        (for commands: what the operator sent from now on)
  interp   : linear interpolation between hold and next neighbours (float fields only)
Each mode also returns the age (t_grid - t_msg, seconds; negative for `next`) so the caller can flag
stale/missing data against a configurable tolerance.
"""
from __future__ import annotations

import dataclasses

import numpy as np

MODES = ("hold", "nearest", "next", "interp")


def make_grid(start_ns: int, end_ns: int, rate_hz: float) -> np.ndarray:
    """Grid ticks t_k = start + k/rate for k = 0.. while t_k <= end (inclusive). Integer ns, exact."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if end_ns < start_ns:
        raise ValueError("end < start")
    period_ns = 1e9 / rate_hz
    n = int(np.floor((end_ns - start_ns) / period_ns + 1e-9)) + 1
    # offsets are computed in float64 (exact to <1 ns for any realistic K) and added to the int64 epoch
    # start AFTER rounding, so no precision is lost on ~1.8e18 ns timestamps.
    offsets = np.round(np.arange(n, dtype=np.float64) * period_ns).astype(np.int64)
    return np.int64(start_ns) + offsets


@dataclasses.dataclass
class Aligned:
    index: np.ndarray        # int64 [K]  source message index (-1 = none available)
    age_s: np.ndarray        # float64 [K] t_grid - t_msg (seconds); nan where index == -1
    index_next: np.ndarray | None = None   # for interp: the upper neighbour
    weight: np.ndarray | None = None       # for interp: weight of upper neighbour in [0,1]


def align(t_msg_ns: np.ndarray, t_grid_ns: np.ndarray, mode: str = "hold") -> Aligned:
    t_msg_ns = np.asarray(t_msg_ns, dtype=np.int64)
    t_grid_ns = np.asarray(t_grid_ns, dtype=np.int64)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if t_msg_ns.size == 0:
        k = t_grid_ns.size
        return Aligned(np.full(k, -1, np.int64), np.full(k, np.nan))
    if np.any(np.diff(t_msg_ns) < 0):
        raise ValueError("message timestamps must be non-decreasing (sort by log_time first)")
    n = t_msg_ns.size
    # hold: last index with t_msg <= t_grid
    hi = np.searchsorted(t_msg_ns, t_grid_ns, side="right") - 1      # -1 if none before
    # next: first index with t_msg >= t_grid
    lo = np.searchsorted(t_msg_ns, t_grid_ns, side="left")            # n if none after
    if mode == "hold":
        idx = hi
    elif mode == "next":
        idx = np.where(lo < n, lo, -1)
    elif mode == "nearest":
        cand_h = np.where(hi >= 0, hi, 0)
        cand_n = np.where(lo < n, lo, n - 1)
        dh = np.abs(t_grid_ns - t_msg_ns[cand_h]); dn = np.abs(t_msg_ns[cand_n] - t_grid_ns)
        idx = np.where(dn < dh, cand_n, cand_h)
    else:  # interp
        i0 = np.clip(hi, 0, n - 1); i1 = np.clip(lo, 0, n - 1)
        span = (t_msg_ns[i1] - t_msg_ns[i0]).astype(np.float64)
        w = np.where(span > 0, (t_grid_ns - t_msg_ns[i0]) / np.where(span > 0, span, 1), 0.0)
        w = np.clip(w, 0.0, 1.0)
        valid = (hi >= 0) & (lo < n)
        age = np.where(valid, (t_grid_ns - t_msg_ns[i0]) / 1e9, np.nan)
        return Aligned(np.where(valid, i0, -1), age, np.where(valid, i1, -1), np.where(valid, w, np.nan))
    valid = idx >= 0
    age = np.where(valid, (t_grid_ns - t_msg_ns[np.clip(idx, 0, n - 1)]) / 1e9, np.nan)
    return Aligned(np.where(valid, idx, -1), age)


def gather(values: np.ndarray, al: Aligned, fill=np.nan) -> np.ndarray:
    """values [N, D] -> [K, D] following an alignment (interp blends the two neighbours)."""
    values = np.asarray(values)
    K = al.index.size
    out = np.full((K,) + values.shape[1:], fill, dtype=np.result_type(values.dtype, np.float64))
    ok = al.index >= 0
    if al.index_next is None:
        out[ok] = values[al.index[ok]]
    else:
        w = al.weight[ok][:, None] if values.ndim > 1 else al.weight[ok]
        out[ok] = (1 - w) * values[al.index[ok]] + w * values[al.index_next[ok]]
    return out


def gap_report(t_msg_ns: np.ndarray, expected_hz: float | None = None, factor: float = 2.0) -> dict:
    """Inter-message gap statistics for a stream (seconds)."""
    t = np.asarray(t_msg_ns, dtype=np.int64)
    if t.size < 2:
        return dict(n=int(t.size))
    dt = np.diff(t) / 1e9
    med = float(np.median(dt))
    thr = factor * (1.0 / expected_hz if expected_hz else med)
    big = np.flatnonzero(dt > thr)
    return dict(n=int(t.size), rate_hz=float((t.size - 1) / ((t[-1] - t[0]) / 1e9)), dt_median_s=med,
                dt_p99_s=float(np.percentile(dt, 99)), dt_max_s=float(dt.max()), gap_threshold_s=float(thr),
                n_gaps=int(big.size), gaps=[(float((t[i] - t[0]) / 1e9), float(dt[i])) for i in big[:50]],
                n_nonmonotonic=int((dt < 0).sum()))
