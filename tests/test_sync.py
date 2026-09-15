import numpy as np
import pytest
from dexteleop.sync import make_grid, align, gather, gap_report

S = 1_000_000_000  # ns


def test_make_grid_inclusive_and_exact():
    g = make_grid(10 * S, 12 * S, 4.0)
    assert g.tolist() == [10 * S + i * 250_000_000 for i in range(9)]
    g15 = make_grid(0, 1 * S, 15.0)
    assert g15.size == 16 and g15[-1] == S and g15[1] == 66_666_667


def test_hold_and_next_and_age():
    t = np.array([0, 100, 200, 300]) * 1_000_000  # 0,0.1,0.2,0.3 s
    grid = np.array([50, 100, 250, 400]) * 1_000_000
    h = align(t, grid, "hold")
    assert h.index.tolist() == [0, 1, 2, 3]
    assert np.allclose(h.age_s, [0.05, 0.0, 0.05, 0.1])
    n = align(t, grid, "next")
    assert n.index.tolist() == [1, 1, 3, -1]
    assert np.allclose(n.age_s[:3], [-0.05, 0.0, -0.05]) and np.isnan(n.age_s[3])


def test_hold_before_first_message_is_missing():
    t = np.array([100, 200]) * 1_000_000
    h = align(t, np.array([0, 150]) * 1_000_000, "hold")
    assert h.index.tolist() == [-1, 0] and np.isnan(h.age_s[0])


def test_nearest_ties_and_edges():
    t = np.array([0, 100]) * 1_000_000
    n = align(t, np.array([-10, 40, 50, 60, 500]) * 1_000_000, "nearest")
    assert n.index.tolist() == [0, 0, 0, 1, 1]  # exact tie -> earlier (hold) neighbour


def test_interp_weights_and_gather():
    t = np.array([0, 100]) * 1_000_000
    v = np.array([[0.0, 10.0], [1.0, 20.0]])
    a = align(t, np.array([25, 100, 150]) * 1_000_000, "interp")
    g = gather(v, a)
    assert np.allclose(g[0], [0.25, 12.5]) and np.allclose(g[1], [1.0, 20.0]) and np.isnan(g[2]).all()


def test_gather_hold_fills_missing_with_nan():
    t = np.array([100]) * 1_000_000
    a = align(t, np.array([0, 100]) * 1_000_000, "hold")
    g = gather(np.array([[7.0]]), a)
    assert np.isnan(g[0, 0]) and g[1, 0] == 7.0


def test_rejects_unsorted():
    with pytest.raises(ValueError):
        align(np.array([2, 1]), np.array([0]))


def test_gap_report_flags_gaps():
    t = np.array([0, 10, 20, 30, 80, 90]) * 1_000_000
    r = gap_report(t, expected_hz=100.0)
    assert r["n_gaps"] == 1 and r["gaps"][0][1] == pytest.approx(0.05) and r["n_nonmonotonic"] == 0


def test_grid_alignment_is_deterministic_and_index_preserving():
    rng = np.random.default_rng(0)
    t = np.sort(rng.integers(0, 10 * S, 500))
    grid = make_grid(0, 10 * S, 15.0)
    a1, a2 = align(t, grid, "hold"), align(t, grid, "hold")
    assert np.array_equal(a1.index, a2.index)
    ok = a1.index >= 0
    assert np.all(t[a1.index[ok]] <= grid[ok])                       # hold never looks ahead
    nxt = np.minimum(a1.index[ok] + 1, t.size - 1)
    assert np.all((t[nxt] > grid[ok]) | (a1.index[ok] == t.size - 1))  # and picks the LATEST such message


def test_make_grid_exact_at_epoch_scale():
    start = 1_787_261_039_284_581_150 + 125 * S          # real MCAP epoch-ns start + 125 s
    g = make_grid(start, start + 85 * S, 15.0)
    d = np.unique(np.diff(g))
    assert g[0] == start and g.size == 1276
    assert set(d.tolist()) <= {66_666_666, 66_666_667, 66_666_668}   # ±1 ns rounding only
    assert g[-1] == start + 85 * S
    g45 = make_grid(start, start + 85 * S, 45.0)
    assert g45.size == 3826 and set(np.unique(np.diff(g45)).tolist()) <= {22_222_222, 22_222_223}
