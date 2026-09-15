"""End-to-end check of the 19-D assembly on synthetic streams (no MCAP needed)."""
import numpy as np
from dexteleop import schema as S
from dexteleop.extract import assemble, check_joint_names
from dexteleop.mcap_io import JointStateStream
from dexteleop.sync import make_grid

NS = 1_000_000_000


def _js(topic, names, t_s, pos=None, vel=None, eff=None):
    t = (np.asarray(t_s) * NS).astype(np.int64); N, J = t.size, len(names)
    z = np.zeros((N, J))
    return JointStateStream(topic, names, t, t, t, z if pos is None else pos, z if vel is None else vel, z if eff is None else eff)


def _streams():
    t = np.arange(0, 2.0, 0.01)                      # 100 Hz, 2 s
    N = t.size
    st = {}
    la = np.outer(np.ones(N), np.arange(7)) + t[:, None]            # value = joint index + time
    st[S.T_LEFT_ARM_STATE] = _js(S.T_LEFT_ARM_STATE, S.ARM_JOINT_NAMES["left"], t, pos=la, vel=-la, eff=2 * la)
    st[S.T_RIGHT_ARM_STATE] = _js(S.T_RIGHT_ARM_STATE, S.ARM_JOINT_NAMES["right"], t, pos=la + 100)
    st[S.T_LEFT_ARM_CMD] = _js(S.T_LEFT_ARM_CMD, S.ARM_JOINT_NAMES["left"], t, pos=la + 1000)
    st[S.T_RIGHT_ARM_CMD] = _js(S.T_RIGHT_ARM_CMD, S.ARM_JOINT_NAMES["right"], t, pos=la + 2000)
    g = t[:, None]
    st[S.T_LEFT_GRIP_STATE] = _js(S.T_LEFT_GRIP_STATE, ["l_joint8"], t, pos=g + 5, vel=g + 6, eff=g + 7)
    st[S.T_RIGHT_GRIP_STATE] = _js(S.T_RIGHT_GRIP_STATE, ["r_joint8"], t, pos=g + 15, vel=g + 16, eff=g + 17)
    st[S.T_LEFT_GRIP_CMD] = _js(S.T_LEFT_GRIP_CMD, ["l_joint8"], t, eff=g + 25)        # cmd rides in effort
    st[S.T_RIGHT_GRIP_CMD] = _js(S.T_RIGHT_GRIP_CMD, ["r_joint8"], t, eff=g + 35)
    b = np.outer(np.ones(N), np.arange(3)) + t[:, None]
    st[S.T_BASE_STATE] = _js(S.T_BASE_STATE, S.BASE_STATE_NAMES, t, pos=b + 50, vel=b + 60, eff=b + 70)
    st[S.T_BASE_CMD] = _js(S.T_BASE_CMD, S.BASE_CMD_NAMES, t, vel=b + 80)
    return st


def test_assemble_picks_exact_topic_field_index():
    st = _streams(); check_joint_names(st)
    grid = make_grid(int(0.5 * NS), int(1.5 * NS), 10.0)
    P, _ = assemble(S.PROPRIO_DIMS, st, grid, "hold")
    A, _ = assemble(S.ACTION_DIMS, st, grid, "hold")
    tg = grid / NS
    assert np.allclose(P[:, 0:7], np.arange(7) + tg[:, None])                # left arm measured position
    assert np.allclose(P[:, 7:14], np.arange(7) + 100 + tg[:, None])
    assert np.allclose(P[:, 14], tg + 5) and np.allclose(P[:, 15], tg + 15)  # gripper POSITION, not effort
    assert np.allclose(P[:, 16:19], np.arange(3) + 60 + tg[:, None])         # base measured VELOCITY
    assert np.allclose(A[:, 0:7], np.arange(7) + 1000 + tg[:, None])
    assert np.allclose(A[:, 7:14], np.arange(7) + 2000 + tg[:, None])
    assert np.allclose(A[:, 14], tg + 25) and np.allclose(A[:, 15], tg + 35)  # gripper cmd from EFFORT field
    assert np.allclose(A[:, 16:19], np.arange(3) + 80 + tg[:, None])         # base cmd velocity


def test_action_next_vs_hold_semantics():
    st = _streams()
    grid = np.array([0.505 * NS], dtype=np.int64)
    A_hold, al_h = assemble(S.ACTION_DIMS, st, grid, "hold")
    A_next, al_n = assemble(S.ACTION_DIMS, st, grid, "next")
    assert np.isclose(A_hold[0, 0], 1000 + 0.50) and np.isclose(A_next[0, 0], 1000 + 0.51)
    assert al_h[S.T_LEFT_ARM_CMD].age_s[0] > 0 > al_n[S.T_LEFT_ARM_CMD].age_s[0]


def test_joint_name_order_guard():
    st = _streams()
    st[S.T_LEFT_ARM_STATE].names = list(reversed(st[S.T_LEFT_ARM_STATE].names))
    import pytest
    with pytest.raises(ValueError):
        check_joint_names(st)
