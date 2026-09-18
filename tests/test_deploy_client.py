import sys, numpy as np, pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
import dexteleop_client as C


def test_trigger_curve_roundtrip_and_endpoints():
    for e in (2.0, 1.0, 0.0, -0.5, -1.6):
        assert C.trigger_to_effort(C.effort_to_trigger(e)) == pytest.approx(e, abs=1e-9)
    assert C.effort_to_trigger(2.0) == 0.0 and C.effort_to_trigger(-1.6) == 1.0     # open -> 0, fully closing -> 1


def test_action19_to_16_layout():
    a = np.arange(19, dtype=float); b = C.action19_to_action16(a)
    assert b[0:7].tolist() == list(range(0, 7)) and b[8:15].tolist() == list(range(7, 14))
    assert b[7] == 14 and b[15] == 15                       # raw gripper efforts pass through (their interface converts)


def test_safety_delta_clamp_and_right_hold():
    s = C.SafetyLayer(max_joint_delta=0.1, hold_right_arm=True, right_arm_hold_pose=np.full(7, 0.5))
    meas = np.zeros(14); a = np.zeros(19); a[0] = 0.3; a[7:14] = 9.0
    out, info = s.filter_step(a, meas)
    assert out[0] == pytest.approx(0.1) and info["clamped_joints"][0] == 0 and (out[7:14] == 0.5).all()
    out2, _ = s.filter_step(a, meas)                        # clamps against the LAST COMMAND, not the measurement
    assert out2[0] == pytest.approx(0.2)


def test_dry_run_replay_end_to_end(tmp_path):
    import os
    ep = Path(os.environ.get("DEXTELEOP_EPISODE", "/n/holylfs05/LABS/hankyang_lab/Lab/jackbjed/dexteleop-vla-data/canonical/rec_20260821_052219_d4f4fc76/t125.5-135_hz15"))
    if not ep.exists(): pytest.skip("canonical episode not available")
    import argparse
    args = argparse.Namespace(mode="replay", fork=None, episode=str(ep), prompt=None, host="", port=0, control_hz=15.0, interp_hz=200.0,
                              chunk=10, speed=1000.0, max_joint_delta=0.5, hold_right_arm=False, max_chunks=0, mode_confirm_replay=False, go_to_start=False,
                              dry_run=True, log=str(tmp_path / "log.json"))
    robot = C.run(args)
    A = np.load(ep / "episode.npz")["action"]
    sent = np.stack(robot.published)
    assert sent.shape == A.shape and np.allclose(sent[:, 14:], A[:, 14:])   # grippers/base untouched
    assert np.abs(sent[:, :14] - A[:, :14]).max() < 0.5 + 1e-6              # arms only ever delta-clamped
