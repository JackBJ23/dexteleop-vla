import numpy as np
from dexteleop import schema as S


def test_dims_are_19():
    assert len(S.PROPRIO_DIMS) == 19 and len(S.ACTION_DIMS) == 19


def test_action_layout_matches_spec():
    a = S.ACTION_DIMS
    for i in range(7):
        assert a[i].topic == "/left_arm/joint_cmd" and a[i].field == "position" and a[i].index == i
        assert a[7 + i].topic == "/right_arm/joint_cmd" and a[7 + i].field == "position" and a[7 + i].index == i
    assert (a[14].topic, a[14].field, a[14].index) == ("/left_gripper/joint_cmd", "effort", 0)
    assert (a[15].topic, a[15].field, a[15].index) == ("/right_gripper/joint_cmd", "effort", 0)
    for i in range(3):
        assert (a[16 + i].topic, a[16 + i].field, a[16 + i].index) == ("/chassis/joint_cmd", "velocity", i)


def test_proprio_layout():
    p = S.PROPRIO_DIMS
    assert all(d.topic == "/left_arm/joint_states" and d.field == "position" for d in p[0:7])
    assert all(d.topic == "/right_arm/joint_states" and d.field == "position" for d in p[7:14])
    assert (p[14].topic, p[14].field) == ("/left_gripper/joint_states", "position")
    assert (p[15].topic, p[15].field) == ("/right_gripper/joint_states", "position")
    assert all(d.topic == "/chassis/joint_states" and d.field == "velocity" for d in p[16:19])
    assert [d.index for d in p[16:19]] == [0, 1, 2]


def test_no_lift_and_no_effort_in_proprio():
    assert not any("kinco" in d.topic for d in S.ACTION_DIMS + S.PROPRIO_DIMS)
    assert not any(d.field == "effort" for d in S.PROPRIO_DIMS)


def test_legacy72_index_maps_reproduce_named_layout():
    # legacy 72-D layout: [pos 0:16 | vel 16:32 | effort 32:48 | ee 48:62 | chassis pos/vel/eff 62:71 | kinco 71]
    names = ([f"left_joint{i}_position" for i in range(1, 8)] + ["left_gripper_position"]
             + [f"right_joint{i}_position" for i in range(1, 8)] + ["right_gripper_position"])
    names += [n.replace("position", "velocity") for n in names] + [n.replace("position", "effort") for n in names[:16]]
    names += [f"{s}_ee_{c}" for s in ("left", "right") for c in ("px", "py", "pz", "qx", "qy", "qz", "qw")]
    names += [f"chassis_motor{i}_{f}" for f in ("position", "velocity", "effort") for i in (1, 2, 3)] + ["kinco_velocity"]
    assert len(names) == 72
    act = [names[i] for i in S.LEGACY72_ACTION_INDEX]
    assert act[:7] == [f"left_joint{i}_position" for i in range(1, 8)]
    assert act[7:14] == [f"right_joint{i}_position" for i in range(1, 8)]
    assert act[14:16] == ["left_gripper_effort", "right_gripper_effort"]
    assert act[16:] == ["chassis_motor1_velocity", "chassis_motor2_velocity", "chassis_motor3_velocity"]
    pro = [names[i] for i in S.LEGACY72_PROPRIO_INDEX]
    assert pro[14:16] == ["left_gripper_position", "right_gripper_position"]


def test_slices_cover_vector_once():
    idx = list(range(19))
    cover = idx[S.LEFT_ARM] + idx[S.RIGHT_ARM] + [S.LEFT_GRIP, S.RIGHT_GRIP] + idx[S.BASE]
    assert sorted(cover) == idx
