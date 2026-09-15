"""Canonical DexTeleop / TeleAvatar-2 episode schema: the 19-D proprio and 19-D action vectors and
exactly which MCAP topic/field each dimension is read from.

DO NOT change the action definition without an explicit decision (see CLAUDE.md §5).
"""
from __future__ import annotations

import dataclasses

SCHEMA_VERSION = "canonical-v0.1"

# ---- topics ---------------------------------------------------------------------------------------
T_LEFT_ARM_STATE = "/left_arm/joint_states"
T_RIGHT_ARM_STATE = "/right_arm/joint_states"
T_LEFT_GRIP_STATE = "/left_gripper/joint_states"
T_RIGHT_GRIP_STATE = "/right_gripper/joint_states"
T_BASE_STATE = "/chassis/joint_states"
T_LEFT_ARM_CMD = "/left_arm/joint_cmd"
T_RIGHT_ARM_CMD = "/right_arm/joint_cmd"
T_LEFT_GRIP_CMD = "/left_gripper/joint_cmd"
T_RIGHT_GRIP_CMD = "/right_gripper/joint_cmd"
T_BASE_CMD = "/chassis/joint_cmd"
T_LIFT_CMD = "/kinco/cmd_velocity"

CAMERA_TOPICS = {
    "left": "/left/color/image_raw/ffmpeg",
    "right": "/right/color/image_raw/ffmpeg",
    "head": "/xr_video_topic/ffmpeg",
}

ARM_JOINT_NAMES = {"left": [f"l_joint{i}" for i in range(1, 8)], "right": [f"r_joint{i}" for i in range(1, 8)]}
GRIPPER_JOINT_NAME = {"left": "l_joint8", "right": "r_joint8"}
BASE_CMD_NAMES = ["chassis_motor1", "chassis_motor2", "chassis_motor3"]
BASE_STATE_NAMES = ["chassis_joint1", "chassis_joint2", "chassis_joint3"]


@dataclasses.dataclass(frozen=True)
class Dim:
    name: str
    topic: str
    field: str          # JointState field: position | velocity | effort
    index: int          # index within that field's array


def _arm(side: str, topic: str, field: str, suffix: str) -> list[Dim]:
    return [Dim(f"{side}_joint{i+1}_{suffix}", topic, field, i) for i in range(7)]


# ---- 19-D PROPRIO (measured) ------------------------------------------------------------------------
PROPRIO_DIMS: tuple[Dim, ...] = tuple(
    _arm("left", T_LEFT_ARM_STATE, "position", "pos")            # 0:7
    + _arm("right", T_RIGHT_ARM_STATE, "position", "pos")        # 7:14
    + [Dim("left_gripper_pos", T_LEFT_GRIP_STATE, "position", 0),    # 14
       Dim("right_gripper_pos", T_RIGHT_GRIP_STATE, "position", 0)]  # 15
    + [Dim(f"base_motor{i+1}_vel", T_BASE_STATE, "velocity", i) for i in range(3)]  # 16:19
)

# ---- 19-D ACTION (commanded) ------------------------------------------------------------------------
ACTION_DIMS: tuple[Dim, ...] = tuple(
    _arm("left", T_LEFT_ARM_CMD, "position", "pos_cmd")          # 0:7
    + _arm("right", T_RIGHT_ARM_CMD, "position", "pos_cmd")      # 7:14
    + [Dim("left_gripper_cmd_raw", T_LEFT_GRIP_CMD, "effort", 0),     # 14  (cmd carried in effort field)
       Dim("right_gripper_cmd_raw", T_RIGHT_GRIP_CMD, "effort", 0)]   # 15
    + [Dim(f"base_motor{i+1}_vel_cmd", T_BASE_CMD, "velocity", i) for i in range(3)]  # 16:19
)

# Auxiliary (NOT in the canonical 19-D; extracted alongside for later ablations)
AUX_DIMS: tuple[Dim, ...] = (
    Dim("left_gripper_effort_meas", T_LEFT_GRIP_STATE, "effort", 0),
    Dim("right_gripper_effort_meas", T_RIGHT_GRIP_STATE, "effort", 0),
    Dim("left_gripper_vel_meas", T_LEFT_GRIP_STATE, "velocity", 0),
    Dim("right_gripper_vel_meas", T_RIGHT_GRIP_STATE, "velocity", 0),
)

PROPRIO_DIM = len(PROPRIO_DIMS)
ACTION_DIM = len(ACTION_DIMS)
assert PROPRIO_DIM == 19 and ACTION_DIM == 19

# slices, for readers
LEFT_ARM = slice(0, 7); RIGHT_ARM = slice(7, 14); LEFT_GRIP = 14; RIGHT_GRIP = 15; BASE = slice(16, 19)

# Mapping from the legacy 72-D DexTeleop vector (prior Mac-side npz) used ONLY for regression checks.
LEGACY72_ACTION_INDEX = list(range(0, 7)) + list(range(8, 15)) + [39, 47] + [65, 66, 67]
LEGACY72_PROPRIO_INDEX = list(range(0, 7)) + list(range(8, 15)) + [7, 15] + [65, 66, 67]


def topics_needed() -> set[str]:
    return {d.topic for d in PROPRIO_DIMS + ACTION_DIMS + AUX_DIMS} | set(CAMERA_TOPICS.values())


def dims_by_topic(dims: tuple[Dim, ...]) -> dict[str, list[tuple[int, Dim]]]:
    out: dict[str, list[tuple[int, Dim]]] = {}
    for i, d in enumerate(dims):
        out.setdefault(d.topic, []).append((i, d))
    return out
