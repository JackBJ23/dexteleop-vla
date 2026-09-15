"""Input/output transforms for ManiSkill Fetch place-cube-on-sign (egocentric head+wrist).

Same as maniskill_policy.py but with the Fetch action dimension. Fetch pd_ee_delta_pose action:
    [dx, dy, dz, d_rx, d_ry, d_rz, gripper, base_forward_vel, base_rotate_vel]  (9 dims)
The base columns are ~0 in the demos (the base does not move), but they are part of the action
vector, so the model must output 9 dims and we slice back to 9.

Cameras (mapped by the LeRobot converter): fetch_head -> base_0_rgb, fetch_hand -> left_wrist_0_rgb.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Fetch pd_ee_delta_pose is 12-dim (arm 6 + gripper 1 + base 5), but the base columns are constant
# ~0 in static-base demos and their ~0 variance destroys action normalization (loss blows up). We
# train on the 7 real dims (arm+gripper) only — same as Panda. The dataset is converted with
# --action-dim 7, so actions here are already 7-dim. (Add base dims back when the base actually moves.)
MANISKILL_FETCH_ACTION_DIM = 7


def make_maniskill_fetch_example() -> dict:
    return {
        "observation/state": np.random.rand(15),  # fetch qpos (arm+torso+gripper+base+head)
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "place the red cube on the green sign",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ManiSkillFetchInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class ManiSkillFetchOutputs(transforms.DataTransformFn):
    # number of real action dims to return at inference. 7 (arm+gripper) for the static tasks; 10
    # (arm+gripper+base x/y/yaw) for the mobile move-pick-move-place policy that drives the base.
    action_dim: int = MANISKILL_FETCH_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :self.action_dim])}
