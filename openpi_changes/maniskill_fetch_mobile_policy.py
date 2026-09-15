"""Input/output transforms for ManiSkill Fetch MOBILE pick-drive-place (shoulder + wrist cameras).

Unlike the static-base Fetch task (maniskill_fetch_policy.py, 7-dim arm+gripper), the base MOVES here,
so the policy must output the base velocities too. The converter keeps action columns
[ee Δpose 6, gripper 1, base_forward_vel, base_rotate_vel] = 9 dims (the 3 body/head/torso columns are
dropped via --action-cols 0,1,2,3,4,5,6,10,11). At eval, map this 9-dim output back to the env's
12-dim pd_ee_delta_pose action by inserting zeros for the 3 body columns: [ee6, grip1, 0,0,0, fwd,ang].

Cameras (mapped by the LeRobot converter): fetch_shoulder -> base_0_rgb, fetch_hand -> left_wrist_0_rgb.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# [ee Δpos 3, ee Δrot 3, gripper 1, base_forward_vel, base_rotate_vel]
MANISKILL_FETCH_MOBILE_ACTION_DIM = 9


def make_maniskill_fetch_mobile_example() -> dict:
    return {
        "observation/state": np.random.rand(15),  # fetch qpos (arm+torso+gripper+base+head)
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick the red cube and place it on the green sign",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ManiSkillFetchMobileInputs(transforms.DataTransformFn):
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
class ManiSkillFetchMobileOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :MANISKILL_FETCH_MOBILE_ACTION_DIM])}
