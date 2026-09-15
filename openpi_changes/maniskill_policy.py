"""Input/output transforms for ManiSkill StackCube (Panda, pd_ee_delta_pose).

Copied from libero_policy.py — our LeRobot dataset (see pi/convert_maniskill_to_lerobot.py)
deliberately uses the same flat keys (image / wrist_image / state / actions / prompt), so the
only ManiSkill-specific choice is the action dimension returned by the outputs transform.

ManiSkill Panda pd_ee_delta_pose action layout (7 dims):
    [dx, dy, dz, d_axisangle_x, d_axisangle_y, d_axisangle_z, gripper]
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Panda pd_ee_delta_pose action dimension. Change if you switch control mode.
MANISKILL_ACTION_DIM = 7


def make_maniskill_example() -> dict:
    """Random input example matching the inference observation dict."""
    return {
        "observation/state": np.random.rand(9),  # panda qpos (7 arm + 2 finger)
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "stack the red cube on top of the green cube",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class ManiSkillInputs(transforms.DataTransformFn):
    """Pipe ManiSkill observations into the model. Used for training and inference."""

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
class ManiSkillOutputs(transforms.DataTransformFn):
    """Slice the padded model action back to the ManiSkill action dimension (inference only)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :MANISKILL_ACTION_DIM])}
