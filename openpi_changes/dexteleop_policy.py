"""Input/output transforms for the DexTeleop / TeleAvatar-2 bimanual mobile humanoid (real robot).

Canonical data (dexteleop-vla repo, `dexteleop/schema.py`):
    state   19-D measured:  left arm q[7], right arm q[7], left/right gripper position, base motor vel[3]
    actions 19-D commanded: left arm q_target[7], right arm q_target[7], left/right raw gripper cmd
                            (joint_cmd.effort field), base motor vel cmd[3]
Both are padded to the model's 32-D by `PadStatesAndActions` in the model transforms; we slice back to 19.

Cameras -> pi0.5 image slots (all three are REAL and unmasked):
    head  (/xr_video_topic/ffmpeg, left eye)        -> base_0_rgb
    left  (/left/color/image_raw/ffmpeg, left eye)  -> left_wrist_0_rgb
    right (/right/color/image_raw/ffmpeg, left eye) -> right_wrist_0_rgb
No synthetic cue inpainting: raw camera observations + natural-language prompt only.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

DEXTELEOP_ACTION_DIM = 19
DEXTELEOP_STATE_DIM = 19


def make_dexteleop_example() -> dict:
    return {
        "observation/state": np.random.rand(DEXTELEOP_STATE_DIM),
        "observation/head_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "Insert the red cylinder onto the vertical peg.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexTeleopInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/head_image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class DexTeleopOutputs(transforms.DataTransformFn):
    action_dim: int = DEXTELEOP_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
