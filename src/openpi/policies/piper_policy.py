import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_aloha_example() -> dict:
    """Creates a random input example for the Aloha policy."""
    return {
        "observation/state": np.random.rand(20),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """Inputs for the Aloha policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [20] eef_pose
    - actions: [action_horizon, 20]
    """
    model_type: _model.ModelType

    # Convert joint angles to [-pi, pi], for joint_position, not used here
    adapt_to_pi: bool = True

    # All three camera keys are required and are marked valid.

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist_image = _parse_image(data["observation/left_wrist_image"])
        right_wrist_image = _parse_image(data["observation/right_wrist_image"])
        state = data["state"] if "state" in data else data["observation/state"]

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "action_loss_mask" in data:
            inputs["action_loss_mask"] = data["action_loss_mask"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):

    adapt_to_pi: bool = True
    def __call__(self, data: dict) -> dict:
        # Return the two physical 10D EEF actions, excluding model padding.
        return {"actions": np.asarray(data["actions"][:, :20])}
