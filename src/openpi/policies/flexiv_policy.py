import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_flexiv_example() -> dict:
    """Creates a random input example for the Flexiv policy eef_pose."""
    return {
        "observation/state": np.random.rand(10),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/secondary_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class FlexivInputs(transforms.DataTransformFn):
    """
    Three cameras + eef_pose-state
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        secondary_raw = data.get("observation/secondary_image")
        if secondary_raw is None:
            secondary_image = np.zeros_like(base_image)
            secondary_image_mask = np.False_
        else:
            secondary_image = _parse_image(secondary_raw)
            secondary_image_mask = np.True_
        state = data["state"] if "state" in data else data["observation/state"]

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": secondary_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": secondary_image_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FlexivOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # returns eef_pose output [x,y,z,r1,...,r6,left_gripper]
        return {"actions": np.asarray(data["actions"][:, :10])}
