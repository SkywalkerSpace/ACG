"""GR1T2 waist-enabled pick-place configuration with an externally selected object.

The following optional environment variables are read when the environment is
created:

``ISAACLAB_PICK_PLACE_OBJECT_USD``
    USD path for the object. Defaults to the stock steering wheel.
``ISAACLAB_PICK_PLACE_OBJECT_SCALE``
    Three comma-separated scale values, for example ``0.5,0.5,0.5``.
``ISAACLAB_PICK_PLACE_OBJECT_POS``
    Three comma-separated world coordinates, for example ``-0.45,0.45,1.0``.
"""

"""
__init__.py

gym.register(
    id="IsaacContrib-PickPlace-GR1T2-WaistEnabled-CustomObject-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.custom_pick_place_gr1t2_env_cfg:CustomObjectPickPlaceGR1T2WaistEnabledEnvCfg"
        ),
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)
"""

import os

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from .pickplace_gr1t2_waist_enabled_env_cfg import PickPlaceGR1T2WaistEnabledEnvCfg


def _vector_from_env(name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Read a three-value comma-separated vector from an environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        vector = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"{name} must contain three comma-separated numbers, got {value!r}.") from error
    if len(vector) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {value!r}.")
    return vector


@configclass
class CustomObjectPickPlaceGR1T2WaistEnabledEnvCfg(PickPlaceGR1T2WaistEnabledEnvCfg):
    """Stock waist-enabled environment with a configurable rigid object."""

    def __post_init__(self):
        super().__post_init__()

        usd_path = os.getenv(
            "ISAACLAB_PICK_PLACE_OBJECT_USD",
            f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
        )
        self.scene.object.spawn = self.scene.object.spawn.replace(
            usd_path=usd_path,
            scale=_vector_from_env("ISAACLAB_PICK_PLACE_OBJECT_SCALE", (0.75, 0.75, 0.75)),
        )
        self.scene.object.init_state.pos = _vector_from_env(
            "ISAACLAB_PICK_PLACE_OBJECT_POS", (-0.45, 0.45, 0.9996)
        )

        # The stock sensors target the steering wheel's nested rigid prim. A
        # custom rigid-object USD is expected to author its body at Object.
        object_contact_path = "{ENV_REGEX_NS}/Object"
        self.scene.left_hand_contact.filter_prim_paths_expr = [object_contact_path]
        self.scene.right_hand_contact.filter_prim_paths_expr = [object_contact_path]
