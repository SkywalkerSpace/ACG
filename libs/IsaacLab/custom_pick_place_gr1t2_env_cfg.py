# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 pick-place task with the two local ACG assets on the table."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import (
    PickPlaceGR1T2EnvCfg,
    PickPlaceGR1T2SceneCfg,
)


_ASSET_DIR = Path(__file__).resolve().parent / "assets"
_HOLE_USD = _ASSET_DIR / "hole_tol0p25mm_right_training_graspable.usd"
_PEG_USD = _ASSET_DIR / "lpeg_matchedmass.usd"

# Adjust these world-space coordinates [m] to move the assets on the tabletop.
# X/Y set the tabletop location and Z sets the root prim height. Restart the
# environment after editing; the reset event returns each object to this pose.
HOLE_POSITION = (-0.35, 0.25, 1.12)
PEG_POSITION = (-0.05, 0.25, 1.02)


@configclass
class CollectSceneCfg(PickPlaceGR1T2SceneCfg):
    """GR1T2 pick-place scene with the hole fixture and peg."""

    # Operated object; this retains the stock observation, reset, and success API.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=HOLE_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_HOLE_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=PEG_POSITION),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_PEG_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Stock sensors filter against a nested steering wheel prim. The imported
    # assets author their rigid body at their root, so target the fixture root.
    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*L_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*R_(index|middle|ring|pinky|thumb)[^/]*_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        update_period=0.0,
        history_length=3,
    )


@configclass
class CollectPickPlaceGR1T2EnvCfg(PickPlaceGR1T2EnvCfg):
    """Stock GR1T2 XR pick-place workflow using the local fixture and peg."""

    scene: CollectSceneCfg = CollectSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        # Contact sensors in the inherited scene now measure the fixture at Object.
        self.haptic_feedback.left_sensor_name = "left_hand_contact"
        self.haptic_feedback.right_sensor_name = "right_hand_contact"
        # Avoid reporting success from the stock bin-placement predicate: this
        # scene is intended for collecting teleop demonstrations with the new parts.
        self.terminations.success = None
