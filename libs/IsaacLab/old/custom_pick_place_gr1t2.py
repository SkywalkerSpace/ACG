"""Run the waist-enabled GR1T2 pick-place scene with a custom object.

Examples (run from the IsaacLab repository)::

    ./isaaclab.sh -p scripts/demos/custom_pick_place_gr1t2.py \
        --object-usd /path/to/my_object.usd --object-scale 0.5 0.5 0.5

The task's robot, actions, observations and reset logic are kept unchanged;
only the rigid object in the scene is replaced.
"""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--object-usd", required=True, help="USD file to use as the pick-place object.")
parser.add_argument(
    "--object-scale",
    type=float,
    nargs=3,
    default=(0.75, 0.75, 0.75),
    metavar=("X", "Y", "Z"),
    help="Object scale (default: 0.75 0.75 0.75).",
)
parser.add_argument(
    "--object-pos",
    type=float,
    nargs=3,
    default=(-0.45, 0.45, 0.9996),
    metavar=("X", "Y", "Z"),
    help="Object position in the environment (default: -0.45 0.45 0.9996).",
)
parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab_tasks  # noqa: F401  # registers the contrib tasks
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_waist_enabled_env_cfg import (
    PickPlaceGR1T2WaistEnabledEnvCfg,
)


def make_custom_cfg() -> PickPlaceGR1T2WaistEnabledEnvCfg:
    """Create the stock waist-enabled config and replace its scene object."""
    env_cfg = PickPlaceGR1T2WaistEnabledEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.object.spawn = UsdFileCfg(
        usd_path=args_cli.object_usd,
        scale=tuple(args_cli.object_scale),
    )
    env_cfg.scene.object.init_state.pos = tuple(args_cli.object_pos)
    # The stock config filters contacts against a steering-wheel body nested
    # inside the original USD.  Custom assets normally expose their rigid body
    # at the Object root, so update both hand sensors accordingly.
    object_contact_path = "{ENV_REGEX_NS}/Object"
    env_cfg.scene.left_hand_contact.filter_prim_paths_expr = [object_contact_path]
    env_cfg.scene.right_hand_contact.filter_prim_paths_expr = [object_contact_path]
    return env_cfg


def main() -> None:
    env = ManagerBasedRLEnv(cfg=make_custom_cfg())
    try:
        obs, _ = env.reset()
        for _ in range(args_cli.steps):
            # Zero actions keep this script useful as a scene/object preview.
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, _, terminated, truncated, _ = env.step(actions)
            if bool(torch.any(terminated | truncated)):
                env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
