"""Preview two converted Sharpa iiwa robots and converted DexMimicGen objects.

Example::

    ./isaaclab.sh -p /path/to/ACG/libs/IsaacLab/preview_dual_iiwa_objects.py \
        --left-robot-usd assets/converted/robots/iiwa14_left_sharpa.usd \
        --right-robot-usd assets/converted/robots/iiwa14_right_sharpa.usd \
        --object-usd assets/converted/objects/coffee_body.usd

This is intentionally a scene preview. It does not reuse the GR1T2 task,
because that task's actions and observations are tied to GR1T2 link names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left-robot-usd", type=Path, required=True)
parser.add_argument("--right-robot-usd", type=Path, required=True)
parser.add_argument("--object-usd", type=Path, action="append", default=[])
parser.add_argument("--object-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("X", "Y", "Z"))
parser.add_argument("--object-z", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; 0 runs until the app closes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


def require_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"USD file does not exist: {path}")
    return str(path)


def make_robot_cfg(usd_path: str, prim_path: str, position: tuple[float, float, float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={},
    )


def make_object_cfg(usd_path: str, prim_path: str, position: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=tuple(args_cli.object_scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def design_scene() -> tuple[dict[str, object], list[tuple[float, float, float]]]:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    left = Articulation(make_robot_cfg(require_file(args_cli.left_robot_usd), "/World/LeftRobot", (-0.55, 0.0, 0.0)))
    right = Articulation(make_robot_cfg(require_file(args_cli.right_robot_usd), "/World/RightRobot", (0.55, 0.0, 0.0)))
    entities: dict[str, object] = {"left_robot": left, "right_robot": right}
    object_origins = []
    for index, object_usd in enumerate(args_cli.object_usd):
        position = (-0.5 + 0.25 * index, 0.55, args_cli.object_z)
        name = f"object_{index}"
        entities[name] = RigidObject(make_object_cfg(require_file(object_usd), f"/World/{name}", position))
        object_origins.append(position)
    return entities, object_origins


def run_simulator(sim: SimulationContext, entities: dict[str, object]) -> None:
    left = entities["left_robot"]
    right = entities["right_robot"]
    step_count = 0
    while simulation_app.is_running():
        for robot in (left, right):
            robot.set_joint_position_target(robot.data.default_joint_pos)
            robot.write_data_to_sim()
        for name, entity in entities.items():
            if name.startswith("object_"):
                entity.write_data_to_sim()
        sim.step()
        step_count += 1
        for entity in entities.values():
            entity.update(sim.get_physics_dt())
        if args_cli.steps and step_count >= args_cli.steps:
            break


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(eye=[2.5, -3.0, 2.2], target=[0.0, 0.4, 1.0])
    entities, _ = design_scene()
    sim.reset()
    for entity in entities.values():
        entity.reset()
    print("[INFO] Dual iiwa and object scene is ready.")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()