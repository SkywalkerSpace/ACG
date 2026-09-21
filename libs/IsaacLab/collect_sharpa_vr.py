"""Collect dual-Sharpa demonstrations from OpenXR hand tracking.

This script deliberately does not use the GR1T2 task configuration. Each iiwa
arm has its own Pink IK controller and the Sharpa hand joints are held at their
initial pose until a hand-specific retargeter is added.

Example::

    ./isaaclab.sh -p /path/to/ACG/libs/IsaacLab/collect_sharpa_vr.py \
        --left-robot-usd assets/converted/robots/iiwa14_left_sharpa.usd \
        --right-robot-usd assets/converted/robots/iiwa14_right_sharpa.usd \
        --left-urdf /path/to/iiwa14_left_sharpa_adjusted_restricted.urdf \
        --right-urdf /path/to/iiwa14_right_sharpa_mirror_restricted.urdf \
        --object-usd assets/converted/objects/coffee_body.usd \
        --dataset-file ./datasets/sharpa_vr_demos.hdf5

The ``same`` mapping mode shares position scale and offset between both arms.
The ``different`` mode accepts independent left/right calibration values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--left-robot-usd", type=Path, required=True)
parser.add_argument("--right-robot-usd", type=Path, required=True)
parser.add_argument("--left-urdf", type=Path, default=None)
parser.add_argument("--right-urdf", type=Path, default=None)
parser.add_argument("--object-usd", type=Path, action="append", default=[])
parser.add_argument("--object-z", type=float, default=1.0)
parser.add_argument(
    "--dataset-file",
    "--output",
    dest="dataset_file",
    type=Path,
    default=Path("datasets/sharpa_vr_demos.hdf5"),
    help="HDF5 dataset containing demo_0, demo_1, ... episodes.",
)
parser.add_argument("--replay", action="store_true", help="Replay an existing HDF5 episode instead of collecting VR data.")
parser.add_argument("--demo", type=int, default=0, help="Episode index to replay, for example 0 loads demo_0.")
parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many simulation steps; 0 means no limit.")
parser.add_argument("--mapping-mode", choices=("same", "different"), default="different")
parser.add_argument("--pose-scale", type=float, default=1.0, help="Shared position scale in same mode.")
parser.add_argument(
    "--pose-offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Shared world-position offset in same mode.",
)
parser.add_argument("--left-pose-scale", type=float, default=1.0)
parser.add_argument("--right-pose-scale", type=float, default=1.0)
parser.add_argument("--left-pose-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
parser.add_argument("--right-pose-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
parser.add_argument("--device-name", default="handtracking", help="Name used for the OpenXR device.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)
app_launcher_args["xr"] = True
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

from pink.tasks import FrameTask

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.pink_ik import PinkIKController, PinkIKControllerCfg
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.sim import SimulationContext
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
import isaaclab.utils.math as math_utils


ARM_JOINT_NAMES = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
]


class SharpaVrRetargeter(RetargeterBase):
    """Return left and right OpenXR wrist poses in the existing world frame."""

    def retarget(self, data: dict[DeviceBase.TrackingTarget, Any]) -> torch.Tensor:
        left = data[DeviceBase.TrackingTarget.HAND_LEFT]["wrist"]
        right = data[DeviceBase.TrackingTarget.HAND_RIGHT]["wrist"]
        return torch.tensor(np.concatenate((left, right)), dtype=torch.float32, device=self._sim_device)

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.HAND_TRACKING]


@dataclass
class SharpaVrRetargeterCfg(RetargeterCfg):
    """Configuration for the wrist-pose-only Sharpa retargeter."""

    retargeter_type: type[RetargeterBase] = SharpaVrRetargeter


def require_file(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def make_robot_cfg(usd_path: Path, prim_path: str, position: tuple[float, float, float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(require_file(usd_path)),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=position, joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={},
    )


def make_object_cfg(usd_path: Path, prim_path: str, position: tuple[float, float, float]) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(require_file(usd_path)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def design_scene() -> dict[str, object]:
    sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    entities: dict[str, object] = {
        "left_robot": Articulation(
            make_robot_cfg(args_cli.left_robot_usd, "/World/LeftRobot", (-0.55, 0.0, 0.0))
        ),
        "right_robot": Articulation(
            make_robot_cfg(args_cli.right_robot_usd, "/World/RightRobot", (0.55, 0.0, 0.0))
        ),
    }
    for index, object_usd in enumerate(args_cli.object_usd):
        position = (-0.5 + 0.25 * index, 0.55, args_cli.object_z)
        entities[f"object_{index}"] = RigidObject(
            make_object_cfg(object_usd, f"/World/Object_{index}", position)
        )
    return entities


def make_ik_controller(robot: Articulation, urdf_path: Path, frame_name: str) -> PinkIKController:
    joint_ids, joint_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    controller_cfg = PinkIKControllerCfg(
        urdf_path=str(require_file(urdf_path)),
        mesh_path=str(require_file(urdf_path).parent),
        variable_input_tasks=[
            FrameTask(
                frame_name,
                position_cost=8.0,
                orientation_cost=1.0,
                lm_damping=12.0,
                gain=0.5,
            )
        ],
        fixed_input_tasks=[],
        joint_names=joint_names,
        all_joint_names=robot.data.joint_names,
        articulation_name="robot",
        base_link_name="iiwa14_link_0",
        show_ik_warnings=False,
        fail_on_joint_limit_violation=False,
        xr_enabled=True,
    )
    return PinkIKController(
        cfg=controller_cfg,
        robot_cfg=robot.cfg,
        device=robot.device,
        controlled_joint_indices=joint_ids,
    )


def map_pose(pose: torch.Tensor, scale: float, offset: tuple[float, float, float]) -> torch.Tensor:
    mapped = pose.clone()
    mapped[:3] = mapped[:3] * scale + torch.tensor(offset, device=pose.device)
    return mapped


def update_task_target(controller: PinkIKController, pose: torch.Tensor) -> None:
    task = controller.cfg.variable_input_tasks[0]
    target = task.transform_target_to_world
    target.translation = pose[:3].detach().cpu().numpy()
    target.rotation = math_utils.matrix_from_quat(pose[3:].unsqueeze(0))[0].detach().cpu().numpy()
    task.set_target(target)


def make_device(callbacks: dict[str, Any]) -> object:
    cfg = OpenXRDeviceCfg(
        retargeters=[SharpaVrRetargeterCfg(sim_device=args_cli.device)],
        sim_device=args_cli.device,
        xr_cfg=XrCfg(anchor_pos=(0.0, 0.0, 0.0), anchor_rot=(1.0, 0.0, 0.0, 0.0)),
    )
    return create_teleop_device(args_cli.device_name, {args_cli.device_name: cfg}, callbacks)


def reset_entities(sim: SimulationContext, entities: dict[str, object]) -> None:
    """Reset the preview scene to its configured initial state."""
    sim.reset()
    for entity in entities.values():
        entity.reset()


def apply_arm_action(entities: dict[str, object], action: torch.Tensor, left_arm_ids, right_arm_ids) -> None:
    """Apply a 14-D action containing left and right iiwa arm joint targets."""
    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]
    left_robot.set_joint_position_target(action[:7].unsqueeze(0), left_arm_ids)
    right_robot.set_joint_position_target(action[7:14].unsqueeze(0), right_arm_ids)


def write_collection_frame(
    episode: EpisodeData,
    action: torch.Tensor,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    entities: dict[str, object],
) -> None:
    """Store actions, VR targets, and low-dimensional states in an IsaacLab episode."""
    episode.add("actions", action.detach().cpu())
    episode.add("obs/vr/left_wrist_pose", left_pose.detach().cpu())
    episode.add("obs/vr/right_wrist_pose", right_pose.detach().cpu())
    episode.add("states/left_robot/joint_pos", entities["left_robot"].data.joint_pos[0].detach().cpu())
    episode.add("states/right_robot/joint_pos", entities["right_robot"].data.joint_pos[0].detach().cpu())
    episode.add("states/left_robot/joint_vel", entities["left_robot"].data.joint_vel[0].detach().cpu())
    episode.add("states/right_robot/joint_vel", entities["right_robot"].data.joint_vel[0].detach().cpu())
    for name, entity in entities.items():
        if name.startswith("object_"):
            episode.add(f"states/{name}/root_state_w", entity.data.root_state_w[0].detach().cpu())


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(eye=[2.5, -3.0, 2.2], target=[0.0, 0.4, 1.0])
    entities = design_scene()
    reset_entities(sim, entities)

    left_robot = entities["left_robot"]
    right_robot = entities["right_robot"]
    left_arm_ids, _ = left_robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    right_arm_ids, _ = right_robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)

    if args_cli.replay:
        replay_dataset(entities, sim, left_arm_ids, right_arm_ids)
        simulation_app.close()
        return

    if args_cli.left_urdf is None or args_cli.right_urdf is None:
        raise ValueError("--left-urdf and --right-urdf are required when collecting VR demonstrations")

    left_ik = make_ik_controller(left_robot, args_cli.left_urdf, "left_hand_C_MC")
    right_ik = make_ik_controller(right_robot, args_cli.right_urdf, "right_hand_C_MC")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_path = args_cli.dataset_file.expanduser().resolve()
    dataset_file_handler.create(str(dataset_path), env_name="DualSharpaVR")
    current_episode = EpisodeData()
    recording_active = True
    reset_requested = False

    def start_recording() -> None:
        nonlocal recording_active, current_episode
        current_episode = EpisodeData()
        recording_active = True

    def stop_recording() -> None:
        nonlocal recording_active, current_episode
        if not current_episode.is_empty():
            current_episode.success = True
            current_episode.pre_export()
            dataset_file_handler.write_episode(current_episode)
            dataset_file_handler.flush()
            print(f"[INFO] Wrote demo_{dataset_file_handler.demo_count - 1} to {dataset_path}")
            current_episode = EpisodeData()
        recording_active = False

    def reset_recording() -> None:
        nonlocal current_episode, reset_requested
        current_episode = EpisodeData()
        reset_requested = True

    callbacks = {"START": start_recording, "STOP": stop_recording, "RESET": reset_recording}
    device = make_device(callbacks)
    step_count = 0
    dt = sim.get_physics_dt()
    print(f"[INFO] VR collection started. Writing HDF5 episodes to {dataset_path}")

    try:
        while simulation_app.is_running():
            vr_pose = device.advance()
            left_pose = vr_pose[:7]
            right_pose = vr_pose[7:14]
            if args_cli.mapping_mode == "same":
                left_pose = map_pose(left_pose, args_cli.pose_scale, tuple(args_cli.pose_offset))
                right_pose = map_pose(right_pose, args_cli.pose_scale, tuple(args_cli.pose_offset))
            else:
                left_pose = map_pose(left_pose, args_cli.left_pose_scale, tuple(args_cli.left_pose_offset))
                right_pose = map_pose(right_pose, args_cli.right_pose_scale, tuple(args_cli.right_pose_offset))

            if reset_requested:
                reset_entities(sim, entities)
                reset_requested = False

            update_task_target(left_ik, left_pose)
            update_task_target(right_ik, right_pose)
            left_target = left_ik.compute(left_robot.data.joint_pos[0].detach().cpu().numpy(), dt)
            right_target = right_ik.compute(right_robot.data.joint_pos[0].detach().cpu().numpy(), dt)
            action = torch.cat((left_target, right_target))
            apply_arm_action(entities, action, left_arm_ids, right_arm_ids)
            left_robot.write_data_to_sim()
            right_robot.write_data_to_sim()
            for name, entity in entities.items():
                if name.startswith("object_"):
                    entity.write_data_to_sim()
            sim.step()
            for entity in entities.values():
                entity.update(dt)

            if recording_active:
                write_collection_frame(current_episode, action, left_pose, right_pose, entities)
            step_count += 1
            if args_cli.max_steps and step_count >= args_cli.max_steps:
                break
    finally:
        if recording_active and not current_episode.is_empty():
            stop_recording()
        dataset_file_handler.close()
        simulation_app.close()


def replay_dataset(entities: dict[str, object], sim: SimulationContext, left_arm_ids, right_arm_ids) -> None:
    """Replay one HDF5 episode using the saved 14-D arm joint targets."""
    dataset_path = args_cli.dataset_file.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    episode_name = f"demo_{args_cli.demo}"
    episode = handler.load_episode(episode_name, str(sim.device))
    if episode is None:
        raise ValueError(f"Episode {episode_name} was not found in {dataset_path}")
    actions = episode.data.get("actions")
    if actions is None:
        raise ValueError(f"Episode {episode_name} does not contain an actions dataset")
    print(f"[INFO] Replaying {episode_name} from {dataset_path} ({len(actions)} steps)")
    for action in actions:
        apply_arm_action(entities, action, left_arm_ids, right_arm_ids)
        entities["left_robot"].write_data_to_sim()
        entities["right_robot"].write_data_to_sim()
        for name, entity in entities.items():
            if name.startswith("object_"):
                entity.write_data_to_sim()
        sim.step()
        for entity in entities.values():
            entity.update(sim.get_physics_dt())
        if not simulation_app.is_running():
            break
    handler.close()
    print(f"[INFO] Finished replaying {episode_name}")


if __name__ == "__main__":
    main()