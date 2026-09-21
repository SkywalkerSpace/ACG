"""Convert the left and right Sharpa iiwa URDF files to Isaac USD assets.

Run from the IsaacLab repository, for example::

    ./isaaclab.sh -p /path/to/ACG/libs/IsaacLab/convert_sharpa_urdf.py

The generated USD files are fixed-base articulations by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
workspace_root = Path(__file__).resolve().parents[3]
default_robot_dir = workspace_root / "peg_screwing_dual_migration_20260918" / "play2perfect_left_official_right_mirror" / "assets" / "robots"
default_output_dir = Path(__file__).resolve().parent / "assets" / "converted" / "robots"
parser.add_argument("--left-urdf", type=Path, default=default_robot_dir / "iiwa14_left_sharpa_adjusted_restricted.urdf")
parser.add_argument("--right-urdf", type=Path, default=default_robot_dir / "iiwa14_right_sharpa_mirror_restricted.urdf")
parser.add_argument("--output-dir", type=Path, default=default_output_dir)
parser.add_argument("--floating-base", action="store_true", help="Do not fix the URDF base.")
parser.add_argument("--joint-stiffness", type=float, default=400.0)
parser.add_argument("--joint-damping", type=float, default=40.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def convert_robot(urdf_path: Path, output_path: Path) -> None:
    """Convert one URDF while preserving its relative mesh references."""
    urdf_path = urdf_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        fix_base=not args_cli.floating_base,
        merge_fixed_joints=False,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            ),
            target_type="position",
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"[INFO] {urdf_path.name} -> {converter.usd_path}")


def main() -> None:
    convert_robot(args_cli.left_urdf, args_cli.output_dir / "iiwa14_left_sharpa.usd")
    convert_robot(args_cli.right_urdf, args_cli.output_dir / "iiwa14_right_sharpa.usd")


if __name__ == "__main__":
    main()
    simulation_app.close()