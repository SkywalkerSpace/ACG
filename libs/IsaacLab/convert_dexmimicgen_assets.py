"""Convert DexMimicGen MJCF object assets to Isaac USD assets.

Run from the IsaacLab repository, for example::

    ./isaaclab.sh -p /path/to/ACG/libs/IsaacLab/convert_dexmimicgen_assets.py

All XML files below ``models/assets/objects`` are converted recursively. The
relative directory layout is kept in the output directory to avoid collisions
between files such as ``model.xml`` in different Objaverse assets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
workspace_root = Path(__file__).resolve().parents[3]
default_input_dir = workspace_root / "ACG" / "libs" / "dexmimicgen" / "dexmimicgen" / "models" / "assets" / "objects"
default_output_dir = Path(__file__).resolve().parent / "assets" / "converted" / "objects"
parser.add_argument("--input-dir", type=Path, default=default_input_dir)
parser.add_argument("--output-dir", type=Path, default=default_output_dir)
parser.add_argument("--pattern", default="*.xml", help="Recursive XML glob (default: *.xml).")
parser.add_argument("--fix-base", action="store_true", help="Add a fixed base to imported MJCF assets.")
parser.add_argument("--no-import-sites", action="store_true", help="Do not import MuJoCo site markers.")
parser.add_argument("--make-instanceable", action="store_true")
parser.add_argument("--continue-on-error", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


def output_path_for(xml_path: Path, input_dir: Path, output_dir: Path) -> Path:
    relative_path = xml_path.relative_to(input_dir).with_suffix(".usd")
    return output_dir / relative_path


def convert_xml(xml_path: Path, input_dir: Path, output_dir: Path) -> None:
    output_path = output_path_for(xml_path, input_dir, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = MjcfConverterCfg(
        asset_path=str(xml_path),
        usd_dir=str(output_path.parent),
        usd_file_name=output_path.name,
        fix_base=args_cli.fix_base,
        import_sites=not args_cli.no_import_sites,
        force_usd_conversion=True,
        make_instanceable=args_cli.make_instanceable,
    )
    converter = MjcfConverter(cfg)
    print(f"[INFO] {xml_path.relative_to(input_dir)} -> {converter.usd_path}")


def main() -> None:
    input_dir = args_cli.input_dir.expanduser().resolve()
    output_dir = args_cli.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"MJCF asset directory does not exist: {input_dir}")

    xml_files = sorted(input_dir.rglob(args_cli.pattern))
    if not xml_files:
        raise FileNotFoundError(f"No XML files matching {args_cli.pattern!r} under {input_dir}")

    failures = 0
    for xml_path in xml_files:
        try:
            convert_xml(xml_path, input_dir, output_dir)
        except Exception as exc:
            failures += 1
            print(f"[ERROR] {xml_path}: {exc}")
            if not args_cli.continue_on_error:
                raise
    if failures:
        raise RuntimeError(f"{failures} MJCF conversions failed")


if __name__ == "__main__":
    main()
    simulation_app.close()