#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from detgeo_annotation_tool.repository import AnnotationRepository
from detgeo_annotation_tool.services.exporter import Exporter
from detgeo_annotation_tool.services.importer import import_pairs_from_manifest, import_pairs_from_pth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DetGeo annotation tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Import DetGeo/CVOGL pairs from .pth splits")
    init_parser.add_argument("--db", required=True, type=Path)
    init_parser.add_argument("--data-root", required=True, type=Path)
    init_parser.add_argument("--data-name", default="CVOGL_DroneAerial")
    init_parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])

    manifest_parser = subparsers.add_parser("import-manifest", help="Import pairs from manifest.json")
    manifest_parser.add_argument("--db", required=True, type=Path)
    manifest_parser.add_argument("--manifest", required=True, type=Path)

    gui_parser = subparsers.add_parser("gui", help="Launch GUI")
    gui_parser.add_argument("--db", required=True, type=Path)
    gui_parser.add_argument("--workspace", required=True, type=Path)

    export_parser = subparsers.add_parser("export", help="Export benchmark and annotations")
    export_parser.add_argument("--db", required=True, type=Path)
    export_parser.add_argument("--workspace", required=True, type=Path)
    export_parser.add_argument("--output-dir", required=True, type=Path)
    export_parser.add_argument("--include-partial-match", action="store_true")

    return parser.parse_args()


def prepare_linux_qt_runtime() -> None:
    if not sys.platform.startswith("linux"):
        return

    env = os.environ.copy()
    marker = "_DETGEO_QT_ENV_PREPARED"
    if env.get(marker) == "1":
        return

    changed = False
    conda_prefix = env.get("CONDA_PREFIX", "")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        if Path(conda_lib).exists():
            ld_paths = [part for part in env.get("LD_LIBRARY_PATH", "").split(":") if part]
            if conda_lib not in ld_paths:
                env["LD_LIBRARY_PATH"] = (
                    f"{conda_lib}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else conda_lib
                )
                changed = True

    if not env.get("QT_QPA_PLATFORM"):
        if env.get("WAYLAND_DISPLAY"):
            env["QT_QPA_PLATFORM"] = "wayland"
            changed = True
        elif env.get("DISPLAY"):
            env["QT_QPA_PLATFORM"] = "xcb"
            changed = True

    if changed:
        env[marker] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def main() -> None:
    args = parse_args()
    repo = AnnotationRepository(args.db)

    if args.command == "init-db":
        imported = import_pairs_from_pth(
            repo=repo,
            data_root=args.data_root,
            data_name=args.data_name,
            splits=args.splits,
        )
        print(f"Imported {imported} pairs into {args.db}")
        return

    if args.command == "import-manifest":
        imported = import_pairs_from_manifest(repo=repo, manifest_path=args.manifest)
        print(f"Imported {imported} manifest items into {args.db}")
        return

    if args.command == "export":
        exporter = Exporter(repo=repo, workspace_dir=args.workspace)
        result = exporter.export_all(
            output_dir=args.output_dir,
            include_partial_match=args.include_partial_match,
        )
        print(f"Export finished under {args.output_dir}")
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.command == "gui":
        prepare_linux_qt_runtime()
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise SystemExit(
                "No GUI display session detected. Launch from a desktop session with DISPLAY or WAYLAND_DISPLAY set."
            )
        try:
            from PySide6.QtWidgets import QApplication
        except ImportError as exc:
            raise SystemExit(
                "PySide6 is not installed. Install requirements.txt before launching the GUI."
            ) from exc
        from detgeo_annotation_tool.ui.main_window import MainWindow

        app = QApplication([])
        window = MainWindow(repo=repo, workspace_dir=args.workspace)
        window.resize(1800, 1040)
        window.show()
        app.exec()


if __name__ == "__main__":
    main()
