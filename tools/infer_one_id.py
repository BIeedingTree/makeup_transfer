#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run infer_kps.py for only one id image without modifying infer_kps.py."
    )
    parser.add_argument(
        "--project-dir",
        default="/DATA/yantongliu/FinalProj/Stable-Makeup",
        help="Stable-Makeup project root.",
    )
    parser.add_argument(
        "--id-name",
        required=True,
        help="Only this id image will be processed, e.g. 5.jpg",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir).expanduser().resolve()
    id_dir = project_dir / "test_imgs" / "id"
    target = args.id_name
    target_path = id_dir / target
    if not target_path.exists():
        raise FileNotFoundError(f"Target id image not found: {target_path}")

    os.chdir(project_dir)
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    original_listdir = os.listdir

    def filtered_listdir(path):
        items = original_listdir(path)
        try:
            abs_path = Path(path).resolve()
        except Exception:
            return items
        if abs_path == id_dir:
            return [name for name in items if name == target]
        return items

    os.listdir = filtered_listdir
    try:
        import infer_kps  # noqa: F401

        infer_kps.infer()
    finally:
        os.listdir = original_listdir


if __name__ == "__main__":
    main()
