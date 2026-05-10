#!/usr/bin/env python3
import argparse
from pathlib import Path

from PIL import Image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESAMPLE_MAP = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in VALID_EXTS:
            return [input_path]
        return []
    if not input_path.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() in VALID_EXTS
    )


def resize_image(src: Path, dst: Path, size: int, resample: str, force_rgb: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        if force_rgb:
            img = img.convert("RGB")
        resized = img.resize((size, size), RESAMPLE_MAP[resample])
        resized.save(dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize one image or a folder of images to 512x512 (or a custom square size)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input image file or input directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image file (single input) or output directory (folder input).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target width/height. Default: 512",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders when input is a directory.",
    )
    parser.add_argument(
        "--resample",
        choices=sorted(RESAMPLE_MAP.keys()),
        default="lanczos",
        help="Resize interpolation method.",
    )
    parser.add_argument(
        "--force-rgb",
        action="store_true",
        help="Convert image to RGB before saving.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    images = collect_images(input_path, args.recursive)
    if not images:
        raise FileNotFoundError(f"No valid images found under: {input_path}")

    if input_path.is_file():
        # Single input image: --output should be a file path.
        if output_path.exists() and output_path.is_dir():
            output_path = output_path / input_path.name
        resize_image(images[0], output_path, args.size, args.resample, args.force_rgb)
        print(f"Done: {images[0]} -> {output_path}")
        return

    # Folder input: keep relative directory structure under output dir.
    output_path.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in images:
        rel = src.relative_to(input_path)
        dst = output_path / rel
        resize_image(src, dst, args.size, args.resample, args.force_rgb)
        count += 1

    print(f"Done: resized {count} images to {args.size}x{args.size}")
    print(f"Output dir: {output_path}")


if __name__ == "__main__":
    main()
