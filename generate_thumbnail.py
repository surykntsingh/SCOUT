import argparse
from pathlib import Path

from PIL import Image


def _default_output_path(image_path: Path) -> Path:
    stem = image_path.stem
    return image_path.with_name(f"{stem}_thumbnail.jpg")


def _thumbnail_with_openslide(image_path: Path, max_size: int) -> Image.Image:
    import openslide

    slide = openslide.OpenSlide(str(image_path))
    try:
        width, height = slide.dimensions
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid slide dimensions for {image_path}: {slide.dimensions}")

        scale = max(width / max_size, height / max_size, 1.0)
        thumb_size = (max(1, int(round(width / scale))), max(1, int(round(height / scale))))
        return slide.get_thumbnail(thumb_size).convert("RGB")
    finally:
        slide.close()


def _thumbnail_with_pil(image_path: Path, max_size: int) -> Image.Image:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return img


def generate_thumbnail(image_path: Path, output_path: Path | None = None, max_size: int = 1024) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    output_path = output_path or _default_output_path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        thumbnail = _thumbnail_with_openslide(image_path, max_size)
    except Exception:
        thumbnail = _thumbnail_with_pil(image_path, max_size)

    thumbnail.save(output_path, quality=95)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a thumbnail image from a WSI TIFF.")
    parser.add_argument("image_path", type=str, help="Path to the WSI image (.tif/.tiff).")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output thumbnail path. Defaults to <input>_thumbnail.jpg",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Maximum width or height of the thumbnail.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)
    output_path = Path(args.output) if args.output else None
    saved_path = generate_thumbnail(image_path, output_path=output_path, max_size=args.max_size)
    print(f"thumbnail saved at: {saved_path}")


if __name__ == "__main__":
    main()
