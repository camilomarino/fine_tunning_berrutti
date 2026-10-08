"""Conservative OCR augmentations that preserve every transcribed glyph."""

from __future__ import annotations

import hashlib
import random
from PIL import Image, ImageEnhance, ImageFilter


def presentation_seed(
    seed: int,
    *,
    stage: str,
    epoch: int,
    position: int,
    page_id: str,
    region: int | None,
) -> int:
    content = f"{seed}:{stage}:{epoch}:{position}:{page_id}:{region}".encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "big")


def conservative_augment(
    image: Image.Image, seed: int, *, page: bool
) -> tuple[Image.Image, dict]:
    """Make a new image; never crop a page or apply text-changing geometry."""
    rng = random.Random(seed)
    result = image.copy()
    operations: dict[str, float] = {}
    if rng.random() < 0.45:
        factor = rng.uniform(0.90, 1.10)
        result = ImageEnhance.Contrast(result).enhance(factor)
        operations["contrast"] = factor
    if rng.random() < 0.30:
        factor = rng.uniform(0.92, 1.08)
        result = ImageEnhance.Brightness(result).enhance(factor)
        operations["brightness"] = factor
    if rng.random() < 0.10:
        radius = rng.uniform(0.10, 0.30)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))
        operations["gaussian_blur_radius"] = radius
    if page and rng.random() < 0.02:
        degrees = rng.uniform(-0.4, 0.4)
        result = result.rotate(degrees, expand=True, fillcolor=(255, 255, 255))
        operations["rotation_degrees"] = degrees
    return result, operations


def padded_crop(image: Image.Image, xyxy: list[int]) -> Image.Image:
    """Include a small margin around an aligned line instead of clipping glyphs."""
    x0, y0, x1, y1 = xyxy
    margin = max(4, min(12, round((y1 - y0) * 0.06)))
    return image.crop(
        (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(image.width, x1 + margin),
            min(image.height, y1 + margin),
        )
    )
