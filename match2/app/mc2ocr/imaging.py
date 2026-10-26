"""画像の読み込みと整え。PIL だけを使う（主催の土台の像に入っている）。

向きを直す → RGB にする（16bit・パレット・CMYK・透明を含む）→ 長い辺が上限を超えたら縮める →
短すぎるなら 2 倍に広げる。
"""
from __future__ import annotations

import os
from typing import Optional

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 300_000_000  # 既定（約 9 千万）より少し広げる。これを超える物は例外＝空の答え


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


def max_side() -> int:
    return _env_int("X052_MAX_SIDE", 1536)


def min_long_side() -> int:
    return _env_int("X052_MIN_LONG_SIDE", 640)


def _to_8bit_gray(img: Image.Image) -> Image.Image:
    """I;16・I・F の濃淡を 8bit へ。最大値が 255 を超えるときだけ、最大値が 255 になるよう比例して縮める。"""
    lo, hi = img.getextrema()
    scale = 255.0 / hi if hi and hi > 255 else 1.0
    if img.mode == "F":
        return img.point(lambda v: v * scale).convert("L")
    return img.point(lambda v: v * scale, "L") if img.mode in ("I", "I;16", "I;16L", "I;16B") else img.convert("L")


def to_rgb(img: Image.Image) -> Image.Image:
    mode = img.mode
    if mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        return _to_8bit_gray(img).convert("RGB")
    if mode == "P" or "transparency" in img.info:
        img = img.convert("RGBA")
        mode = "RGBA"
    if mode in ("RGBA", "LA", "PA", "La", "RGBa"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if mode == "RGB":
        return img
    return img.convert("RGB")  # L・1・CMYK・YCbCr・LAB など


def prepare(path: str, limit: Optional[int] = None, min_long: Optional[int] = None) -> Image.Image:
    limit = limit or max_side()
    min_long = min_long if min_long is not None else min_long_side()
    img = Image.open(path)
    if getattr(img, "format", None) == "JPEG":
        img.draft("RGB", (limit * 2, limit * 2))  # 大きな JPEG は読むときに縮めて速くする（2 倍以上は残す）
    try:
        img.seek(0)  # 複数頁の TIFF などは先頭の 1 枚
    except (EOFError, AttributeError):
        pass
    img.load()
    img = ImageOps.exif_transpose(img) or img
    img = to_rgb(img)
    w, h = img.size
    long_side = max(w, h)
    if long_side > limit:
        r = limit / long_side
        img = img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    elif long_side < min_long:
        img = img.resize((w * 2, h * 2), Image.LANCZOS)
    return img
