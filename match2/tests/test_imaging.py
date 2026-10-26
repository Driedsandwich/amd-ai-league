"""画像の整え: 向き・RGB への変換（16bit・パレット・CMYK・透明）・大きい物の縮小・小さい物の拡大。"""
import numpy as np
import pytest
from PIL import Image

from mc2ocr import imaging


def prep(path, **kw):
    return imaging.prepare(str(path), **kw)


def test_big_image_is_shrunk_to_max_side(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (4000, 1000), (10, 20, 30)).save(p)
    im = prep(p, limit=1536)
    assert im.mode == "RGB" and im.size == (1536, 384)


def test_big_image_default_limit_env(tmp_path, monkeypatch):
    p = tmp_path / "big.png"
    Image.new("RGB", (3000, 3000)).save(p)
    monkeypatch.setenv("X052_MAX_SIDE", "800")
    assert prep(p).size == (800, 800)


def test_small_image_is_doubled(tmp_path):
    p = tmp_path / "small.png"
    Image.new("RGB", (300, 120)).save(p)
    assert prep(p).size == (600, 240)


def test_boundary_sizes_unchanged(tmp_path):
    for w in (640, 1536):
        p = tmp_path / f"w{w}.png"
        Image.new("RGB", (w, 100)).save(p)
        assert prep(p).size == (w, 100)


def test_16bit_tiff_gray(tmp_path):
    arr = (np.linspace(0, 65535, 700 * 100).reshape(100, 700)).astype(np.uint16)
    p = tmp_path / "g16.tif"
    Image.fromarray(arr).save(p)
    assert Image.open(p).mode.startswith("I;16")
    im = prep(p)
    assert im.mode == "RGB"
    out = np.asarray(im)
    assert out.max() >= 250 and out.min() <= 5  # 8bit の全域に直る（真っ黒・真っ白に潰れない）


def test_16bit_low_values_not_black(tmp_path):
    arr = np.full((100, 700), 3000, dtype=np.uint16)
    arr[40:60, 100:600] = 1000
    p = tmp_path / "low16.tif"
    Image.fromarray(arr).save(p)
    out = np.asarray(prep(p))
    assert out.max() == 255 and out.min() < 255  # 比例して広げる


def test_int32_and_float_modes(tmp_path):
    for mode, arr in (("I", np.full((100, 700), 40000, dtype=np.int32)), ("F", np.full((100, 700), 0.5, dtype=np.float32))):
        p = tmp_path / f"{mode}.tif"
        Image.fromarray(arr, mode=mode).save(p)
        assert prep(p).mode == "RGB"


def test_cmyk_jpeg_and_tiff(tmp_path):
    for ext in ("jpg", "tif"):
        p = tmp_path / f"cmyk.{ext}"
        Image.new("CMYK", (700, 200), (0, 255, 255, 0)).save(p)
        im = prep(p)
        assert im.mode == "RGB"
        r, g, b = im.getpixel((10, 10))
        assert r > 150 and g < 100 and b < 100  # 赤っぽい


def test_palette_png(tmp_path):
    p = tmp_path / "pal.png"
    im = Image.new("P", (700, 200))
    im.putpalette([255, 0, 0] + [0, 0, 255] * 255)
    im.save(p)
    assert prep(p).getpixel((5, 5)) == (255, 0, 0)


def test_palette_png_with_transparency_goes_white(tmp_path):
    p = tmp_path / "palt.png"
    im = Image.new("P", (700, 200), 0)
    im.putpalette([255, 0, 0] + [0, 0, 255] * 255)
    im.save(p, transparency=0)
    assert prep(p).getpixel((5, 5)) == (255, 255, 255)


def test_rgba_png_transparent_over_white(tmp_path):
    p = tmp_path / "rgba.png"
    im = Image.new("RGBA", (700, 200), (0, 0, 0, 0))
    im.paste((0, 0, 0, 255), (0, 0, 100, 200))
    im.save(p)
    out = prep(p)
    assert out.mode == "RGB" and out.getpixel((300, 100)) == (255, 255, 255) and out.getpixel((10, 10)) == (0, 0, 0)


def test_la_and_1bit_and_gray(tmp_path):
    for mode in ("LA", "1", "L"):
        p = tmp_path / f"m_{mode}.png"
        Image.new(mode, (700, 200)).save(p)
        assert prep(p).mode == "RGB"


def test_exif_orientation_applied(tmp_path):
    p = tmp_path / "rot.jpg"
    im = Image.new("RGB", (800, 700), (200, 200, 200))
    exif = Image.Exif()
    exif[0x0112] = 6  # 時計回りに 90 度
    im.save(p, exif=exif)
    assert prep(p).size == (700, 800)


def test_multiframe_tiff_takes_first_frame(tmp_path):
    p = tmp_path / "multi.tif"
    a = Image.new("RGB", (700, 200), (255, 0, 0))
    b = Image.new("RGB", (700, 200), (0, 0, 255))
    a.save(p, save_all=True, append_images=[b])
    assert prep(p).getpixel((3, 3)) == (255, 0, 0)


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    with pytest.raises(Exception):
        prep(p)
