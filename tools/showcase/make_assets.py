"""Generates the two pictures the showcase film needs, with no real people or real screens in them:
  wallpaper.png   the "desktop" behind the island (the glass blurs it)
  callgrid.png    a made-up video-call screen: neutral tiles, invented names, a green ring on the tile that is speaking
"""
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_B = r"C:\Windows\Fonts\segoeuib.ttf"


def gradient(c1, c2):
    y, x = np.mgrid[0:H, 0:W]
    t = ((x / W) * 0.55 + (y / H) * 0.45)[..., None]
    return Image.fromarray((np.array(c1) * (1 - t) + np.array(c2) * t).astype(np.uint8))


def person(d, box, hue):
    """a neutral silhouette (head and shoulders) - nobody in particular"""
    x1, y1, x2, y2 = box
    cx, w, h = (x1 + x2) // 2, x2 - x1, y2 - y1
    col = tuple(int(c * 0.55) for c in hue)
    d.ellipse((cx - w * 0.11, y1 + h * 0.22, cx + w * 0.11, y1 + h * 0.22 + w * 0.22), fill=col)
    d.pieslice((cx - w * 0.22, y1 + h * 0.52, cx + w * 0.22, y1 + h * 1.25), 180, 360, fill=col)


def grid(base, names, ring=None, rect=(130, 110, 1790, 930), cols=3, rows=2, gap=14, name_px=17):
    im = base.copy()
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, name_px)
    x0, y0, x1, y1 = rect
    tw, th = (x1 - x0 - gap * (cols - 1)) / cols, (y1 - y0 - gap * (rows - 1)) / rows
    hues = [(120, 170, 255), (255, 170, 120), (150, 220, 170), (230, 150, 210), (255, 220, 120), (140, 200, 230)]
    k = 0
    for r in range(rows):
        for c in range(cols):
            a = (int(x0 + c * (tw + gap)), int(y0 + r * (th + gap)))
            b = (int(a[0] + tw), int(a[1] + th))
            hue = hues[k % len(hues)]
            d.rounded_rectangle((*a, *b), radius=10, fill=(int(hue[0] * .30), int(hue[1] * .30), int(hue[2] * .34), 255))
            person(d, (*a, *b), hue)
            nm = names[k % len(names)]
            wtxt = d.textlength(nm, font=f)
            d.rounded_rectangle((a[0] + 8, b[1] - name_px - 18, a[0] + 8 + wtxt + 38, b[1] - 8), radius=6, fill=(20, 20, 24, 255))
            d.ellipse((a[0] + 15, b[1] - name_px - 12, a[0] + 27, b[1] - 15), fill=(210, 70, 70, 255))
            d.text((a[0] + 34, b[1] - name_px - 16), nm, font=f, fill=(255, 255, 255, 255))
            if ring == k:
                for t in range(5):
                    d.rectangle((a[0] - t, a[1] - t, b[0] + t, b[1] + t), outline=(170, 215, 100, 255))
            k += 1
    return im


def main():
    random.seed(4)
    wall = gradient((16, 20, 44), (74, 36, 104))
    blobs = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(blobs)
    for _ in range(9):
        x, y, r = random.randint(0, W), random.randint(0, H), random.randint(160, 380)
        d.ellipse((x - r, y - r, x + r, y + r), fill=random.choice([(90, 60, 200), (200, 80, 130), (50, 140, 220), (240, 150, 60)]))
    blobs = blobs.filter(ImageFilter.GaussianBlur(120))
    wall = Image.blend(wall, blobs, 0.45)
    # a made-up meeting window under the island, so the glass has something to blur
    win = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    wd = ImageDraw.Draw(win)
    wd.rounded_rectangle((90, 70, 1830, 990), radius=18, fill=(14, 16, 24, 235))
    wall = Image.alpha_composite(wall.convert("RGBA"), win)
    wall = grid(wall.convert("RGB"), ["Asha", "Ravi", "Meera", "Kabir", "Nisha", "Dev"])
    wall.save(HERE / "wallpaper.png")
    call = gradient((12, 14, 22), (22, 24, 36))
    rect = (120, 150, 1800, 960)
    grid(call, ["Asha Verma", "Ravi Menon", "Meera Iyer", "Kabir Shah", "Nisha Rao", "Dev Patel"], ring=1, rect=rect).save(HERE / "callgrid.png")
    print("assets written")


if __name__ == "__main__":
    main()
