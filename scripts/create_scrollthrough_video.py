from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tif
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
STACK_DIR = (
    ROOT
    / "reports/260213_pilot_20260623_125859/single_neuron_examples/registered_roi_stacks"
)
METRICS = (
    ROOT
    / "reports/260213_pilot_20260623_125859/single_neuron_examples/single_neuron_mcherry_metrics.csv"
)
OUT_DIR = ROOT / "outputs/videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [8, 12, 16]
FPS = 24
SECONDS = 12
WIDTH = 1280
HEIGHT = 720
BG = (252, 253, 251)
DARK = (62, 62, 62)
MUTED = (92, 92, 92)
GREEN = (214, 236, 221)
BLUE = (121, 195, 226)


def main() -> None:
    stacks = {
        "E05 reporter control": tif.imread(STACK_DIR / "E05_single_neuron_registered_tcyx.ome.tif"),
        "F05 PLD3+TMEM106B": tif.imread(STACK_DIR / "F05_single_neuron_registered_tcyx.ome.tif"),
    }
    frames = build_frames(stacks)
    mp4_path = OUT_DIR / "tmem_single_neuron_scrollthrough.mp4"
    gif_path = OUT_DIR / "tmem_single_neuron_scrollthrough.gif"
    imageio.mimsave(mp4_path, frames, fps=FPS, quality=8, macro_block_size=16)
    imageio.mimsave(gif_path, frames[::3], duration=1000 / (FPS / 3), loop=0)
    print(mp4_path)
    print(gif_path)


def build_frames(stacks: dict[str, np.ndarray]) -> list[np.ndarray]:
    total_frames = FPS * SECONDS
    day_positions = np.linspace(0, len(DAYS) - 1, total_frames)
    frames: list[np.ndarray] = []
    for frame_index, pos in enumerate(day_positions):
        day_float = pos
        left_day = int(np.floor(day_float))
        right_day = min(left_day + 1, len(DAYS) - 1)
        alpha = day_float - left_day
        if frame_index > total_frames - FPS * 1.5:
            alpha = 0
            left_day = right_day = len(DAYS) - 1
        frames.append(draw_frame(stacks, left_day, right_day, alpha, frame_index, total_frames))
    return frames


def draw_frame(
    stacks: dict[str, np.ndarray],
    left_day: int,
    right_day: int,
    alpha: float,
    frame_index: int,
    total_frames: int,
) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    title_font = safe_font(40, bold=True)
    subtitle_font = safe_font(21)
    label_font = safe_font(24, bold=True)
    small_font = safe_font(17)
    tiny_font = safe_font(14)

    draw_organic_background(draw)
    draw.text((74, 54), "Single-neuron candidate scroll-through", fill=DARK, font=title_font)
    draw.text(
        (78, 105),
        "Registered local ROI crops | mCherry channel | Days 8, 12, 16",
        fill=MUTED,
        font=subtitle_font,
    )

    active_day = DAYS[right_day] if alpha > 0.5 else DAYS[left_day]
    draw.text((1035, 64), f"Day {active_day}", fill=DARK, font=label_font)
    draw_timeline(draw, active_day)

    panel_y = 188
    panel_w = 470
    panel_h = 360
    x_positions = [105, 705]
    for (label, stack), x in zip(stacks.items(), x_positions, strict=True):
        frame = interpolate_stack_frame(stack[:, 1], left_day, right_day, alpha)
        overlay = interpolate_overlay(stack, left_day, right_day, alpha)
        m_img = colorize(frame, cmap="magma").resize((300, 300), Image.Resampling.NEAREST)
        o_img = overlay.resize((136, 136), Image.Resampling.NEAREST)
        rounded_rect(draw, (x, panel_y, x + panel_w, panel_y + panel_h), fill=(255, 255, 255), outline=(225, 225, 225))
        draw.text((x + 22, panel_y + 20), label, fill=DARK, font=label_font)
        draw.text((x + 22, panel_y + 52), "mCherry intensity", fill=MUTED, font=small_font)
        canvas.paste(m_img, (x + 22, panel_y + 82))
        canvas.paste(o_img, (x + 330, panel_y + 142))
        draw.text((x + 325, panel_y + 112), "overlay", fill=MUTED, font=tiny_font)
        draw.text((x + 305, panel_y + 286), "red=mCherry", fill=(150, 30, 30), font=tiny_font)
        draw.text((x + 305, panel_y + 306), "green=488", fill=(40, 120, 60), font=tiny_font)

    draw.text(
        (114, 604),
        "Interpretation: punctate-to-diffuse reporter redistribution is a screening metric, not proof of lysosomal rupture.",
        fill=DARK,
        font=small_font,
    )
    draw.text(
        (114, 635),
        "Use this video as a visual example; same-neuron identity still needs manual ROI review.",
        fill=MUTED,
        font=tiny_font,
    )
    return np.asarray(canvas)


def draw_organic_background(draw: ImageDraw.ImageDraw) -> None:
    draw.ellipse((-230, 65, 92, 760), fill=GREEN)
    draw.ellipse((1130, 485, 1390, 800), fill=GREEN)
    draw.ellipse((1050, 600, 1190, 740), fill=(236, 248, 241))
    for x, y, r in [(1137, 635, 7), (1180, 658, 5), (1115, 685, 5)]:
        draw.ellipse((x, y, x + r, y + r), fill=BLUE)
    draw.text((1160, 612), "+", fill=BLUE, font=safe_font(24, bold=True))


def draw_timeline(draw: ImageDraw.ImageDraw, active_day: int) -> None:
    x0, y = 800, 125
    x1 = 1160
    draw.line((x0, y, x1, y), fill=(190, 190, 190), width=2)
    for day, x in zip(DAYS, np.linspace(x0, x1, len(DAYS)), strict=True):
        color = BLUE if day == active_day else (190, 190, 190)
        draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
        draw.text((x - 23, y + 18), f"D{day}", fill=MUTED, font=safe_font(14))


def interpolate_stack_frame(stack: np.ndarray, left: int, right: int, alpha: float) -> np.ndarray:
    a = stack[left].astype(np.float32)
    b = stack[right].astype(np.float32)
    return (1 - alpha) * a + alpha * b


def interpolate_overlay(stack: np.ndarray, left: int, right: int, alpha: float) -> Image.Image:
    mcherry = interpolate_stack_frame(stack[:, 1], left, right, alpha)
    stable = interpolate_stack_frame(stack[:, 2], left, right, alpha)
    red = normalize(mcherry)
    green = normalize(stable)
    rgb = np.zeros((*red.shape, 3), dtype=np.uint8)
    rgb[..., 0] = red
    rgb[..., 1] = green
    return Image.fromarray(rgb, "RGB")


def colorize(frame: np.ndarray, cmap: str) -> Image.Image:
    norm = normalize_float(frame)
    rgb = (plt.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def normalize(frame: np.ndarray) -> np.ndarray:
    return (normalize_float(frame) * 255).astype(np.uint8)


def normalize_float(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame.astype(np.float32), [1, 99.5])
    return np.clip((frame.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill, outline) -> None:
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=1)


def safe_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Georgia Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Georgia.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
