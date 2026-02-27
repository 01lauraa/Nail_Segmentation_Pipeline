from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class RefineConfig:
    """
    Refine segmented nail crops using a single adaptive ellipse.

    Pipeline per image:
      1) infer foreground from non-black pixels
      2) fit ellipse to largest foreground contour
      3) shrink fitted ellipse
      4) cut top and bottom of the ellipse mask
      5) apply final mask to the crop

    Assumption:
      input crops have black background outside the segmented nail.
    """
    shrink_ratio: float = 0.95

    # Fractions of the FINAL fitted-shrunk ellipse mask height
    top_cut: float = 0.22
    bottom_cut: float = 0.06

    min_contour_area: int = 20


def get_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Infer foreground from non-black pixels.
    Returns binary mask in {0, 255}.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg = (gray > 0).astype(np.uint8) * 255
    return fg


def largest_contour(bin_mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def fit_oval_to_fg(fg_mask: np.ndarray, cfg: RefineConfig) -> Tuple[int, int, int, int, float]:
    """
    Fit ellipse to largest contour.

    Returns:
      (cx, cy, ax, ay, angle)

    where:
      - (cx, cy) is the center
      - ax, ay are SEMI-axes
      - angle is ellipse rotation in degrees

    If the contour has < 5 points, falls back to an ellipse derived from the
    bounding rectangle.
    """
    cnt = largest_contour(fg_mask, cfg.min_contour_area)
    h, w = fg_mask.shape[:2]

    if cnt is None:
        return (w // 2, h // 2, max(1, w // 4), max(1, h // 4), 0.0)

    if len(cnt) >= 5:
        (cx, cy), (major, minor), angle = cv2.fitEllipse(cnt)

        # fitEllipse returns full axis lengths -> convert to semi-axes
        ax = max(1, int(round(major * 0.5)))
        ay = max(1, int(round(minor * 0.5)))

        return int(round(cx)), int(round(cy)), ax, ay, float(angle)

    # fallback if too few points
    x, y, bw, bh = cv2.boundingRect(cnt)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    ax = max(1, int(round(bw * 0.5)))
    ay = max(1, int(round(bh * 0.5)))
    return int(round(cx)), int(round(cy)), ax, ay, 0.0


def shrink_oval(
    cx: int,
    cy: int,
    ax: int,
    ay: int,
    angle: float,
    shrink_ratio: float,
) -> Tuple[int, int, int, int, float]:
    """
    Shrink ellipse semi-axes by shrink_ratio.
    Center and angle stay unchanged.
    """
    ax2 = max(1, int(round(ax * shrink_ratio)))
    ay2 = max(1, int(round(ay * shrink_ratio)))
    return cx, cy, ax2, ay2, angle


def draw_oval_mask(
    shape: Tuple[int, ...],
    cx: int,
    cy: int,
    ax: int,
    ay: int,
    angle: float,
) -> np.ndarray:
    """
    Draw filled ellipse mask in {0,255}.
    """
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
    return m


def unified_cut_top_bottom(mask: np.ndarray, top_frac: float, bottom_frac: float) -> np.ndarray:
    """
    Find the occupied vertical span of the current mask and remove:
      - top top_frac of that span
      - bottom bottom_frac of that span
    """
    out = mask.copy()

    ys, _xs = np.where(out > 0)
    if len(ys) == 0:
        return out

    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1

    top_cut_px = int(round(height * top_frac))
    bottom_cut_px = int(round(height * bottom_frac))

    y_cut_top = y_min + top_cut_px
    y_cut_bottom = y_max - bottom_cut_px

    out[:max(0, y_cut_top), :] = 0
    out[min(out.shape[0], y_cut_bottom + 1):, :] = 0
    return out


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask)


def refine_single_crop(
    img_bgr: np.ndarray,
    cfg: RefineConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Refine one segmented nail crop using a single adaptive ellipse.

    Returns:
      - refined_output_bgr
      - final_mask_255 (or None if no foreground)
    """
    fg = get_foreground_mask(img_bgr)

    if np.count_nonzero(fg) == 0:
        return img_bgr, None

    # 1) fit ellipse to current foreground
    cx, cy, ax, ay, ang = fit_oval_to_fg(fg, cfg)

    # 2) shrink fitted ellipse
    cx2, cy2, ax2, ay2, ang2 = shrink_oval(cx, cy, ax, ay, ang, cfg.shrink_ratio)

    # 3) draw shrunk ellipse
    final_mask = draw_oval_mask(img_bgr.shape, cx2, cy2, ax2, ay2, ang2)

    # 4) cut top and bottom
    final_mask = unified_cut_top_bottom(final_mask, cfg.top_cut, cfg.bottom_cut)

    # 5) apply final mask
    refined_out = apply_mask(img_bgr, final_mask)

    return refined_out, final_mask


def refine_folder(
    img_folder: str,
    out_folder: str,
    cfg: Optional[RefineConfig] = None,
) -> None:
    """
    Refine all segmented nail crops in img_folder and save outputs to out_folder.
    """
    cfg = cfg or RefineConfig()

    os.makedirs(out_folder, exist_ok=True)

    files = [
        f for f in os.listdir(img_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Found {len(files)} images.")

    for fname in sorted(files):
        in_path = os.path.join(img_folder, fname)
        img = cv2.imread(in_path)

        if img is None:
            print(f"[WARN] Could not read {fname}")
            continue

        refined, _final_mask = refine_single_crop(img, cfg)

        out_path = os.path.join(out_folder, fname)
        cv2.imwrite(out_path, refined)

    print(f"\nDone! Refined ellipse outputs saved to: {out_folder}")