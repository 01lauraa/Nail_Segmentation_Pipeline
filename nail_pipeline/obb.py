from __future__ import annotations
from typing import Tuple
import cv2
import numpy as np


def box_from_contour(contour: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Find minimum-area rotated rectangle around contour.
    Returns:
      - box_pts: (4,2) int corners
      - angle: normalized to follow long axis
      - h_box: SHORT side length after normalization (useful for shift ratio)
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w_box, h_box), angle = rect

    # Normalize so angle follows long axis by forcing w_box >= h_box
    if w_box < h_box:
        angle += 90.0
        w_box, h_box = h_box, w_box

    box = cv2.boxPoints(((cx, cy), (w_box, h_box), angle))
    box = np.intp(box)
    return box, float(angle), float(h_box)


def shift_bottom_two_vertices(box_pts: np.ndarray, shift_px: int, img_h: int) -> np.ndarray:
    """
    Shift the two bottom-most vertices downward.
    """
    pts = box_pts.astype(np.float32).copy()
    ys = pts[:, 1]
    bottom_idx = ys.argsort()[-2:]
    for idx in bottom_idx:
        pts[idx, 1] = min(img_h - 1, pts[idx, 1] + shift_px)
    return np.intp(pts)


def polygon_crop(image_bgr: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """
    Mask outside polygon and return tight bounding-rect crop.
    """
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    patch = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    x, y, wc, hc = cv2.boundingRect(polygon)
    return patch[y : y + hc, x : x + wc]