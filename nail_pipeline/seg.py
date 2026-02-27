from __future__ import annotations
from typing import List
import cv2
import numpy as np


def combine_masks_binary(selected_masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine masks: to draw overlay if multiple nails in picture

    """
    if not selected_masks:
        raise ValueError("combine_masks_binary called with empty selected_masks")

    combined = np.zeros_like(selected_masks[0], dtype=np.uint8)
    for m in selected_masks:
        combined = np.logical_or(combined.astype(bool), m.astype(bool)).astype(np.uint8)
    return combined * 255


def make_overlay(
    image_bgr: np.ndarray,
    mask_255: np.ndarray,
    color_bgr=(0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a colored mask on an image with transparency alpha.
    """
    colored = np.zeros_like(image_bgr, dtype=np.uint8)
    colored[:] = np.array(color_bgr, dtype=np.uint8)

    overlay = image_bgr.astype(np.float32).copy()
    img_f = image_bgr.astype(np.float32)
    col_f = colored.astype(np.float32)
    mask_bool = mask_255.astype(bool)

    overlay[mask_bool] = (1 - alpha) * img_f[mask_bool] + alpha * col_f[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_nail_only_crop_from_single_mask(image_bgr: np.ndarray, single_mask_255: np.ndarray) -> np.ndarray:
    """
    Tight crop for ONE nail mask. Background is black.
    """
    mask_bool = single_mask_255 > 0
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    nail_only = np.zeros_like(image_bgr, dtype=np.uint8)
    nail_only[mask_bool] = image_bgr[mask_bool]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return nail_only[y_min : y_max + 1, x_min : x_max + 1]


def extract_contours_from_mask(
    mask_255: np.ndarray,
    min_area: int = 50,
    approx_epsilon_ratio: float = 0.002,
) -> List[np.ndarray]:
    """
    Extract external contours from a binary mask (255 foreground),
    optionally simplify contour with approxPolyDP.
    Output contours are (N,2) int arrays, sorted by area desc.
    """
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output: List[np.ndarray] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if approx_epsilon_ratio > 0:
            eps = approx_epsilon_ratio * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)

        c2 = c.reshape(-1, 2).astype(int)
        if len(c2) >= 3:
            output.append(c2)

    output.sort(
        key=lambda arr: cv2.contourArea(arr.reshape(-1, 1, 2).astype(np.int32)),
        reverse=True,
    )
    return output


def write_contours_txt(
    mask_255: np.ndarray,
    txt_path: str,
    min_area: int = 50,
    approx_epsilon_ratio: float = 0.002,
) -> None:
    """
    Write contours to txt:
      header includes image w/h and number of contours
      each contour: area + point list
    """
    h, w = mask_255.shape[:2]
    contours = extract_contours_from_mask(mask_255, min_area=min_area, approx_epsilon_ratio=approx_epsilon_ratio)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# width={w} height={h}\n")
        f.write(f"# num_contours={len(contours)}\n")
        f.write("# format=contours_xy\n")

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt.reshape(-1, 1, 2).astype(np.float32))
            f.write(f"contour {i} area={area:.2f} points={len(cnt)}\n")
            for x, y in cnt:
                f.write(f"{x},{y}\n")
            f.write("end\n")