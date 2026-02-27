from __future__ import annotations
import glob
import os
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ExifTags
from ultralytics import YOLO


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def list_images(input_dir: str) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths: List[str] = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(paths)


def fix_exif_rotation(image_path: str) -> np.ndarray:
    """
    Load image with EXIF orientation correction and return BGR image (OpenCV format).
    """
    img = Image.open(image_path)
    try:
        orientation_tag = None
        for tag, name in ExifTags.TAGS.items():
            if name == "Orientation":
                orientation_tag = tag
                break

        exif = img._getexif()
        if exif is not None and orientation_tag is not None:
            val = exif.get(orientation_tag)
            if val == 3:
                img = img.rotate(180, expand=True)
            elif val == 6:
                img = img.rotate(270, expand=True)
            elif val == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_mask_to_image(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize mask to match original image using nearest neighbor"""
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def largest_valid_contour(mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    """Return largest external contour"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return c if cv2.contourArea(c) >= min_area else None


def contour_centroid(contour: np.ndarray) -> Optional[np.ndarray]:
    """Return centroid (cx, cy) """
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)


def select_single_most_central(
    masks_resized: List[np.ndarray],
    w: int,
    h: int,
    min_area: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    n1 logic: choose the single most central valid mask.
    """
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_dist = float("inf")

    for m in masks_resized:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        center = contour_centroid(c)
        if center is None:
            continue

        dist = float(np.linalg.norm(center - img_center))
        if dist < best_dist:
            best_dist = dist
            best = (m, c)

    return [best] if best is not None else []


def select_top_n_by_confidence(
    masks_resized_sorted: List[np.ndarray],
    n_out: int,
    min_area: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    n2 logic: take top-N masks by confidence 
    """
    selected: List[Tuple[np.ndarray, np.ndarray]] = []
    for m in masks_resized_sorted:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        selected.append((m, c))
        if len(selected) >= n_out:
            break
    return selected


def predict_masks_sorted(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Run YOLO segmentation.
    Returns:
      - orig_img (BGR) or None
      - masks_sorted: np.ndarray (N, mh, mw) sorted by confidence desc, or None
      - confs_sorted: np.ndarray (N,) sorted desc, or None
    """
    r = model.predict(source=image_bgr, conf=conf, iou=iou, save=False, verbose=False)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return None, None, None

    orig = r.orig_img.copy()
    masks = r.masks.data.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    order = np.argsort(-confs)
    return orig, masks[order], confs[order]