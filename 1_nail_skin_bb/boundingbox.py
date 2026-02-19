from ultralytics import YOLO
from PIL import Image, ExifTags
import os
import glob
import cv2
import numpy as np
from typing import List, Optional, Tuple

MODEL_PATH = r"best.pt"
INPUT_DIR = r"data\example"   
OUTPUT_DIR = r"1_nail_skin_bb\output"

CONF_THRESH = 0.15
IOU_THRESH = 0.60
MIN_AREA = 50

# Number of boxes to output per image:
# - 1  => choose most central valid nail 
# - >1 => choose top-N by confidence 
NUM_BOXES_TO_OUTPUT = 1

# Extent bounding box to include adjacent skin under nail
SHIFT_RATIO = 1.0
APPLY_SHIFT = True    

TEST_MODE = False
TEST_LIMIT = 30
BATCH_SPLIT = False     # split into two halves to keep memory usage stable


def ensure_dirs(out_dir: str) -> Tuple[str, str]:
    txt_dir = os.path.join(out_dir, "txt")
    crop_dir = os.path.join(out_dir, "skin_crop")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    return txt_dir, crop_dir


def list_images(input_dir: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(paths)


def fix_exif_rotation(image_path: str) -> np.ndarray:
    """
    Load image with EXIF orientation correction and return BGR np.ndarray.
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
        # If EXIF is missing/corrupted, continue with original orientation
        pass

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_mask_to_image(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def largest_valid_contour(mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def contour_center(contour: np.ndarray) -> Optional[Tuple[float, float]]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def box_from_contour(contour: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      - box_pts: np.ndarray shape (4,2), int
      - angle: normalized angle along long axis
      - h_box_long: long-side length after normalization (used for ratio shift)
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w_box, h_box), angle = rect

    # Normalize so angle follows long axis
    if w_box < h_box:
        angle += 90.0
        w_box, h_box = h_box, w_box

    box = cv2.boxPoints(((cx, cy), (w_box, h_box), angle))
    box = np.intp(box)
    return box, angle, h_box


def shift_bottom_two_vertices(box_pts: np.ndarray, shift_px: int, img_h: int) -> np.ndarray:
    """
    Shift the two bottom-most vertices downward in y direction.
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
    return patch[y:y + hc, x:x + wc]


def select_masks_single_central(
    masks_resized: List[np.ndarray],
    img_w: int,
    img_h: int,
    min_area: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    n1 logic: choose the single most central valid mask.
    Returns list of (mask, contour) with length 0 or 1.
    """
    img_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32)
    best = None
    best_dist = float("inf")

    for m in masks_resized:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        cc = contour_center(c)
        if cc is None:
            continue
        dist = np.linalg.norm(np.array(cc, dtype=np.float32) - img_center)
        if dist < best_dist:
            best_dist = dist
            best = (m, c)

    return [best] if best is not None else []


def select_masks_top_n_confidence(
    masks_resized_sorted: List[np.ndarray],
    n_out: int,
    min_area: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    n2 logic: take top-N masks by confidence (already sorted), keep valid contours only.
    """
    selected = []
    for m in masks_resized_sorted:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        selected.append((m, c))
        if len(selected) >= n_out:
            break
    return selected


def process_one_image(
    model: YOLO,
    image_path: str,
    txt_dir: str,
    crop_dir: str,
    num_boxes_to_output: int
) -> None:
    base = os.path.basename(image_path)
    stem, ext = os.path.splitext(base)
    txt_path = os.path.join(txt_dir, f"{stem}.txt")

    corrected = fix_exif_rotation(image_path)

    pred = model.predict(
        source=corrected,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        verbose=False
    )[0]

    if pred.masks is None or len(pred.masks.data) == 0:
        print(f"[WARN] No nails found: {base}")
        return

    image = pred.orig_img.copy()
    h, w = image.shape[:2]

    masks = pred.masks.data.cpu().numpy()
    confs = pred.boxes.conf.cpu().numpy()
    order = np.argsort(-confs)  # descending confidence
    masks = masks[order]

    masks_resized = [resize_mask_to_image(m, w, h) for m in masks]

    if num_boxes_to_output == 1:
        selected = select_masks_single_central(masks_resized, w, h, MIN_AREA)
    else:
        selected = select_masks_top_n_confidence(masks_resized, num_boxes_to_output, MIN_AREA)

    if not selected:
        print(f"[WARN] No valid contour(s): {base}")
        return

    lines = []
    saved_count = 0

    for idx, (_m, contour) in enumerate(selected, start=1):
        box, angle, h_box = box_from_contour(contour)

        if APPLY_SHIFT:
            shift_px = int(SHIFT_RATIO * h_box) 
            box = shift_bottom_two_vertices(box, shift_px, h)

        coords = " ".join([f"{int(x)},{int(y)}" for x, y in box])
        lines.append(f"{coords} angle={angle:.2f}")

        crop = polygon_crop(image, box)
        crop_name = f"{stem}_nail{idx}{ext}"
        crop_path = os.path.join(crop_dir, crop_name)
        cv2.imwrite(crop_path, crop)
        saved_count += 1

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] {base}: saved {saved_count} crop(s) + txt")


def process_images(image_paths: List[str], model: YOLO, txt_dir: str, crop_dir: str, num_boxes_to_output: int) -> None:
    for path in image_paths:
        process_one_image(
            model=model,
            image_path=path,
            txt_dir=txt_dir,
            crop_dir=crop_dir,
            num_boxes_to_output=num_boxes_to_output
        )


def main() -> None:
    if NUM_BOXES_TO_OUTPUT < 1:
        raise ValueError("NUM_BOXES_TO_OUTPUT must be >= 1")

    txt_dir, crop_dir = ensure_dirs(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)

    image_paths = list_images(INPUT_DIR)
    if TEST_MODE:
        image_paths = image_paths[:TEST_LIMIT]
        print(f"[INFO] TEST MODE: processing {len(image_paths)} image(s)")
    else:
        print(f"[INFO] FULL RUN: processing {len(image_paths)} image(s)")

    if not image_paths:
        print("[INFO] No images found. Nothing to do.")
        return

    if BATCH_SPLIT:
        mid = len(image_paths) // 2
        first_half = image_paths[:mid]
        second_half = image_paths[mid:]

        print(f"[INFO] Starting FIRST HALF ({len(first_half)} images)")
        process_images(first_half, model, txt_dir, crop_dir, NUM_BOXES_TO_OUTPUT)

        print(f"[INFO] Starting SECOND HALF ({len(second_half)} images)")
        process_images(second_half, model, txt_dir, crop_dir, NUM_BOXES_TO_OUTPUT)
    else:
        process_images(image_paths, model, txt_dir, crop_dir, NUM_BOXES_TO_OUTPUT)

    print("[DONE] All images processed.")


if __name__ == "__main__":
    main()
