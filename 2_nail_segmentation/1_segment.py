from ultralytics import YOLO
from PIL import Image, ExifTags
import glob
import os
import cv2
import numpy as np
from typing import List, Optional, Tuple

MODEL_PATH = r"best.pt"
INPUT_DIR = r"data\example"
OUTPUT_DIR = r"2_nail_segmentation\output\segmentation"

CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.60
MIN_AREA = 50

# - 1  => select most central valid nail 
# - >1 => select top-N by confidence 
NUM_MASKS_TO_OUTPUT = 1

# Contour export tuning
CONTOUR_MIN_AREA = 50
CONTOUR_APPROX_EPSILON_RATIO = 0.002  # lower = more precise contour, larger file

TEST_MODE = False
TEST_LIMIT = 30
TOTAL_CHUNKS = 10 # Optional chunk selection for very large datasets
CHUNK_INDEX: Optional[int] = None  # e.g., 9 for last 1/10th; None -> all images

# Optional split into two batches for stable memory usage
BATCH_SPLIT = False


def ensure_dirs(output_dir: str) -> Tuple[str, str, str]:
    overlay_dir = os.path.join(output_dir, "overlay")
    txt_dir = os.path.join(output_dir, "txt")
    nail_crop_dir = os.path.join(output_dir, "nail_crop")

    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(nail_crop_dir, exist_ok=True)

    return overlay_dir, txt_dir, nail_crop_dir


def list_images(input_dir: str) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, p)))
    return sorted(paths)


def apply_chunk_selection(image_paths: List[str], total_chunks: int, chunk_index: Optional[int]) -> List[str]:
    if chunk_index is None:
        return image_paths

    if total_chunks <= 0:
        raise ValueError("TOTAL_CHUNKS must be > 0")
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise ValueError(f"CHUNK_INDEX must be in [0, {total_chunks - 1}]")

    total = len(image_paths)
    chunk_size = total // total_chunks

    start = chunk_index * chunk_size
    end = (chunk_index + 1) * chunk_size if chunk_index < total_chunks - 1 else total
    return image_paths[start:end]


def fix_exif_rotation(image_path: str) -> np.ndarray:
    """
    Load image with EXIF orientation correction and return BGR image.
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


def resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)


def largest_valid_contour(mask: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return c


def contour_centroid(contour: np.ndarray) -> Optional[np.ndarray]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)


def select_single_most_central_mask(
    masks_resized: List[np.ndarray],
    w: int,
    h: int,
    min_area: int
) -> List[np.ndarray]:
    img_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    best_mask = None
    best_dist = float("inf")

    for m in masks_resized:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        center = contour_centroid(c)
        if center is None:
            continue

        dist = np.linalg.norm(center - img_center)
        if dist < best_dist:
            best_dist = dist
            best_mask = m

    return [best_mask] if best_mask is not None else []


def select_top_n_masks_by_confidence(
    masks_resized_sorted: List[np.ndarray],
    n_out: int,
    min_area: int
) -> List[np.ndarray]:
    selected = []
    for m in masks_resized_sorted:
        c = largest_valid_contour(m, min_area)
        if c is None:
            continue
        selected.append(m)
        if len(selected) >= n_out:
            break
    return selected


def combine_masks_binary(selected_masks: List[np.ndarray]) -> np.ndarray:
    if not selected_masks:
        raise ValueError("combine_masks_binary called with empty selected_masks")

    combined = np.zeros_like(selected_masks[0], dtype=np.uint8)
    for m in selected_masks:
        combined = np.logical_or(combined.astype(bool), m.astype(bool)).astype(np.uint8)
    return combined * 255


def make_overlay(
    image_bgr: np.ndarray,
    combined_mask_255: np.ndarray,
    color_bgr=(0, 255, 0),
    alpha=0.5
) -> np.ndarray:
    colored = np.zeros_like(image_bgr, dtype=np.uint8)
    colored[:] = np.array(color_bgr, dtype=np.uint8)

    overlay = image_bgr.copy().astype(np.float32)
    img_f = image_bgr.astype(np.float32)
    colored_f = colored.astype(np.float32)
    mask_bool = combined_mask_255.astype(bool)

    overlay[mask_bool] = (1 - alpha) * img_f[mask_bool] + alpha * colored_f[mask_bool]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_nail_only_crop_from_single_mask(image_bgr: np.ndarray, single_mask_255: np.ndarray) -> np.ndarray:
    """
    Return a tight crop containing only one nail (from one mask).
    Background is black.
    """
    mask_bool = single_mask_255 > 0
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    nail_only = np.zeros_like(image_bgr, dtype=np.uint8)
    nail_only[mask_bool] = image_bgr[mask_bool]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return nail_only[y_min:y_max + 1, x_min:x_max + 1]


def extract_contours_from_mask(
    mask_255: np.ndarray,
    min_area: int = 50,
    approx_epsilon_ratio: float = 0.002
) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        if approx_epsilon_ratio > 0:
            eps = approx_epsilon_ratio * cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, eps, True)

        c = c.reshape(-1, 2).astype(int)
        if len(c) >= 3:
            output.append(c)

    output.sort(
        key=lambda arr: cv2.contourArea(arr.reshape(-1, 1, 2).astype(np.int32)),
        reverse=True
    )
    return output


def write_contours_txt(
    mask_255: np.ndarray,
    txt_path: str,
    min_area: int = 50,
    approx_epsilon_ratio: float = 0.002
) -> None:
    h, w = mask_255.shape[:2]
    contours = extract_contours_from_mask(
        mask_255=mask_255,
        min_area=min_area,
        approx_epsilon_ratio=approx_epsilon_ratio
    )

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


def process_one_image(
    model: YOLO,
    image_path: str,
    overlay_dir: str,
    txt_dir: str,
    nail_crop_dir: str,
    num_masks_to_output: int
) -> None:
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)

    corrected = fix_exif_rotation(image_path)

    r = model.predict(
        source=corrected,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=False,
        verbose=False
    )[0]

    if r.masks is None or len(r.masks.data) == 0:
        print(f"[WARN] No masks found for {base}")
        return

    masks = r.masks.data.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    order = np.argsort(-confs)
    masks = masks[order]

    im = r.orig_img.copy()
    h, w = im.shape[:2]
    masks_resized = [resize_mask(m, w, h) for m in masks]

    if num_masks_to_output == 1:
        selected = select_single_most_central_mask(masks_resized, w, h, MIN_AREA)
    else:
        selected = select_top_n_masks_by_confidence(masks_resized, num_masks_to_output, MIN_AREA)

    if not selected:
        print(f"[WARN] No valid mask after filtering for {base}")
        return

    # Combined overlay (all selected nails)
    combined_mask = combine_masks_binary(selected)
    overlay = make_overlay(
        image_bgr=im,
        combined_mask_255=combined_mask,
        color_bgr=(0, 255, 0),
        alpha=0.5
    )
    overlay_path = os.path.join(overlay_dir, base)
    ok_overlay = cv2.imwrite(overlay_path, overlay)
    if not ok_overlay:
        print(f"[WARN] Could not write overlay file: {overlay_path}")

    # Save one txt + one crop PER selected nail
    saved_crops = 0
    for idx, single_mask in enumerate(selected, start=1):
        single_mask_255 = (single_mask > 0).astype(np.uint8) * 255

        # contours per nail
        txt_path = os.path.join(txt_dir, f"{stem}_nail{idx}_mask.txt")
        write_contours_txt(
            mask_255=single_mask_255,
            txt_path=txt_path,
            min_area=CONTOUR_MIN_AREA,
            approx_epsilon_ratio=CONTOUR_APPROX_EPSILON_RATIO
        )

        # crop per nail
        nail_crop = make_nail_only_crop_from_single_mask(im, single_mask_255)
        nail_crop_path = os.path.join(nail_crop_dir, f"{stem}_nail{idx}_crop.png")
        ok_nail_crop = cv2.imwrite(nail_crop_path, nail_crop)

        if not ok_nail_crop:
            print(f"[WARN] Could not write nail_crop file: {nail_crop_path}")
        else:
            saved_crops += 1

    print(
        f"[OK] {base} -> selected {len(selected)} nail(s) | "
        f"overlay: {overlay_path} | per-nail contours+crop saved: {saved_crops}"
    )


def process_batch(
    model: YOLO,
    image_paths: List[str],
    overlay_dir: str,
    txt_dir: str,
    nail_crop_dir: str,
    num_masks_to_output: int,
    batch_label: str
) -> None:
    print(f"\n[INFO] Starting {batch_label} ({len(image_paths)} images)")
    for path in image_paths:
        process_one_image(
            model=model,
            image_path=path,
            overlay_dir=overlay_dir,
            txt_dir=txt_dir,
            nail_crop_dir=nail_crop_dir,
            num_masks_to_output=num_masks_to_output
        )
    print(f"[INFO] Finished {batch_label}")


def main() -> None:
    if NUM_MASKS_TO_OUTPUT < 1:
        raise ValueError("NUM_MASKS_TO_OUTPUT must be >= 1")

    overlay_dir, txt_dir, nail_crop_dir = ensure_dirs(OUTPUT_DIR)
    model = YOLO(MODEL_PATH)

    image_paths = list_images(INPUT_DIR)
    image_paths = apply_chunk_selection(image_paths, TOTAL_CHUNKS, CHUNK_INDEX)

    if TEST_MODE:
        image_paths = image_paths[:TEST_LIMIT]
        print(f"[INFO] TEST MODE: processing {len(image_paths)} image(s)")
    else:
        print(f"[INFO] FULL RUN: processing {len(image_paths)} image(s)")

    if len(image_paths) == 0:
        print("[INFO] No images found. Exiting.")
        return

    if BATCH_SPLIT:
        mid = len(image_paths) // 2
        process_batch(
            model, image_paths[:mid], overlay_dir, txt_dir, nail_crop_dir,
            NUM_MASKS_TO_OUTPUT, "FIRST HALF"
        )
        process_batch(
            model, image_paths[mid:], overlay_dir, txt_dir, nail_crop_dir,
            NUM_MASKS_TO_OUTPUT, "SECOND HALF"
        )
    else:
        process_batch(
            model, image_paths, overlay_dir, txt_dir, nail_crop_dir,
            NUM_MASKS_TO_OUTPUT, "FULL SET"
        )

    print("\n[DONE] Segmentation pipeline completed.")


if __name__ == "__main__":
    main()
