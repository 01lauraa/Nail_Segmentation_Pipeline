from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2
import numpy as np
from ultralytics import YOLO
from nail_pipeline.common import (
    ensure_dirs,
    fix_exif_rotation,
    list_images,
    predict_masks_sorted,
    resize_mask_to_image,
    select_single_most_central,
    select_top_n_by_confidence,
)
from nail_pipeline.obb import box_from_contour, polygon_crop, shift_bottom_two_vertices

MODEL_PATH = r"models\best.pt"
INPUT_DIR = r"data\example"
OUTPUT_DIR = r"outputs\obb"

CONF_THRESH = 0.15 # minnimum confidence for bounding box prediction
IOU_THRESH = 0.60 # maximum amount of overlapping possible between two bounding boxes
MIN_AREA = 50 # minnimum nail area for possible detection

NUM_BOXES_TO_OUTPUT = 1  # number of boxes to output per image: 1  => choose most central valid nail, >1 => choose top-N by confidence 

APPLY_SHIFT = True # extend bounding box to include adjacent skin under nail
SHIFT_RATIO = 1.0  

TEST_MODE = False
TEST_LIMIT = 30
BATCH_SPLIT = False  # split into two halves to keep memory usage stable


def process_one_image(model: YOLO, image_path: str, txt_dir: str, crop_dir: str):
    base = os.path.basename(image_path)
    stem, ext = os.path.splitext(base)
    txt_path = os.path.join(txt_dir, f"{stem}.txt")

    corrected = fix_exif_rotation(image_path)
    orig, masks_sorted, _confs = predict_masks_sorted(model, corrected, conf=CONF_THRESH, iou=IOU_THRESH)

    if orig is None or masks_sorted is None:
        print(f"[WARN] No nails found: {base}")
        return

    image = orig
    h, w = image.shape[:2]

    masks_resized = [resize_mask_to_image(m, w, h) for m in masks_sorted]

    if NUM_BOXES_TO_OUTPUT == 1:
        selected = select_single_most_central(masks_resized, w, h, MIN_AREA)
    else:
        selected = select_top_n_by_confidence(masks_resized, NUM_BOXES_TO_OUTPUT, MIN_AREA)

    if not selected:
        print(f"[WARN] No valid contour(s): {base}")
        return

    lines = []
    saved_count = 0

    for idx, (_m, contour) in enumerate(selected, start=1):
        box, angle, h_box_short = box_from_contour(contour)

        if APPLY_SHIFT:
            shift_px = int(SHIFT_RATIO * h_box_short)
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


def process_images(image_paths: list[str], model: YOLO, txt_dir: str, crop_dir: str):
    for p in image_paths:
        process_one_image(model, p, txt_dir, crop_dir)


def main():
    txt_dir = os.path.join(OUTPUT_DIR, "txt")
    crop_dir = os.path.join(OUTPUT_DIR, "skin_crop")
    ensure_dirs(txt_dir, crop_dir)

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
        print(f"[INFO] Starting FIRST HALF ({mid} images)")
        process_images(image_paths[:mid], model, txt_dir, crop_dir)
        print(f"[INFO] Starting SECOND HALF ({len(image_paths) - mid} images)")
        process_images(image_paths[mid:], model, txt_dir, crop_dir)
    else:
        process_images(image_paths, model, txt_dir, crop_dir)

    print("[DONE] All images processed.")


if __name__ == "__main__":
    main()