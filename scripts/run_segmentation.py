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
from nail_pipeline.seg import (
    combine_masks_binary,
    make_overlay,
    make_nail_only_crop_from_single_mask,
    write_contours_txt,)

MODEL_PATH = r"models\best.pt"
INPUT_DIR = r"0_old\0_datasets\example_sanquin\2"
OUTPUT_DIR = r"outputs\segmentationS"

CONF_THRESH = 0.15 # minnimum confidence for bounding box prediction
IOU_THRESH = 0.60 # maximum amount of overlapping possible between two bounding boxes
MIN_AREA = 50 # minnimum nail area for possible detection

NUM_MASKS_TO_OUTPUT = 4  # number of boxes to output per image: 1  => choose most central valid nail, >1 => choose top-N by confidence 

CONTOUR_MIN_AREA = 50
CONTOUR_APPROX_EPSILON_RATIO = 0.002 # lower = more precise contour, larger file

TEST_MODE = False
TEST_LIMIT = 30
BATCH_SPLIT = False # Optional split into two batches for stable memory usage


def process_one_image(model: YOLO, image_path: str, overlay_dir: str, txt_dir: str, nail_crop_dir: str):
    base = os.path.basename(image_path)
    stem, _ = os.path.splitext(base)

    corrected = fix_exif_rotation(image_path)
    orig, masks_sorted, _confs = predict_masks_sorted(model, corrected, conf=CONF_THRESH, iou=IOU_THRESH)

    if orig is None or masks_sorted is None:
        print(f"[WARN] No masks found for {base}")
        return

    im = orig
    h, w = im.shape[:2]

    masks_resized = [resize_mask_to_image(m, w, h) for m in masks_sorted]

    if NUM_MASKS_TO_OUTPUT == 1:
        selected_pairs = select_single_most_central(masks_resized, w, h, MIN_AREA)
    else:
        selected_pairs = select_top_n_by_confidence(masks_resized, NUM_MASKS_TO_OUTPUT, MIN_AREA)

    if not selected_pairs:
        print(f"[WARN] No valid mask after filtering for {base}")
        return

    selected_masks = [m for (m, _c) in selected_pairs]

    combined_mask_255 = combine_masks_binary(selected_masks)
    overlay = make_overlay(im, combined_mask_255, color_bgr=(0, 255, 0), alpha=0.5)

    overlay_path = os.path.join(overlay_dir, base)
    if not cv2.imwrite(overlay_path, overlay):
        print(f"[WARN] Could not write overlay file: {overlay_path}")

    saved_crops = 0
    for idx, single_mask in enumerate(selected_masks, start=1):
        single_mask_255 = (single_mask > 0).astype(np.uint8) * 255

        txt_path = os.path.join(txt_dir, f"{stem}_nail{idx}_mask.txt")
        write_contours_txt(
            mask_255=single_mask_255,
            txt_path=txt_path,
            min_area=CONTOUR_MIN_AREA,
            approx_epsilon_ratio=CONTOUR_APPROX_EPSILON_RATIO,
        )

        nail_crop = make_nail_only_crop_from_single_mask(im, single_mask_255)
        nail_crop_path = os.path.join(nail_crop_dir, f"{stem}_nail{idx}_crop.png")
        if cv2.imwrite(nail_crop_path, nail_crop):
            saved_crops += 1
        else:
            print(f"[WARN] Could not write nail_crop file: {nail_crop_path}")

    print(
        f"[OK] {base} -> selected {len(selected_masks)} nail(s) | "
        f"overlay: {overlay_path} | per-nail contours+crop saved: {saved_crops}"
    )


def process_images(image_paths: list[str], model: YOLO, overlay_dir: str, txt_dir: str, nail_crop_dir: str, label: str):
    print(f"\n[INFO] Starting {label} ({len(image_paths)} images)")
    for p in image_paths:
        process_one_image(model, p, overlay_dir, txt_dir, nail_crop_dir)
    print(f"[INFO] Finished {label}")


def main():
    overlay_dir = os.path.join(OUTPUT_DIR, "overlay")
    txt_dir = os.path.join(OUTPUT_DIR, "txt")
    nail_crop_dir = os.path.join(OUTPUT_DIR, "nail_crop")
    ensure_dirs(overlay_dir, txt_dir, nail_crop_dir)

    model = YOLO(MODEL_PATH)

    image_paths = list_images(INPUT_DIR)
    if TEST_MODE:
        image_paths = image_paths[:TEST_LIMIT]
        print(f"[INFO] TEST MODE: processing {len(image_paths)} image(s)")
    else:
        print(f"[INFO] FULL RUN: processing {len(image_paths)} image(s)")

    if not image_paths:
        print("[INFO] No images found. Exiting.")
        return

    if BATCH_SPLIT:
        mid = len(image_paths) // 2
        process_images(image_paths[:mid], model, overlay_dir, txt_dir, nail_crop_dir, "FIRST HALF")
        process_images(image_paths[mid:], model, overlay_dir, txt_dir, nail_crop_dir, "SECOND HALF")
    else:
        process_images(image_paths, model, overlay_dir, txt_dir, nail_crop_dir, "FULL SET")

    print("\n[DONE] Segmentation pipeline completed.")


if __name__ == "__main__":
    main()