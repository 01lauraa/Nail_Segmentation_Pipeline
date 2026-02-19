![Example outputs of the segmentation pipeline](assets/ex.png)

*From top to bottom: original hand images, extracted nail + adjacent skin crops, and refined nail-only outputs.*

# Nail Segmentation Pipeline (3 scripts)

This repo contains a YOLOv8-based pipeline that produces (1) nail+skin crops, (2) per-nail contours and nail-only crops, and (3) refined nail-only outputs.

## 1) `1_nail_skin_bb/boundingbox.py` — nail + adjacent skin crop
**Logic:** run YOLO segmentation → resize masks → select nail(s) (most central or top-N by confidence) → take largest contour → compute min-area rectangle → *(optional)* shift bottom edge down to include adjacent skin → polygon crop.

**Outputs**
- `1_nail_skin_bb/output/skin_crop/` polygon crops (`*_nail1...`)
- `1_nail_skin_bb/output/txt/` 4 corner points + `angle=...`

## 2) `2_nail_segmentation/1_segment.py` — masks, overlays, contours, nail-only crops
**Logic:** run YOLO segmentation → select nail mask(s) (most central or top-N) → save overlay for visualization → for each selected nail: export contour points to TXT + save tight nail-only crop (black background).

**Outputs**
- `2_nail_segmentation/output/segmentation/overlay/` overlays
- `2_nail_segmentation/output/segmentation/txt/` per-nail contours (`*_nail1_mask.txt`)
- `2_nail_segmentation/output/segmentation/nail_crop/` per-nail crops (`*_nail1_crop.png`)

## 3) `2_nail_segmentation/2_refine_segmentation.py` — geometric refinement (circle vs ellipse)
**Logic:** for each nail crop → build foreground mask from non-black pixels → fit **circle** and **ellipse** candidates → shrink shape → trim top/bottom → score each candidate *(kept_foreground − λ·leaked_background)* → keep best → apply mask.

**Outputs**
- `2_nail_segmentation/output/refined_segmentation/` refined nail-only outputs
