# Nail Segmentation and Region Extraction Pipeline

This repository contains a set of Python pipelines for processing fingernail images with a YOLO segmentation model. It uses three main workflows:

1. **Oriented bounding box extraction** of the nail region plus adjacent skin
2. **Segmentation export** of nail masks, contours, and nail-only crops
3. **Refinement of segmented nail crops** using an ellipse-based postprocessing step

## Example output

Below is an example of the pipeline output:

![Example output](assets\output_example.png)

## Repository structure

repo/
├── nail_pipeline/
│   ├── **init**.py
│   ├── common.py
│   ├── obb.py
│   ├── seg.py
│   └── refine.py
├── scripts/
│   ├── **init**.py
│   ├── run_oriented_bb.py
│   ├── run_segmentation.py
│   └── run_refine_segmentation.py
├── models/
│   └── best.pt
└── data/
└── example/

## Pipelines

### 1. Oriented bounding box pipeline

`scripts/run_oriented_bb.py`

Runs YOLO segmentation, selects the nail mask(s), converts each selected mask into an oriented bounding box, optionally extends the box downward to include adjacent skin, and saves:

* `.txt` files with box corner coordinates and angle
* cropped nail + adjacent skin images

Default output:

* `1_nail_skin_bb/output/txt/`
* `1_nail_skin_bb/output/skin_crop/`

### 2. Segmentation pipeline

`scripts/run_segmentation.py`

Runs YOLO segmentation, selects the nail mask(s), and saves:

* overlay images
* contour `.txt` files
* nail-only segmented crops

Default output:

* `2_nail_segmentation/output/segmentation/overlay/`
* `2_nail_segmentation/output/segmentation/txt/`
* `2_nail_segmentation/output/segmentation/nail_crop/`

### 3. Refinement pipeline

`scripts/run_refine_segmentation.py`

Takes the `nail_crop` output from the segmentation pipeline and refines it using an adaptive ellipse:

1. infer foreground from non-black pixels
2. fit ellipse to the segmented nail
3. shrink the ellipse
4. cut top and bottom of the ellipse mask
5. apply the refined mask

Default output:

* `2_nail_segmentation/output/refined_segmentation/`

## Installation

Install dependencies:

pip install ultralytics opencv-python numpy Pillow

## Model

Place your trained model weights at:

`models/best.pt`

## Input data

By default, input images are read from:

`data/example/`

Supported formats:

* `.jpg`
* `.jpeg`
* `.png`

## Run

From the repository root:

### Oriented bounding boxes

python -m scripts.run_oriented_bb

### Segmentation outputs

python -m scripts.run_segmentation

### Refinement

python -m scripts.run_refine_segmentation

## Selection logic

The OBB and segmentation pipelines support:

* selecting the **single most central valid nail**
* or selecting the **top-N masks by confidence**

This is controlled by:

* `NUM_BOXES_TO_OUTPUT` in `run_oriented_bb.py`
* `NUM_MASKS_TO_OUTPUT` in `run_segmentation.py`

## Notes

* EXIF orientation is corrected before inference
* masks are resized to match the original image before contour analysis
* the refinement stage assumes black background outside the segmented nail crop

## Suggested .gitignore entries

**pycache**/
*.pyc

Optional:
1_nail_skin_bb/output/
2_nail_segmentation/output/
