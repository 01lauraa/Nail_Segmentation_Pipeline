from __future__ import annotations
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nail_pipeline.refine import RefineConfig, refine_folder


IMG_FOLDER = r"outputs\segmentationS\nail_crop"
OUT_FOLDER = r"outputs\refined_segmentationS"

cfg = RefineConfig(
    shrink_ratio=0.80,
    top_cut=0.22,
    bottom_cut=0.06,
    min_contour_area=20
)


def main():
    refine_folder(
        img_folder=IMG_FOLDER,
        out_folder=OUT_FOLDER,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()