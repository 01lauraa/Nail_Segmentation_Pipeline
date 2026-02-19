import os
import cv2
import numpy as np
from tqdm import tqdm

IMG_FOLDER = r"2_nail_segmentation\output\segmentation\nail_crop"
OUT_FOLDER = r"2_nail_segmentation\output\refined_segmentation"

# Shrink original mask to avoid including surrounding areas
SHRINK_RATIO = 0.95

# Cut top and bottom of mask 
TOP_CUT = 0.22       # fraction of fitted shape height
BOTTOM_CUT = 0.06    # fraction of fitted shape height

# Fit scoring
# score = kept_fg - LAMBDA_BG * leaked_bg
LAMBDA_BG = 1.2

MIN_CONTOUR_AREA = 20

SAVE_DEBUG_COMPARE = False
SAVE_DEBUG_MASKS = False

os.makedirs(OUT_FOLDER, exist_ok=True)
if SAVE_DEBUG_COMPARE:
    os.makedirs(os.path.join(OUT_FOLDER, "_debug_compare"), exist_ok=True)
if SAVE_DEBUG_MASKS:
    os.makedirs(os.path.join(OUT_FOLDER, "_debug_masks"), exist_ok=True)

def get_foreground_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Foreground from non-black pixels in nail_crop.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fg = (gray > 0).astype(np.uint8) * 255

    # light cleanup
    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
    return fg


def largest_contour(bin_mask: np.ndarray):
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return None
    return c


def fit_circle_to_fg(fg_mask: np.ndarray):
    """
    Fit circle to largest contour using minEnclosingCircle.
    Returns center (cx, cy), radius r.
    """
    cnt = largest_contour(fg_mask)
    h, w = fg_mask.shape[:2]
    if cnt is None:
        return (w // 2, h // 2, max(1, min(w, h) // 4))

    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))
    r = max(1, r)
    return (cx, cy, r)


def fit_oval_to_fg(fg_mask: np.ndarray):
    """
    Fit oval (ellipse) to largest contour.
    Returns center (cx,cy), axes (ax,ay) as semi-axes, angle.
    """
    cnt = largest_contour(fg_mask)
    h, w = fg_mask.shape[:2]
    if cnt is None:
        return (w // 2, h // 2, max(1, w // 4), max(1, h // 4), 0.0)

    if len(cnt) >= 5:
        (cx, cy), (major, minor), angle = cv2.fitEllipse(cnt)
        # fitEllipse gives full axis lengths -> semi-axes
        ax = max(1, int(round(major * 0.5)))
        ay = max(1, int(round(minor * 0.5)))
        return (int(round(cx)), int(round(cy)), ax, ay, float(angle))

    # fallback from bounding rect
    x, y, bw, bh = cv2.boundingRect(cnt)
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    ax = max(1, int(round(bw * 0.5)))
    ay = max(1, int(round(bh * 0.5)))
    return (int(round(cx)), int(round(cy)), ax, ay, 0.0)


def shrink_circle(cx: int, cy: int, r: int, shrink_ratio: float):
    r2 = max(1, int(round(r * shrink_ratio)))
    return cx, cy, r2


def shrink_oval(cx: int, cy: int, ax: int, ay: int, angle: float, shrink_ratio: float):
    ax2 = max(1, int(round(ax * shrink_ratio)))
    ay2 = max(1, int(round(ay * shrink_ratio)))
    return cx, cy, ax2, ay2, angle


def draw_circle_mask(shape, cx: int, cy: int, r: int) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def draw_oval_mask(shape, cx: int, cy: int, ax: int, ay: int, angle: float) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (ax, ay), angle, 0, 360, 255, -1)
    return m


def unified_cut_top_bottom(mask: np.ndarray, top_frac: float, bottom_frac: float) -> np.ndarray:
    """
    Apply the same vertical cut rule to any shape mask:
    - find mask's occupied vertical span [y_min, y_max]
    - remove top top_frac and bottom bottom_frac of that span
    """
    out = mask.copy()
    ys, xs = np.where(out > 0)
    if len(ys) == 0:
        return out

    y_min = int(ys.min())
    y_max = int(ys.max())
    height = y_max - y_min + 1

    top_cut_px = int(round(height * top_frac))
    bottom_cut_px = int(round(height * bottom_frac))

    y_cut_top = y_min + top_cut_px
    y_cut_bottom = y_max - bottom_cut_px

    out[:max(0, y_cut_top), :] = 0
    out[min(out.shape[0], y_cut_bottom + 1):, :] = 0
    return out


def score_candidate(candidate_mask: np.ndarray, fg_mask: np.ndarray, lambda_bg: float):
    cand = candidate_mask > 0
    fg = fg_mask > 0
    bg = ~fg

    kept_fg = int(np.count_nonzero(cand & fg))
    leaked_bg = int(np.count_nonzero(cand & bg))
    score = kept_fg - lambda_bg * leaked_bg
    return score, kept_fg, leaked_bg


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=mask)


def make_debug_triptych(inp, circ, oval, text):
    h, w = inp.shape[:2]
    canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
    canvas[:, :w] = inp
    canvas[:, w:2*w] = circ
    canvas[:, 2*w:] = oval

    cv2.putText(canvas, "input", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "circle", (w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, "oval", (2 * w + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return canvas


files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
print(f"Found {len(files)} images.")

for fname in tqdm(sorted(files)):
    path = os.path.join(IMG_FOLDER, fname)
    img = cv2.imread(path)

    if img is None:
        print(f"[WARN] Could not read {fname}")
        continue

    fg = get_foreground_mask(img)
    if np.count_nonzero(fg) == 0:
        # nothing to refine
        cv2.imwrite(os.path.join(OUT_FOLDER, fname), img)
        continue

    # -------- Candidate A: Circle --------
    cx, cy, r = fit_circle_to_fg(fg)                                  # 1) fit
    cx, cy, r = shrink_circle(cx, cy, r, SHRINK_RATIO)                # 2) shrink
    circle_mask = draw_circle_mask(img.shape, cx, cy, r)
    circle_mask = unified_cut_top_bottom(circle_mask, TOP_CUT, BOTTOM_CUT)  # 3) cut
    out_circle = apply_mask(img, circle_mask)

    # -------- Candidate B: Oval --------
    ocx, ocy, ax, ay, ang = fit_oval_to_fg(fg)                        # 1) fit
    ocx, ocy, ax, ay, ang = shrink_oval(ocx, ocy, ax, ay, ang, SHRINK_RATIO)  # 2) shrink
    oval_mask = draw_oval_mask(img.shape, ocx, ocy, ax, ay, ang)
    oval_mask = unified_cut_top_bottom(oval_mask, TOP_CUT, BOTTOM_CUT)        # 3) cut
    out_oval = apply_mask(img, oval_mask)

    # -------- Select better fit --------
    sc, kc, lc = score_candidate(circle_mask, fg, LAMBDA_BG)
    so, ko, lo = score_candidate(oval_mask, fg, LAMBDA_BG)

    if so > sc:
        chosen_name = "oval"
        chosen_mask = oval_mask
        chosen_out = out_oval
    else:
        chosen_name = "circle"
        chosen_mask = circle_mask
        chosen_out = out_circle

    cv2.imwrite(os.path.join(OUT_FOLDER, fname), chosen_out)

    if SAVE_DEBUG_MASKS:
        dbg_mask_name = os.path.splitext(fname)[0] + f"_chosen_{chosen_name}.png"
        cv2.imwrite(os.path.join(OUT_FOLDER, "_debug_masks", dbg_mask_name), chosen_mask)

    if SAVE_DEBUG_COMPARE:
        txt = f"C:{sc} (k{kc}/b{lc}) | O:{so} (k{ko}/b{lo}) -> {chosen_name}"
        dbg = make_debug_triptych(img, out_circle, out_oval, txt)
        cv2.imwrite(os.path.join(OUT_FOLDER, "_debug_compare", fname), dbg)

print(f"\nDone! Best-fit circle/oval outputs saved to: {OUT_FOLDER}")
