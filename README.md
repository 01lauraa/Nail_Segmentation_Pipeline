<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Nail Segmentation Pipeline (YOLOv8 + Geometric Refinement)</title>
  <meta name="description" content="YOLOv8-based nail segmentation pipeline with oriented crops and geometric refinement." />
  <style>
    :root {
      --bg: #0b0f14;
      --panel: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #60a5fa;
      --border: rgba(255,255,255,0.08);
      --code: #0f172a;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: var(--sans);
      background: radial-gradient(1200px 800px at 20% 10%, rgba(96,165,250,0.16), transparent 55%),
                  radial-gradient(900px 600px at 90% 20%, rgba(34,197,94,0.10), transparent 55%),
                  var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    .wrap {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 18px 56px;
    }
    header {
      border: 1px solid var(--border);
      background: rgba(17,24,39,0.72);
      backdrop-filter: blur(8px);
      border-radius: 16px;
      padding: 22px 20px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 1.7rem;
      letter-spacing: 0.2px;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
    }
    .pillrow {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      background: rgba(15,23,42,0.55);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.9rem;
    }
    .pill b { color: var(--accent); font-weight: 600; }
    section {
      margin-top: 18px;
      border: 1px solid var(--border);
      background: rgba(17,24,39,0.58);
      backdrop-filter: blur(8px);
      border-radius: 16px;
      padding: 18px 18px;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 1.25rem;
    }
    h3 {
      margin: 18px 0 8px;
      font-size: 1.05rem;
      color: #dbeafe;
    }
    p { margin: 10px 0; }
    ul { margin: 10px 0 10px 20px; }
    li { margin: 6px 0; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, pre {
      font-family: var(--mono);
      font-size: 0.92rem;
    }
    pre {
      margin: 12px 0;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: auto;
    }
    .note {
      padding: 10px 12px;
      border-left: 4px solid var(--accent);
      background: rgba(96,165,250,0.08);
      border-radius: 10px;
      color: var(--text);
      margin: 12px 0;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    @media (min-width: 900px) {
      .grid { grid-template-columns: 1.2fr 0.8fr; }
    }
    figure {
      margin: 0;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      background: rgba(0,0,0,0.25);
    }
    figure img {
      width: 100%;
      display: block;
      height: auto;
    }
    figcaption {
      padding: 10px 12px;
      color: var(--muted);
      font-size: 0.92rem;
      border-top: 1px solid var(--border);
      background: rgba(17,24,39,0.55);
    }
    .tree {
      margin: 12px 0 0;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: auto;
      white-space: pre;
    }
    footer {
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.9rem;
      text-align: center;
    }
    .small { font-size: 0.92rem; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Nail Segmentation Pipeline (YOLOv8 + Geometric Refinement)</h1>
      <p class="subtitle">
        End-to-end pipeline to extract nail + adjacent skin crops, generate nail segmentation outputs, and refine masks using circle/ellipse fitting.
      </p>
      <div class="pillrow">
        <span class="pill"><b>Stage 1</b> Oriented nail/skin crop</span>
        <span class="pill"><b>Stage 2</b> Segmentation + contour export</span>
        <span class="pill"><b>Stage 3</b> Circle vs ellipse refinement</span>
      </div>
    </header>

    <section>
      <div class="grid">
        <div>
          <h2>Overview</h2>
          <p>
            This repository contains a reproducible computer vision pipeline built around a YOLOv8 segmentation model.
            It is intended for real-world hand images where multiple nails may be visible and raw masks may include surrounding skin.
          </p>
          <ul>
            <li><b>YOLOv8 segmentation</b> for nail detection / masks</li>
            <li><b>Oriented bounding-box crops</b> (min-area rectangle from mask contour) with optional downward extension</li>
            <li><b>Geometric refinement</b>: fit <b>circle vs ellipse</b> to foreground, shrink, trim top/bottom, select best fit</li>
          </ul>

          <div class="note">
            <b>Image in this page:</b> place your output figure at <code>assets/segmentation_outputs.png</code> (relative to this HTML file).
          </div>
        </div>

        <figure>
          <img src="assets/segmentation_outputs.png" alt="Example outputs of the segmentation pipeline">
          <figcaption>
            Example outputs (top → bottom): original hand images, extracted nail + adjacent skin regions, refined nail-only segmentations.
          </figcaption>
        </figure>
      </div>
    </section>

    <section>
      <h2>Repository structure</h2>
      <div class="tree">.
├── best.pt                              # YOLOv8 segmentation weights (optional / local)
├── example_dataset_valles/              # example input images (optional)
├── 1_nail_skin_bb/
│   ├── boundingbox.py
│   └── output/
│       ├── skin_crop/                   # polygon crops: nail + adjacent skin
│       └── txt/                         # oriented box vertices + angle per image
└── 2_nail_segmentation/
    ├── 1_segment.py
    ├── 2_refine_segmentation.py
    └── output/
        ├── segmentation/
        │   ├── overlay/                 # overlay visualization of selected mask(s)
        │   ├── txt/                     # per-nail contour exports
        │   └── nail_crop/               # tight nail-only crops (black background)
        └── refined_segmentation/        # refined nail-only outputs</div>
      <p class="small">
        Outputs are generated into <code>output/</code> folders by the scripts and should typically be ignored by git.
      </p>
    </section>

    <section>
      <h2>Setup</h2>
      <h3>1) Create and activate a virtual environment</h3>
      <pre><code>python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate</code></pre>

      <h3>2) Install dependencies</h3>
      <pre><code>pip install -r requirements.txt</code></pre>

      <h3>requirements.txt (example)</h3>
      <pre><code>ultralytics&gt;=8.0.0
opencv-python&gt;=4.7.0
numpy&gt;=1.23.0
Pillow&gt;=9.0.0
tqdm&gt;=4.64.0</code></pre>
    </section>

    <section>
      <h2>Model weights</h2>
      <p>
        The scripts expect a YOLOv8 segmentation model at <code>best.pt</code>.
        If your weights are stored elsewhere, update <code>MODEL_PATH</code> inside the scripts.
      </p>
      <div class="note">
        If <code>best.pt</code> is large or trained on private data, do <b>not</b> commit it to GitHub.
        Keep it local and/or provide a download link separately.
      </div>
    </section>

    <section>
      <h2>Run the pipeline</h2>

      <h3>Step 1 — Oriented nail + adjacent skin crop</h3>
      <p>
        Detect nails, compute an oriented bounding box from the selected mask contour,
        optionally extend the lower edge to include adjacent skin, and save polygon crops + box coordinates.
      </p>
      <pre><code>python 1_nail_skin_bb/boundingbox.py</code></pre>
      <p><b>Outputs</b></p>
      <ul>
        <li><code>1_nail_skin_bb/output/skin_crop/</code> → <code>*_nail1.jpg/png</code>, <code>*_nail2...</code></li>
        <li><code>1_nail_skin_bb/output/txt/</code> → per-image <code>.txt</code> with 4 corner points and <code>angle=...</code></li>
      </ul>
      <p><b>Selection behavior</b></p>
      <ul>
        <li><code>NUM_BOXES_TO_OUTPUT = 1</code> selects the <b>most central</b> valid nail</li>
        <li><code>NUM_BOXES_TO_OUTPUT &gt; 1</code> selects <b>top-N</b> by confidence</li>
        <li><code>APPLY_SHIFT=True</code> + <code>SHIFT_RATIO</code> extends the crop downward (adjacent skin)</li>
      </ul>

      <h3>Step 2 — Segmentation + contour export + nail-only crops</h3>
      <p>
        Run YOLO segmentation, select mask(s), save an overlay, export per-nail contour points, and save tight nail-only crops.
      </p>
      <pre><code>python 2_nail_segmentation/1_segment.py</code></pre>
      <p><b>Outputs</b></p>
      <ul>
        <li><code>2_nail_segmentation/output/segmentation/overlay/</code> → overlay image(s)</li>
        <li><code>2_nail_segmentation/output/segmentation/txt/</code> → <code>*_nail1_mask.txt</code> with contour points</li>
        <li><code>2_nail_segmentation/output/segmentation/nail_crop/</code> → <code>*_nail1_crop.png</code> (black background)</li>
      </ul>

      <h3>Step 3 — Refinement (circle vs ellipse fit)</h3>
      <p>
        Refine each nail crop by fitting a circle and ellipse candidate to the foreground, shrinking the fitted shape,
        trimming top/bottom, scoring both candidates, and saving the best fit.
      </p>
      <pre><code>python 2_nail_segmentation/2_refine_segmentation.py</code></pre>
      <p><b>Input</b></p>
      <ul>
        <li><code>2_nail_segmentation/output/segmentation/nail_crop/</code></li>
      </ul>
      <p><b>Output</b></p>
      <ul>
        <li><code>2_nail_segmentation/output/refined_segmentation/</code></li>
      </ul>
    </section>

    <section>
      <h2>Notes</h2>
      <ul>
        <li>EXIF rotation is corrected before inference.</li>
        <li>For quick testing, enable <code>TEST_MODE=True</code> inside the scripts.</li>
        <li>Make sure <code>INPUT_DIR</code> values match your local folder names (e.g., <code>example_dataset_valles</code>).</li>
      </ul>
    </section>

    <section>
      <h2>License</h2>
      <p>
        Add a <code>LICENSE</code> file (e.g., MIT) if you want others to reuse the code.
      </p>
    </section>

    <footer>
      <p>Tip: For GitHub, you can also include a <code>README.md</code> in addition to this HTML file.</p>
    </footer>
  </div>
</body>
</html>
