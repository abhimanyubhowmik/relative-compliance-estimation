# Relative Compliance Estimation of Wind-Excited Plants from Monocular Video

This is the code and data release for our ICRA 2026 workshop paper. The idea is simple: given a video of a plant branch blowing in the wind, how can we figure out which parts of the branch are stiffer and which are more flexible?
We track branch nodes through the video using TAPIR, fit a rank-1 compliance model to the bending angles, and get per-edge relative compliance values with bootstrap confidence intervals. No depth sensor, no force measurements, just a plain RGB video.
The TAPIR tracks are already computed and included, so you can run the full analysis pipeline without touching any point tracking.

---

## Getting started

```bash
pip install -r requirements.txt
```

That's it for the analysis pipeline. If you want to process your own ROS bag recordings, also uncomment `rosbags` and `lz4` in the requirements file.

---

## Reproducing the paper results

### Run the compliance analysis

This is the core step — smoothing, bending angle extraction, rank-1 model fit, bootstrap CIs:

```bash
# Plant 1
python scripts/analyze_branch_compliance.py \
  data/plant_1/tapir_tracks.csv \
  data/plant_1/branch_graph.json \
  --out-dir results/plant_1 \
  --video-path data/plant_1/video.mp4 \
  --smooth-window 5 \
  --use-factor-graph-smoother

# Plant 2
python scripts/analyze_branch_compliance.py \
  data/plant_2/tapir_tracks.csv \
  data/plant_2/branch_graph.json \
  --out-dir results/plant_2 \
  --video-path data/plant_2/video.mp4 \
  --smooth-window 5 \
  --use-factor-graph-smoother
```

The main outputs you'll care about are `compliance_summary.csv` (per-edge κ_e with 95% CI) and `annotated_nodes.mp4` (the source video with tracks overlaid). Everything else — bending metrics, node motion, zone summaries, smoothing comparisons — lands in the same output folder.

### Regenerate the paper figures

```bash
python scripts/make_paper_figures.py --data-dir results/plant_1 --out-dir figures/plant_1
python scripts/make_paper_figures.py --data-dir results/plant_2 --out-dir figures/plant_2
```

Figures 4–7 and the supplementary zone plot will appear as PDFs. The pre-generated ones are already in `figures/`.

### Render the smoothed track videos

The MA-smoothed track video is written automatically during the analysis step. For the factor-graph smoothed version:

```bash
python scripts/render_graph_tracks_video.py \
  data/plant_1/video.mp4 \
  results/plant_1/analysis_tracks.csv \
  data/plant_1/branch_graph.json \
  --out-path results/plant_1/annotated_nodes_factor_graph_smoothed.mp4
```

### Compliance heatmap overlays

Overlays compliance (and uncertainty) as a colour map on the first video frame:

```bash
python scripts/create_compliance_overlay.py results/plant_1 --video-path data/plant_1/video.mp4
python scripts/create_compliance_overlay.py results/plant_2 --video-path data/plant_2/video.mp4
```

### Hero video

The side-by-side video showing both plants with compliance colours live on the branch:

```bash
python scripts/make_hero_compliance_video.py \
  data/plant_1/video.mp4  results/plant_1/analysis_tracks.csv \
  data/plant_1/branch_graph.json  results/plant_1/compliance_summary.csv \
  data/plant_2/video.mp4  results/plant_2/analysis_tracks.csv \
  data/plant_2/branch_graph.json  results/plant_2/compliance_summary.csv \
  --out results/hero_compliance.mp4
```

### 3D point cloud (optional)

If you have the original RealSense bag file:

```bash
python scripts/create_compliance_pointcloud.py results/plant_1 /path/to/recording.bag --mode compliance
```

Or without a bag, using monocular depth estimation:

```bash
python scripts/create_monocular_compliance_pointcloud.py \
  results/plant_1 \
  --video-path data/plant_1/video.mp4 \
  --model transformers-depth-anything-v2 \
  --allow-downloads \
  --mode compliance
```

---

## Using it on your own plant

**Step 1 — Get your video.** If you're recording with a RealSense (or any other ROS bag setup), drop your `.bag` files into a folder and run:

```bash
python data_preparation/bags_to_mp4.py
```

This uses a pure Python bag reader — no ROS installation needed.

**Step 2 — Annotate the branch nodes.** Open the first frame and click along the branch:

```bash
python data_preparation/select_points.py data/my_plant/video.mp4 -o data/my_plant/annotation_points.json
```

Left-click to place a point, `s` to save, `q` to quit. The order matters — TAPIR assigns `point_id` 0, 1, 2, … in the click order, and you'll reference these IDs in the branch graph.

**Step 3 — Track the points with TAPIR.** We used [TAPIR](https://github.com/google-deepmind/tapnet) for point tracking. Download it and run your own tracking to generate a CSV with columns `point_id, frame, x, y`. See `data/plant_1/tapir_tracks.csv` for the expected format.

**Step 4 — Define the branch graph.** This tells the pipeline how the tracked nodes connect as a rooted tree:

```json
{
  "root": 5,
  "edges": [[5,4],[4,3],[3,2],[2,1],[1,0]]
}
```

`root` is the node at the base of the branch (closest to the pot or trunk). Every edge goes parent → child, i.e. root toward tip. See the included `branch_graph.json` files for real examples — plant 1 has 20 nodes, plant 2 has 25.

**Step 5 — Run the analysis** using the same commands as above with your own paths.

---

## A few parameters worth knowing

The defaults work well, but if your results look noisy or over-smoothed:

- `--smooth-window` (default 5) — moving-average pre-smoothing in frames. Increase for noisier tracks.
- `--use-factor-graph-smoother` — adds a second smoothing pass that also enforces edge-length consistency. Usually worth enabling.
- `--compliance-bootstrap-samples` (default 250) — more samples give tighter CI estimates but take longer.
- `--compliance-prior-strength` (default 12.0) — regularises edges with little motion toward κ_e = 1. Raise this if you see wild values on short or barely-moving edges.

---

## Citation

If this is useful for your work, please cite the paper:

```bibtex

@inproceedings{bhowmik2026uncertainty,
  title     = {Uncertainty-Aware Relative Compliance Estimation of Wind-Excited Plant from Monocular Video},
  author    = {Bhowmik, Abhimanyu and Behrens, Jan and Babuska, Robert},
  booktitle = {ICRA 2026 Workshop on Uncertainty in Open-World Robotics},
  year      = {2026}
}

```
