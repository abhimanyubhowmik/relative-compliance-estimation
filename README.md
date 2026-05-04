# Uncertainty-Aware Relative Compliance Estimation of Wind-Excited Plants from Monocular Video

Repository for the ICRA 2026 workshop paper of the same name.

Given a monocular video of a wind-excited plant branch with annotated nodes, the pipeline estimates a per-edge **relative compliance map** κ_e (how much each branch segment bends relative to others under the same wind load), together with 95% bootstrap confidence intervals.

Pre-computed TAPIR tracks, full analysis results, and publication figures for both plant experiments are included so all results can be reproduced without re-running point tracking.

---

## Repository structure

```
.
├── data/
│   ├── plant_1/
│   │   ├── video.mp4               # source RGB video (RealSense D435i)
│   │   ├── tapir_tracks.csv        # pre-computed TAPIR point tracks (point_id, frame, x, y)
│   │   ├── branch_graph.json       # rooted branch skeleton (20 nodes, root=11)
│   │   └── annotation_points.json  # initial node positions clicked on frame 0
│   └── plant_2/
│       ├── video.mp4
│       ├── tapir_tracks.csv
│       ├── branch_graph.json       # rooted branch skeleton (25 nodes, root=24)
│       └── annotation_points.json
├── results/
│   ├── plant_1/                    # full analysis output for plant 1
│   ├── plant_2/                    # full analysis output for plant 2
│   └── hero_compliance.mp4         # side-by-side compliance video (both plants)
├── figures/
│   ├── plant_1/                    # publication figures (PDF) for plant 1
│   └── plant_2/                    # publication figures (PDF) for plant 2
├── scripts/                        # analysis and visualisation pipeline
│   ├── analyze_branch_compliance.py          # core compliance estimation
│   ├── render_graph_tracks_video.py          # overlay smoothed tracks on video
│   ├── plot_tracks_pca_video.py              # PCA motion-arrow video
│   ├── make_paper_figures.py                 # generate all paper figures (Figs 4–7)
│   ├── create_compliance_overlay.py          # compliance heatmap overlays on first frame
│   ├── create_compliance_pointcloud.py       # 3D compliance point cloud (from RealSense bag)
│   ├── create_monocular_compliance_pointcloud.py  # 3D point cloud via monocular depth
│   └── make_hero_compliance_video.py         # side-by-side hero compliance video
└── data_preparation/               # tools for processing your own recordings
    ├── select_points.py            # interactive GUI to annotate node positions
    ├── bags_to_mp4.py              # convert ROS1 bag files to MP4 (no ROS install needed)
    ├── bag_reader.py               # low-level pure-Python bag reader
    ├── decompress_bag.py           # decompress lz4-compressed bags
    └── patch_rosbag_lz4.py        # monkey-patch rosbag for lz4 support
```

---

## Installation

```bash
pip install -r requirements.txt
```

Uncomment the optional sections in `requirements.txt` if you also want to:
- Process your own ROS bag files (`rosbags`, `lz4`)
- Generate monocular depth point clouds (`torch`, `transformers`, `Pillow`)

---

## Reproducing the results

All commands below run from the repository root.

### 1 — Run the compliance analysis

The TAPIR tracks are already provided. Run the core pipeline directly:

**Plant 1**
```bash
python scripts/analyze_branch_compliance.py \
  data/plant_1/tapir_tracks.csv \
  data/plant_1/branch_graph.json \
  --out-dir results/plant_1 \
  --video-path data/plant_1/video.mp4 \
  --smooth-window 5 \
  --use-factor-graph-smoother
```

**Plant 2**
```bash
python scripts/analyze_branch_compliance.py \
  data/plant_2/tapir_tracks.csv \
  data/plant_2/branch_graph.json \
  --out-dir results/plant_2 \
  --video-path data/plant_2/video.mp4 \
  --smooth-window 5 \
  --use-factor-graph-smoother
```

Key outputs written to `results/plant_X/`:

| File | Contents |
|------|----------|
| `compliance_summary.csv` | Per-edge κ_e, 95% CI bounds, root distance, zone |
| `bending_metrics.csv` | Per-bend angle RMS and CI |
| `node_metrics.csv` | Per-node relative motion RMS and CI |
| `analysis_tracks.csv` | Factor-graph smoothed tracks used in all downstream steps |
| `annotated_nodes.mp4` | Source video with MA-smoothed tracks overlaid |

### 2 — Render smoothed-track videos

```bash
# Factor-graph smoothed tracks (plant 1)
python scripts/render_graph_tracks_video.py \
  data/plant_1/video.mp4 \
  results/plant_1/analysis_tracks.csv \
  data/plant_1/branch_graph.json \
  --out-path results/plant_1/annotated_nodes_factor_graph_smoothed.mp4

# PCA motion arrows video
python scripts/plot_tracks_pca_video.py \
  data/plant_1/video.mp4 \
  data/plant_1/tapir_tracks.csv \
  --out results/plant_1/tapir_pca_video.mp4
```

### 3 — Generate paper figures

```bash
python scripts/make_paper_figures.py \
  --data-dir results/plant_1 \
  --out-dir figures/plant_1

python scripts/make_paper_figures.py \
  --data-dir results/plant_2 \
  --out-dir figures/plant_2
```

| Output file | Paper figure |
|-------------|-------------|
| `fig4_smoothing.pdf` | Fig. 4 — raw vs smoothed bending/edge metrics |
| `fig5_motion.pdf` | Fig. 5 — node motion and per-joint bending with CI |
| `fig6_compliance.pdf` | Fig. 6 — per-edge κ_e with CI bars |
| `fig7_excitation.pdf` | Fig. 7 — shared excitation proxy time series |
| `figS_zones.pdf` | Supplementary — zone-averaged metric bar charts |

### 4 — Generate compliance overlay images

```bash
python scripts/create_compliance_overlay.py \
  results/plant_1 \
  --video-path data/plant_1/video.mp4
```

Writes colour-coded PNG heatmaps to `results/plant_1/`:
- `relative_compliance_overlay_first_frame.png` — κ_e (blue=stiff, red=compliant)
- `relative_compliance_uncertainty_overlay_first_frame.png` — CI width / κ_e

### 5 — Generate the hero compliance video

```bash
python scripts/make_hero_compliance_video.py \
  data/plant_1/video.mp4 \
  results/plant_1/analysis_tracks.csv \
  data/plant_1/branch_graph.json \
  results/plant_1/compliance_summary.csv \
  data/plant_2/video.mp4 \
  results/plant_2/analysis_tracks.csv \
  data/plant_2/branch_graph.json \
  results/plant_2/compliance_summary.csv \
  --out results/hero_compliance.mp4
```

### 6 — 3D compliance point cloud (optional)

Requires the original RealSense bag file (not included due to size):

```bash
python scripts/create_compliance_pointcloud.py \
  results/plant_1 \
  /path/to/recording.bag \
  --mode compliance
```

Or using monocular depth estimation (no bag file needed):

```bash
# Via Hugging Face Transformers (downloads model on first run)
python scripts/create_monocular_compliance_pointcloud.py \
  results/plant_1 \
  --video-path data/plant_1/video.mp4 \
  --model transformers-depth-anything-v2 \
  --allow-downloads \
  --mode compliance
```

---

## Using your own plant recordings

### Step 1 — Convert ROS bag to video

```bash
# Place your .bag files in data/my_plant/bags/ and run:
python data_preparation/bags_to_mp4.py
# Output: MP4 video alongside the bag file
```

### Step 2 — Annotate branch nodes on the first frame

```bash
python data_preparation/select_points.py data/my_plant/video.mp4 \
  -o data/my_plant/annotation_points.json
```

Controls: **left-click** to add a point, **`s`** to save, **`q`** to quit.

Output is a JSON list of `[x, y]` pixel coordinates. TAPIR will assign `point_id` 0, 1, 2, … in click order — use these IDs when defining the branch graph.

### Step 3 — Track points with TAPIR

Download TAPIR (google-deepmind/tapnet) and run point tracking on your video to produce a CSV with columns `point_id, frame, x, y`. Use `data/plant_1/tapir_tracks.csv` as a reference for the expected format.

### Step 4 — Define the branch graph

Create a JSON file describing the rooted branch skeleton:

```json
{
  "root": 5,
  "edges": [
    [5, 4], [4, 3], [3, 2], [2, 1], [1, 0]
  ]
}
```

- `"root"` is the `point_id` of the branch base (closest to pot/trunk)
- Each edge `[parent, child]` must point **from root toward tip**

See `data/plant_1/branch_graph.json` (20 nodes) and `data/plant_2/branch_graph.json` (25 nodes) for reference.

### Step 5 — Run the analysis

Follow the same commands as in *Reproducing the results* above, pointing to your own data paths.

---

## Key analysis parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--smooth-window` | 5 | Moving-average pre-smoothing window (frames) |
| `--use-factor-graph-smoother` | off | Apply factor-graph smoother after MA pre-smoothing |
| `--compliance-prior-strength` | 12.0 | Log-shrinkage toward κ_e=1 for low-support edges |
| `--compliance-bootstrap-samples` | 250 | Bootstrap resamples for 95% confidence intervals |

---

## Citation

If you use this code or data, please cite:

```
Uncertainty-Aware Relative Compliance Estimation of Wind-Excited Plant from Monocular Video
ICRA 2026 Workshop
```
