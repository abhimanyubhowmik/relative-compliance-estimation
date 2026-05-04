#!/usr/bin/env python3
"""
Hero compliance video: two plants side-by-side with a static compliance map
overlay derived directly from the saved branch compliance analysis results.

kappa_e values are loaded from compliance_summary.csv (no ALS recomputation).
Plant 1 is time-scaled so both recordings fill the same output duration.
Each panel fades from grey to its final compliance colours over a short warmup:
  Plant 1: --warmup1  seconds  (default 0.04 s)
  Plant 2: --warmup2  seconds  (default 0.03 s)

Edge rendering is an exact match to create_compliance_overlay.py.

Usage:
    python scripts/make_hero_compliance_video.py \
        Data/20260212_180454_video_rgb.mp4 \
        results/branch_compliance_plant1_rgb/analysis_tracks.csv \
        data_preparation/tapir_branch_graph_template.json \
        results/branch_compliance_plant_1/compliance_summary.csv \
        Data/20260503_120645_video.mp4 \
        results/branch_compliance_plant_2/analysis_tracks.csv \
        data_preparation/branch_graph_stem_2.json \
        results/branch_compliance_plant_2/compliance_summary.csv \
        --out results/hero_compliance.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EPS = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers — identical to create_compliance_overlay.py
# ─────────────────────────────────────────────────────────────────────────────

def _color_interp(stops, t):
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            s = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return tuple(int(round((1 - s) * c0[j] + s * c1[j])) for j in range(3))
    return stops[-1][1]


def compliance_color(t):
    return _color_interp(
        [(0.0, (180, 60, 30)), (0.5, (80, 220, 240)), (1.0, (30, 30, 210))], t)


def _blend(c0, c1, alpha):
    return tuple(int(round((1 - alpha) * c0[j] + alpha * c1[j])) for j in range(3))


def _smoothstep(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


# ─────────────────────────────────────────────────────────────────────────────
# Legend — identical to create_compliance_overlay.py draw_legend()
# ─────────────────────────────────────────────────────────────────────────────

def draw_legend(canvas, title, vmin, vmax, color_fn, tick_fmt="{:.2f}"):
    h, w     = canvas.shape[:2]
    box_w    = 18
    box_h    = min(145, h - 108)
    right_mg = 8
    x0       = w - 88
    y0       = 58
    label_x  = x0 + box_w + 10

    ov = canvas.copy()
    cv2.rectangle(ov, (x0 - 10, 18), (w - right_mg, y0 + box_h + 28), (18, 18, 18), -1)
    canvas[:] = cv2.addWeighted(ov, 0.45, canvas, 0.55, 0)

    for i in range(box_h):
        t = 1.0 - (i / max(box_h - 1, 1))
        cv2.line(canvas, (x0, y0 + i), (x0 + box_w, y0 + i), color_fn(t), 1)

    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)

    words = title.split()
    lines = (words if len(words) <= 2
             else [" ".join(words[:(len(words) + 1) // 2]),
                   " ".join(words[(len(words) + 1) // 2:])])
    for li, line in enumerate(lines):
        cv2.putText(canvas, line, (x0 - 8, 32 + li * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)

    for t, value in [(0.0, vmax), (0.5, 0.5 * (vmin + vmax)), (1.0, vmin)]:
        yy = int(round(y0 + t * box_h))
        cv2.line(canvas, (x0 + box_w + 3, yy), (x0 + box_w + 8, yy), (255, 255, 255), 1)
        cv2.putText(canvas, tick_fmt.format(value), (label_x, yy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.29, (245, 245, 245), 1, cv2.LINE_AA)


def draw_title(canvas, title, subtitle):
    ts = cv2.getTextSize(title,    cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0]
    ss = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0]
    tw = max(ts[0], ss[0]) + 24
    ov = canvas.copy()
    cv2.rectangle(ov, (18, 18), (18 + tw, 64), (15, 15, 15), -1)
    canvas[:] = cv2.addWeighted(ov, 0.45, canvas, 0.55, 0.0)
    cv2.putText(canvas, title,    (28, 39), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (29, 57), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (220, 220, 220), 1, cv2.LINE_AA)


def draw_speed_label(canvas, label, x_center, frame_h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.45, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    pad = 6
    x0, y0 = x_center - tw // 2 - pad, frame_h - th - 2 * pad - 6
    x1, y1 = x_center + tw // 2 + pad, frame_h - 6
    ov = canvas.copy()
    cv2.rectangle(ov, (x0, y0), (x1, y1), (15, 15, 15), -1)
    canvas[:] = cv2.addWeighted(ov, 0.50, canvas, 0.50, 0.0)
    cv2.putText(canvas, label, (x_center - tw // 2, y1 - pad),
                font, scale, (220, 220, 220), thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_kappa_map(compliance_csv: Path) -> dict:
    """Return {(parent, child): kappa_e} from compliance_summary.csv."""
    df = pd.read_csv(compliance_csv)
    return {
        (int(r.parent), int(r.child)): float(r.relative_compliance_kappa)
        for r in df.itertuples(index=False)
        if np.isfinite(float(r.relative_compliance_kappa))
    }


def load_tracks(tracks_csv: Path) -> tuple[list, dict]:
    """Return (sorted_frame_ids, {frame: {node_id: (x, y)}})."""
    df  = pd.read_csv(tracks_csv)
    lu  = {}
    for row in df.itertuples(index=False):
        lu.setdefault(int(row.frame), {})[int(row.point_id)] = (
            float(row.x), float(row.y))
    return sorted(lu.keys()), lu


def load_edges(graph_json: Path) -> list:
    g = json.loads(graph_json.read_text())
    return [(int(e[0]), int(e[1])) for e in g["edges"]]


# ─────────────────────────────────────────────────────────────────────────────
# Draw compliance overlay — exact match to create_compliance_overlay.py
# ─────────────────────────────────────────────────────────────────────────────

def draw_compliance(frame, coords, edges, kappa_map, vmin, vmax, fade):
    """
    Overlay compliance-coloured edges on `frame` (BGR, in-place).
    `fade` in [0,1]: 0 = grey, 1 = full compliance colour.
    Line thickness and blend are identical to create_compliance_overlay.py.
    """
    overlay = frame.copy()
    for parent, child in edges:
        if parent not in coords or child not in coords:
            continue
        p0 = tuple(np.round(np.array(coords[parent])).astype(int))
        p1 = tuple(np.round(np.array(coords[child])).astype(int))

        kappa = kappa_map.get((parent, child), np.nan)
        if np.isfinite(kappa):
            t     = float(np.clip((kappa - vmin) / max(vmax - vmin, EPS), 0, 1))
            color = _blend((180, 180, 180), compliance_color(t), fade)
        else:
            color = (180, 180, 180)

        cv2.line(overlay, p0, p1, color, 16, cv2.LINE_AA)
        cv2.line(overlay, p0, p1, color, 10, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0.0, frame)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p1_video");    ap.add_argument("p1_tracks")
    ap.add_argument("p1_graph");    ap.add_argument("p1_compliance")
    ap.add_argument("p2_video");    ap.add_argument("p2_tracks")
    ap.add_argument("p2_graph");    ap.add_argument("p2_compliance")
    ap.add_argument("--out",      default="results/hero_compliance.mp4")
    ap.add_argument("--fps",      type=float, default=30.0)
    ap.add_argument("--warmup1",  type=float, default=4.0,
                    help="fade-in duration for Plant 1 in seconds (default 4.0)")
    ap.add_argument("--warmup2",  type=float, default=3.0,
                    help="fade-in duration for Plant 2 in seconds (default 3.0)")
    args = ap.parse_args()

    fps = args.fps
    warmup_frames1 = max(1, round(args.warmup1 * fps))
    warmup_frames2 = max(1, round(args.warmup2 * fps))
    print(f"Warmup: Plant 1 = {args.warmup1}s ({warmup_frames1} frame(s)),  "
          f"Plant 2 = {args.warmup2}s ({warmup_frames2} frame(s))")

    # Load kappa values from saved analysis
    kappa1 = load_kappa_map(Path(args.p1_compliance))
    kappa2 = load_kappa_map(Path(args.p2_compliance))
    print(f"Plant 1: {len(kappa1)} edges with finite kappa")
    print(f"Plant 2: {len(kappa2)} edges with finite kappa")

    # Load tracks and graph edges
    frames1, lu1 = load_tracks(Path(args.p1_tracks))
    frames2, lu2 = load_tracks(Path(args.p2_tracks))
    edges1 = load_edges(Path(args.p1_graph))
    edges2 = load_edges(Path(args.p2_graph))
    n1, n2 = len(frames1), len(frames2)

    # Per-plant colour scales — identical to create_compliance_overlay.py normalize()
    # so each panel matches its own static overlay image exactly.
    def _scale(kmap):
        vals = np.array(list(kmap.values()))
        vals = vals[np.isfinite(vals)]
        lo, hi = float(vals.min()), float(vals.max())
        return lo, hi if abs(hi - lo) > 1e-9 else hi + 1.0

    vmin1, vmax1 = _scale(kappa1)
    vmin2, vmax2 = _scale(kappa2)
    print(f"Colour scale  Plant 1: [{vmin1:.2f}, {vmax1:.2f}]")
    print(f"Colour scale  Plant 2: [{vmin2:.2f}, {vmax2:.2f}]")

    # Output duration = plant 2 (shorter); plant 1 time-scaled to match
    n_out    = n2
    speedup1 = n1 / n2      # ≈ 1.87×
    speedup2 = 1.0
    print(f"Output: {n_out} frames ({n_out/fps:.1f} s) @ {fps:.0f} fps")
    print(f"  Plant 1: {n1} src frames → x{speedup1:.2f} speed")
    print(f"  Plant 2: {n2} src frames → x{speedup2:.2f} speed")

    # Video dimensions
    cap_tmp = cv2.VideoCapture(str(args.p1_video))
    W = int(cap_tmp.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_tmp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_tmp.release()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W * 2, H))

    lbl1 = f"Plant 1  x{speedup1:.2f} speed"
    lbl2 = f"Plant 2  x{speedup2:.2f} speed"

    cap1 = cv2.VideoCapture(str(args.p1_video))
    cap2 = cv2.VideoCapture(str(args.p2_video))
    cur1 = -1

    print("\nRendering...")
    for ti in range(n_out):
        src1 = min(int(round(ti * speedup1)), n1 - 1)

        # Read plant 2 sequentially
        ret2, f2 = cap2.read()
        if not ret2:
            break

        # Read plant 1, seeking only when src1 is non-consecutive
        if src1 != cur1 + 1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, src1)
        ret1, f1 = cap1.read()
        cur1 = src1
        if not ret1:
            break

        # Per-plant fade (each has its own warmup duration)
        fade1 = _smoothstep(ti / warmup_frames1)
        fade2 = _smoothstep(ti / warmup_frames2)

        # Track positions for this output frame
        coords1 = lu1.get(frames1[src1], {})
        coords2 = lu2.get(frames2[ti],   {})

        draw_compliance(f1, coords1, edges1, kappa1, vmin1, vmax1, fade1)
        draw_compliance(f2, coords2, edges2, kappa2, vmin2, vmax2, fade2)

        # Draw per-plant legends directly on each frame before combining
        draw_legend(f1, "Relative Compliance", vmin1, vmax1,
                    compliance_color, tick_fmt="{:.2f}")
        draw_legend(f2, "Relative Compliance", vmin2, vmax2,
                    compliance_color, tick_fmt="{:.2f}")

        canvas = np.concatenate([f1, f2], axis=1)

        draw_title(canvas, "Relative Compliance", "kappa_e")
        draw_speed_label(canvas, lbl1, W // 2,     H)
        draw_speed_label(canvas, lbl2, W + W // 2, H)

        writer.write(canvas)

        if (ti + 1) % 100 == 0:
            print(f"  {ti + 1}/{n_out}  (src1={src1})")

    writer.release()
    cap1.release()
    cap2.release()
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
