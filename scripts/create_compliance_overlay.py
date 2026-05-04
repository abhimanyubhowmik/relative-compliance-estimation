import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_graph(graph_path):
    spec = json.loads(Path(graph_path).read_text())
    return int(spec["root"]), [tuple(map(int, edge)) for edge in spec["edges"]]


def load_first_frame_coords(tracks_csv):
    coords = {}
    with open(tracks_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        frame0 = None
        for row in reader:
            frame = int(float(row["frame"]))
            if frame0 is None:
                frame0 = frame
            if frame != frame0:
                continue
            point_id = int(float(row["point_id"]))
            coords[point_id] = np.array([float(row["x"]), float(row["y"])], dtype=float)
    return coords


def load_compliance_rows(compliance_csv):
    rows = []
    with open(compliance_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key in {"distance_zone"}:
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value) if value not in {"", "nan", "NaN"} else np.nan
                except ValueError:
                    parsed[key] = value
            parsed["parent"] = int(parsed["parent"])
            parsed["child"] = int(parsed["child"])
            rows.append(parsed)
    return rows


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read first frame from: {video_path}")
    return frame


def color_interp(stops, t):
    t = float(np.clip(t, 0.0, 1.0))
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            local = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            return tuple(int(round((1 - local) * c0[j] + local * c1[j])) for j in range(3))
    return stops[-1][1]


def compliance_color(t):
    return color_interp(
        [
            (0.0, (180, 60, 30)),
            (0.5, (80, 220, 240)),
            (1.0, (30, 30, 210)),
        ],
        t,
    )


def uncertainty_color(t):
    return color_interp(
        [
            (0.0, (50, 180, 60)),
            (0.5, (30, 220, 220)),
            (1.0, (40, 40, 220)),
        ],
        t,
    )


def derived_color(t):
    return color_interp(
        [
            (0.0, (120, 40, 180)),
            (0.5, (70, 200, 240)),
            (1.0, (30, 210, 120)),
        ],
        t,
    )


def normalize(values):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    return lo, hi


def split_title(title):
    words = title.split()
    if len(words) <= 1:
        return [title]
    if len(words) == 2:
        return words
    midpoint = (len(words) + 1) // 2
    return [" ".join(words[:midpoint]), " ".join(words[midpoint:])]


def draw_legend(canvas, title, vmin, vmax, color_fn, tick_fmt="{:.2f}"):
    h, w = canvas.shape[:2]
    box_w = 18
    box_h = min(145, h - 108)
    right_margin = 8
    x0 = w - 88
    y0 = 58
    label_x = x0 + box_w + 10

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 10, 18), (w - right_margin, y0 + box_h + 28), (18, 18, 18), -1)
    canvas[:] = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

    for i in range(box_h):
        t = 1.0 - (i / max(box_h - 1, 1))
        color = color_fn(t)
        cv2.line(canvas, (x0, y0 + i), (x0 + box_w, y0 + i), color, 1)

    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1)
    for line_idx, line in enumerate(split_title(title)):
        cv2.putText(
            canvas,
            line,
            (x0 - 8, 32 + line_idx * 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    ticks = [(0.0, vmax), (0.5, 0.5 * (vmin + vmax)), (1.0, vmin)]
    for t, value in ticks:
        yy = int(round(y0 + t * box_h))
        cv2.line(canvas, (x0 + box_w + 3, yy), (x0 + box_w + 8, yy), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            tick_fmt.format(value),
            (label_x, yy + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def metric_config(mode):
    if mode == "compliance":
        return {
            "title": "Relative Compliance",
            "subtitle": "kappa_e",
            "tick_fmt": "{:.2f}",
            "color_fn": compliance_color,
            "filename": "relative_compliance_overlay_first_frame.png",
        }
    if mode == "uncertainty":
        return {
            "title": "Relative Uncertainty",
            "subtitle": "(CI width / kappa_e)",
            "tick_fmt": "{:.2f}x",
            "color_fn": uncertainty_color,
            "filename": "relative_compliance_uncertainty_overlay_first_frame.png",
        }
    if mode == "compliance_per_length":
        return {
            "title": "Compliance / Edge Length",
            "subtitle": "kappa_e / edge_length",
            "tick_fmt": "{:.3f}",
            "color_fn": derived_color,
            "filename": "relative_compliance_per_edge_length_overlay_first_frame.png",
        }
    if mode == "compliance_per_uncertainty":
        return {
            "title": "Compliance / Uncertainty",
            "subtitle": "kappa_e / rel_uncertainty",
            "tick_fmt": "{:.2f}",
            "color_fn": derived_color,
            "filename": "relative_compliance_per_uncertainty_overlay_first_frame.png",
        }
    if mode == "compliance_per_length_variation":
        return {
            "title": "Compliance / Length Variation",
            "subtitle": "kappa_e / length_CV",
            "tick_fmt": "{:.2f}",
            "color_fn": derived_color,
            "filename": "relative_compliance_per_edge_length_variation_overlay_first_frame.png",
        }
    raise ValueError(f"Unknown overlay mode: {mode}")


def metric_value(row, mode):
    if row is None:
        return np.nan
    kappa = row.get("relative_compliance_kappa", np.nan)
    if mode == "compliance":
        return kappa
    ci_low = row.get("relative_compliance_ci_low", np.nan)
    ci_high = row.get("relative_compliance_ci_high", np.nan)
    if not (np.isfinite(ci_low) and np.isfinite(ci_high) and np.isfinite(kappa) and kappa > 1e-9):
        rel_uncertainty = np.nan
    else:
        rel_uncertainty = (ci_high - ci_low) / kappa
    if mode == "uncertainty":
        return rel_uncertainty
    if mode == "compliance_per_length":
        edge_length = row.get("rest_length_px", np.nan)
        if not (np.isfinite(kappa) and np.isfinite(edge_length) and edge_length > 1e-9):
            return np.nan
        return kappa / edge_length
    if mode == "compliance_per_uncertainty":
        if not (np.isfinite(kappa) and np.isfinite(rel_uncertainty) and rel_uncertainty > 1e-9):
            return np.nan
        return kappa / rel_uncertainty
    if mode == "compliance_per_length_variation":
        length_cv = row.get("length_cv", np.nan)
        if not (np.isfinite(kappa) and np.isfinite(length_cv) and length_cv > 1e-9):
            return np.nan
        return kappa / length_cv
    return np.nan


def render_overlay(frame, coords, edges, rows_by_edge, root_id, mode):
    canvas = frame.copy()
    cfg = metric_config(mode)
    values = [metric_value(row, mode) for row in rows_by_edge.values()]
    vmin, vmax = normalize(values)
    color_fn = cfg["color_fn"]
    title = cfg["title"]
    subtitle = cfg["subtitle"]

    overlay = canvas.copy()
    for parent, child in edges:
        if parent not in coords or child not in coords:
            continue
        row = rows_by_edge.get((parent, child))
        value = metric_value(row, mode) if row is not None else np.nan
        if np.isfinite(value):
            t = (value - vmin) / max(vmax - vmin, 1e-9)
            color = color_fn(t)
        else:
            color = (180, 180, 180)
        p0 = tuple(np.round(coords[parent]).astype(int))
        p1 = tuple(np.round(coords[child]).astype(int))
        cv2.line(overlay, p0, p1, color, 16, cv2.LINE_AA)
        cv2.line(overlay, p0, p1, color, 10, cv2.LINE_AA)

    canvas = cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0.0)

    title_scale = 0.50
    subtitle_scale = 0.34
    title_thickness = 1
    subtitle_thickness = 1
    title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, title_thickness)[0]
    subtitle_size = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, subtitle_thickness)[0]
    text_w = min(max(title_size[0], subtitle_size[0]) + 24, canvas.shape[1] - 130)
    text_w = max(text_w, 150)

    text_overlay = canvas.copy()
    cv2.rectangle(text_overlay, (18, 18), (18 + text_w, 64), (15, 15, 15), -1)
    canvas[:] = cv2.addWeighted(text_overlay, 0.45, canvas, 0.55, 0.0)
    cv2.putText(canvas, title, (28, 39), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (29, 57), cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, (220, 220, 220), subtitle_thickness, cv2.LINE_AA)

    draw_legend(canvas, title, vmin, vmax, color_fn, tick_fmt=cfg["tick_fmt"])
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Create first-frame compliance overlays from branch analysis outputs.")
    parser.add_argument("analysis_dir", help="Directory containing compliance_summary.csv, analysis_tracks.csv, and graph_spec_used.json")
    parser.add_argument("--video-path", required=True, help="Video or annotated MP4 used as the background source")
    parser.add_argument("--frame-tracks", default=None, help="Optional CSV to use for node coordinates; defaults to analysis_tracks.csv in the analysis dir")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "compliance",
            "uncertainty",
            "compliance_per_length",
            "compliance_per_uncertainty",
            "compliance_per_length_variation",
        ],
        choices=[
            "compliance",
            "uncertainty",
            "compliance_per_length",
            "compliance_per_uncertainty",
            "compliance_per_length_variation",
        ],
        help="Overlay modes to render.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    compliance_csv = analysis_dir / "compliance_summary.csv"
    graph_json = analysis_dir / "graph_spec_used.json"
    tracks_csv = Path(args.frame_tracks) if args.frame_tracks else analysis_dir / "analysis_tracks.csv"

    root_id, edges = load_graph(graph_json)
    coords = load_first_frame_coords(tracks_csv)
    rows = load_compliance_rows(compliance_csv)
    rows_by_edge = {(row["parent"], row["child"]): row for row in rows}
    frame = extract_first_frame(Path(args.video_path))

    for mode in args.modes:
        img = render_overlay(frame, coords, edges, rows_by_edge, root_id, mode=mode)
        cv2.imwrite(str(analysis_dir / metric_config(mode)["filename"]), img)


if __name__ == "__main__":
    main()
