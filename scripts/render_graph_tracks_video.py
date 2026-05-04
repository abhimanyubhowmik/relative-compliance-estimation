import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_graph(graph_json):
    spec = json.loads(Path(graph_json).read_text())
    root = int(spec["root"])
    edges = [tuple(map(int, edge)) for edge in spec["edges"]]
    nodes = sorted({root, *[u for u, _ in edges], *[v for _, v in edges]})
    return root, nodes, edges


def load_track_lookup(csv_path, allowed_nodes=None):
    allowed = None if allowed_nodes is None else set(int(n) for n in allowed_nodes)
    lookup = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            point_id = int(float(row["point_id"]))
            if allowed is not None and point_id not in allowed:
                continue
            frame = int(float(row["frame"]))
            x = float(row["x"])
            y = float(row["y"])
            lookup.setdefault(point_id, {})[frame] = np.array([x, y], dtype=float)
    return lookup


def build_color_map(node_ids):
    palette = [
        (255, 0, 0),
        (0, 200, 0),
        (0, 0, 255),
        (255, 180, 0),
        (255, 0, 255),
        (0, 220, 220),
        (180, 90, 0),
        (120, 0, 180),
        (0, 120, 180),
        (180, 180, 0),
    ]
    return {node_id: palette[i % len(palette)] for i, node_id in enumerate(sorted(node_ids))}


def render(video_path, tracks_csv, graph_json, out_path, title=None):
    root, nodes, edges = load_graph(graph_json)
    track_lookup = load_track_lookup(tracks_csv, allowed_nodes=nodes)
    colors = build_color_map(nodes)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_pts = {
            node_id: frame_map[frame_idx]
            for node_id, frame_map in track_lookup.items()
            if frame_idx in frame_map
        }

        for parent, child in edges:
            if parent in current_pts and child in current_pts:
                p0 = tuple(np.round(current_pts[parent]).astype(int))
                p1 = tuple(np.round(current_pts[child]).astype(int))
                cv2.line(frame, p0, p1, (220, 220, 220), 1, lineType=cv2.LINE_AA)

        for node_id, pt in current_pts.items():
            p = tuple(np.round(pt).astype(int))
            color = colors[node_id]
            radius = 5 if node_id == root else 4
            cv2.circle(frame, p, radius, color, thickness=-1, lineType=cv2.LINE_AA)
            if node_id == root:
                cv2.circle(frame, p, 9, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(
                frame,
                str(node_id),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                lineType=cv2.LINE_AA,
            )

        header = title or "graph-smoothed tracks"
        cv2.putText(
            frame,
            header,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (12, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render graph-constrained tracked nodes over a source video.")
    parser.add_argument("video_path")
    parser.add_argument("tracks_csv")
    parser.add_argument("graph_json")
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--title", default="factor-graph smoothed tracks")
    args = parser.parse_args()

    out_path = Path(args.out_path) if args.out_path else Path(args.tracks_csv).with_name("annotated_nodes_factor_graph_smoothed.mp4")
    render(args.video_path, args.tracks_csv, args.graph_json, out_path, title=args.title)
    print(f"Wrote smoothed graph video to: {out_path}")


if __name__ == "__main__":
    main()
