# Script: read CSV of tracks, re-open video, draw color-coded small tracked points up to each frame,
# compute per-point PCA on that point's (x,y) history up to the current frame, draw PCA arrows,
# write output video.
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def pca_2d(X):
    # X: (N,2)
    Xc = X - X.mean(axis=0)
    if Xc.shape[0] < 2:
        return None
    cov = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    lengths = np.sqrt(np.maximum(vals, 0.0))
    return vecs, lengths

def run(video_path, csv_path, out_path=None, fps=None, viz_scale=3.0):
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    if out_path is None:
        out_path = csv_path.parent / 'tapir_pca_video.mp4'
    out_path = Path(out_path)

    df = pd.read_csv(csv_path)
    grouped = {pid: g.sort_values('frame').to_numpy() for pid, g in df.groupby('point_id')}
    # grouped[pid] is array of rows [point_id, frame, x, y]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit("Failed to open video")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = fps if fps is not None else cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (width, height))

    # color palette (BGR)
    palette = [
        (255, 0, 0),(0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255),
        (128, 0, 0),(0, 128, 0),(0, 0, 128),(128, 128, 0),(128, 0, 128),(0, 128, 128)
    ]
    point_ids = sorted(grouped.keys())
    point_colors = {pid: palette[i % len(palette)] for i, pid in enumerate(point_ids)}

    # Precompute per-point frames list and xy arrays for fast slicing
    per_point_frames = {}
    per_point_xy = {}
    for pid, arr in grouped.items():
        frames = arr[:,1].astype(int)
        xy = arr[:,2:4].astype(float)
        per_point_frames[pid] = frames
        per_point_xy[pid] = xy

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        canvas = frame_bgr.copy()

        # For each point: get all samples up to current frame
        for pid in point_ids:
            frames = per_point_frames[pid]
            xy = per_point_xy[pid]
            # find indices where frame <= frame_idx
            valid_until = np.searchsorted(frames, frame_idx, side='right')
            if valid_until == 0:
                continue
            hist = xy[:valid_until]  # shape (k,2)
            current_pos = hist[-1].astype(int)
            color = point_colors[pid]
            # draw small current point
            cv2.circle(canvas, tuple(current_pos.tolist()), 2, color, thickness=-1)

            # PCA on history (at least 2 points)
            if hist.shape[0] >= 2:
                pca = pca_2d(hist)
                if pca is not None:
                    vecs, lengths = pca
                    center = hist.mean(axis=0)
                    # scale vectors for visibility
                    v0 = vecs[:,0] * lengths[0] * viz_scale
                    v1 = vecs[:,1] * lengths[1] * viz_scale
                    p0 = center
                    p0a = (p0 + v0).astype(int)
                    p0b = (p0 - v0).astype(int)
                    p1a = (p0 + v1).astype(int)
                    p1b = (p0 - v1).astype(int)
                    c = tuple(int(x) for x in color)
                    # draw primary and secondary axes (primary thicker)
                    cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(p0a), c, 2, tipLength=0.2)
                    cv2.arrowedLine(canvas, tuple(center.astype(int)), tuple(p1a), c, 1, tipLength=0.2)

        out.write(canvas)
        frame_idx += 1

    cap.release()
    out.release()
    print(f'Wrote PCA visualization video to: {out_path}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: python plot_tracks_pca_video.py /path/to/video.mp4 /path/to/tapir_tracks.csv [out_video.mp4] [fps] [viz_scale]')
        sys.exit(1)
    video = sys.argv[1]
    csvf = sys.argv[2]
    outv = sys.argv[3] if len(sys.argv) > 3 else None
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else None
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    run(video, csvf, out_path=outv, fps=fps, viz_scale=scale)