#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from create_compliance_overlay import (
    load_compliance_rows,
    load_first_frame_coords,
    load_graph,
    metric_config,
    metric_value,
    normalize,
)
from create_compliance_pointcloud import (
    build_compliance_images,
    orient_points_for_output,
    write_binary_ply,
    write_html_viewer,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def read_rgb_image(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def select_input_frame(args, out_dir):
    if args.image_path:
        image_path = Path(args.image_path)
        return read_rgb_image(image_path), image_path

    if args.image_dir:
        image_dir = Path(args.image_dir)
        images = sorted(
            [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES],
            key=natural_key,
        )
        if not images:
            raise ValueError(f"No image files found in {image_dir}")
        if args.image_index < 0 or args.image_index >= len(images):
            raise ValueError(f"--image-index {args.image_index} is outside 0..{len(images) - 1}")
        image_path = images[args.image_index]
        return read_rgb_image(image_path), image_path

    if args.video_path:
        video_path = Path(args.video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read frame {args.frame_index} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image_path = out_dir / f"{video_path.stem}_frame_{args.frame_index:04d}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return rgb, image_path

    raise ValueError("Provide one of --image-path, --image-dir, or --video-path")


def torch_device(device):
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def estimate_depth_with_depth_anything_v2(image_path, args):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Depth Anything V2 needs torch, but torch is not installed.") from exc

    if args.depth_anything_repo:
        repo_path = Path(args.depth_anything_repo)
        sys.path.insert(0, str(repo_path))

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything V2 is not importable. Clone https://github.com/DepthAnything/Depth-Anything-V2 "
            "and pass --depth-anything-repo /path/to/Depth-Anything-V2, or install a compatible package."
        ) from exc

    if not args.depth_anything_checkpoint:
        raise RuntimeError(
            "Depth Anything V2 needs a checkpoint. Pass --depth-anything-checkpoint "
            "/path/to/depth_anything_v2_vits.pth (or vitb/vitl)."
        )

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if args.depth_anything_encoder not in model_configs:
        raise ValueError(f"Unknown Depth Anything encoder: {args.depth_anything_encoder}")

    device = torch_device(args.device)
    model = DepthAnythingV2(**model_configs[args.depth_anything_encoder])
    checkpoint = torch.load(args.depth_anything_checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    raw_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    with torch.no_grad():
        depth = model.infer_image(raw_bgr, input_size=args.input_size)
    return np.asarray(depth, dtype=np.float32), "relative_inverse"


def estimate_depth_with_transformers(image_path, args):
    try:
        from PIL import Image
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The Transformers depth-estimation path needs transformers and Pillow. "
            "Install transformers, or use --model depth-anything-v2 with a local repo/checkpoint."
        ) from exc

    device = torch_device(args.device)
    pipe_device = 0 if device == "cuda" else -1
    pipeline_kwargs = {} if args.allow_downloads else {"local_files_only": True}
    pipe = pipeline(
        task="depth-estimation",
        model=args.hf_model,
        device=pipe_device,
        **pipeline_kwargs,
    )
    output = pipe(Image.open(image_path).convert("RGB"))
    if "predicted_depth" in output:
        pred = output["predicted_depth"]
        try:
            pred = pred.detach().cpu().numpy()
        except AttributeError:
            pred = np.asarray(pred)
        return np.asarray(pred, dtype=np.float32), "relative_depth"
    return np.asarray(output["depth"], dtype=np.float32), "relative_depth"


def estimate_depth_with_depth_pro(image_path, args):
    try:
        import depth_pro
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Depth Pro is not importable. Install Apple ml-depth-pro, then rerun with --model depth-pro."
        ) from exc

    device = torch_device(args.device)
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device).eval()
    image, _metadata, f_px = depth_pro.load_rgb(str(image_path))
    image = transform(image).to(device)
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy()
    return np.asarray(depth, dtype=np.float32), "metric"


def load_precomputed_depth(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)

    unchanged = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if unchanged is None:
        raise ValueError(f"Could not read depth image: {path}")
    if unchanged.ndim == 3:
        unchanged = cv2.cvtColor(unchanged, cv2.COLOR_BGR2GRAY)
    return unchanged.astype(np.float32)


def estimate_depth(image_path, args):
    if args.depth_path:
        return load_precomputed_depth(args.depth_path), args.depth_kind

    errors = []
    models = [args.model]
    if args.model == "auto":
        models = ["depth-pro", "depth-anything-v2", "transformers-depth-anything-v2"]

    for model_name in models:
        try:
            if model_name == "depth-pro":
                return estimate_depth_with_depth_pro(image_path, args)
            if model_name == "depth-anything-v2":
                return estimate_depth_with_depth_anything_v2(image_path, args)
            if model_name == "transformers-depth-anything-v2":
                return estimate_depth_with_transformers(image_path, args)
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")
            if args.model != "auto":
                raise

    joined = "\n".join(f"  - {error}" for error in errors)
    raise RuntimeError(f"No monocular depth backend could run:\n{joined}")


def robust_normalize(values, low_percentile=2.0, high_percentile=98.0):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Depth map has no finite values")
    lo, hi = np.percentile(finite, [low_percentile, high_percentile])
    if abs(float(hi) - float(lo)) < 1e-9:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def depth_to_z(depth, depth_kind, args):
    depth = np.asarray(depth, dtype=np.float32)
    if depth_kind == "metric":
        z = depth * args.metric_depth_scale
    else:
        relative = robust_normalize(depth, args.depth_low_percentile, args.depth_high_percentile)
        if depth_kind in {"relative_inverse", "inverse"}:
            z = args.relative_far - relative * (args.relative_far - args.relative_near)
        elif depth_kind in {"relative_depth", "depth"}:
            z = args.relative_near + relative * (args.relative_far - args.relative_near)
        else:
            raise ValueError(f"Unknown depth kind: {depth_kind}")

    z = np.asarray(z, dtype=np.float32)
    z[~np.isfinite(z)] = 0.0
    return z


def save_depth_debug(depth_raw, z_m, out_dir, stem):
    np.save(out_dir / f"{stem}_raw_depth.npy", depth_raw.astype(np.float32))
    np.save(out_dir / f"{stem}_pointcloud_z.npy", z_m.astype(np.float32))

    for name, values in [("raw_depth", depth_raw), ("pointcloud_z", z_m)]:
        vis = robust_normalize(values)
        vis_u8 = np.round(vis * 255.0).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_u8, cv2.COLORMAP_MAGMA)
        cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), vis_color)


def virtual_camera_intrinsics(width, height, fov_deg, focal_px):
    if focal_px is None:
        focal_px = 0.5 * width / np.tan(np.deg2rad(fov_deg) * 0.5)
    return float(focal_px), float(focal_px), 0.5 * (width - 1), 0.5 * (height - 1)


def make_pointcloud_from_monocular_depth(
    rgb,
    z_m,
    compliance_rgb,
    compliance_mask,
    args,
):
    height, width = z_m.shape[:2]
    ys = np.arange(0, height, max(args.stride, 1), dtype=np.int32)
    xs = np.arange(0, width, max(args.stride, 1), dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    z = z_m[vv, uu]
    valid = np.isfinite(z) & (z > args.min_depth) & (z < args.max_depth)

    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    fx, fy, cx, cy = virtual_camera_intrinsics(width, height, args.fov_deg, args.focal_px)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    points = np.column_stack([x, y, z]).astype(np.float32)
    points = orient_points_for_output(points, args.coordinate_frame)

    ui = np.round(uu).astype(np.int32)
    vi = np.round(vv).astype(np.int32)
    if args.background == "rgb":
        colors = rgb[vi, ui].copy()
    elif args.background == "black":
        colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 145, dtype=np.uint8)

    compliance_hits = compliance_mask[vi, ui] > 0
    colors[compliance_hits] = compliance_rgb[vi[compliance_hits], ui[compliance_hits]]

    if args.background == "hide":
        points = points[compliance_hits]
        colors = colors[compliance_hits]
        compliance_hits = compliance_hits[compliance_hits]

    if args.max_points is not None and points.shape[0] > args.max_points:
        rng = np.random.default_rng(0)
        branch_idx = np.flatnonzero(compliance_hits)
        background_idx = np.flatnonzero(~compliance_hits)
        if branch_idx.size >= args.max_points:
            keep_idx = np.sort(rng.choice(branch_idx, size=args.max_points, replace=False))
        else:
            remaining = args.max_points - branch_idx.size
            sampled_bg = rng.choice(background_idx, size=remaining, replace=False) if remaining > 0 else np.array([], dtype=int)
            keep_idx = np.sort(np.concatenate([branch_idx, sampled_bg]))
        points = points[keep_idx]
        colors = colors[keep_idx]
        compliance_hits = compliance_hits[keep_idx]

    stats = {
        "written_points": int(points.shape[0]),
        "compliance_points": int(np.count_nonzero(compliance_hits)),
        "focal_px": f"{fx:.2f}",
        "coordinate_frame": args.coordinate_frame,
    }
    return points, colors, stats


def parse_size(value):
    if value is None:
        return None
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT")
    return int(parts[0]), int(parts[1])


def scaled_coords(coords, source_size, target_size):
    if source_size is None:
        return coords
    src_w, src_h = source_size
    dst_w, dst_h = target_size
    sx = dst_w / max(float(src_w), 1.0)
    sy = dst_h / max(float(src_h), 1.0)
    return {pid: np.asarray([xy[0] * sx, xy[1] * sy], dtype=float) for pid, xy in coords.items()}


def safe_stem(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Estimate monocular depth for one RGB frame and export a compliance-colored point cloud."
    )
    parser.add_argument("analysis_dir", help="Directory with compliance_summary.csv, analysis_tracks.csv, graph_spec_used.json")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-path", default=None)
    source.add_argument("--image-dir", default=None)
    source.add_argument("--video-path", default=None)
    parser.add_argument("--out-dir", default=None, help="Output directory; defaults to analysis_dir/monocular_pointcloud")
    parser.add_argument("--image-index", type=int, default=0, help="Image index when using --image-dir")
    parser.add_argument("--frame-index", type=int, default=0, help="Video frame index when using --video-path")
    parser.add_argument("--model", choices=["auto", "depth-pro", "depth-anything-v2", "transformers-depth-anything-v2"], default="auto")
    parser.add_argument("--depth-path", default=None, help="Use a precomputed .npy/.png depth map instead of running a model")
    parser.add_argument(
        "--depth-kind",
        choices=["metric", "relative_inverse", "relative_depth", "inverse", "depth"],
        default="relative_inverse",
        help="Depth convention for --depth-path.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--depth-anything-repo", default=None)
    parser.add_argument("--depth-anything-checkpoint", default=None)
    parser.add_argument("--depth-anything-encoder", choices=["vits", "vitb", "vitl", "vitg"], default="vits")
    parser.add_argument("--hf-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow Transformers to download the HF model if not cached")
    parser.add_argument("--input-size", type=int, default=518, help="Depth Anything V2 input size")
    parser.add_argument("--mode", default="compliance", choices=[
        "compliance",
        "uncertainty",
        "compliance_per_length",
        "compliance_per_uncertainty",
        "compliance_per_length_variation",
    ])
    parser.add_argument("--edge-thickness", type=int, default=16)
    parser.add_argument("--track-image-size", type=parse_size, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--background", choices=["rgb", "gray", "black", "hide"], default="rgb")
    parser.add_argument("--coordinate-frame", choices=["right-handed", "camera", "legacy"], default="right-handed")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="Virtual camera FOV used when focal length is unknown")
    parser.add_argument("--focal-px", type=float, default=None, help="Override virtual focal length in pixels")
    parser.add_argument("--relative-near", type=float, default=0.35)
    parser.add_argument("--relative-far", type=float, default=2.5)
    parser.add_argument("--metric-depth-scale", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--depth-low-percentile", type=float, default=2.0)
    parser.add_argument("--depth-high-percentile", type=float, default=98.0)
    return parser


def main():
    args = make_parser().parse_args()
    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir / "monocular_pointcloud"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb, image_path = select_input_frame(args, out_dir)
    image_stem = safe_stem(image_path.stem)
    extracted_path = out_dir / f"{image_stem}_rgb.png"
    cv2.imwrite(str(extracted_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    depth_raw, inferred_depth_kind = estimate_depth(image_path, args)
    if depth_raw.shape[:2] != rgb.shape[:2]:
        depth_raw = cv2.resize(depth_raw, rgb.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    z_m = depth_to_z(depth_raw, inferred_depth_kind, args)
    save_depth_debug(depth_raw, z_m, out_dir, image_stem)

    _root_id, edges = load_graph(analysis_dir / "graph_spec_used.json")
    coords = load_first_frame_coords(analysis_dir / "analysis_tracks.csv")
    coords = scaled_coords(coords, args.track_image_size, rgb.shape[1::-1])
    rows = load_compliance_rows(analysis_dir / "compliance_summary.csv")
    rows_by_edge = {(row["parent"], row["child"]): row for row in rows}

    compliance_rgb, compliance_mask, _edge_id, vmin, vmax = build_compliance_images(
        rgb.shape,
        coords,
        edges,
        rows_by_edge,
        mode=args.mode,
        thickness=args.edge_thickness,
    )

    points, colors, stats = make_pointcloud_from_monocular_depth(
        rgb=rgb,
        z_m=z_m,
        compliance_rgb=compliance_rgb,
        compliance_mask=compliance_mask,
        args=args,
    )

    model_name = "precomputed" if args.depth_path else args.model
    stem = f"relative_{args.mode}_monocular_{safe_stem(model_name)}_{image_stem}"
    if args.background == "hide":
        stem += "_branch_only"
    ply_path = out_dir / f"{stem}.ply"
    html_path = out_dir / f"{stem}.html"
    write_binary_ply(ply_path, points, colors)

    cfg = metric_config(args.mode)
    values = [metric_value(row, args.mode) for row in rows_by_edge.values()]
    _metric_min, _metric_max = normalize(values)
    metadata = {
        "points": stats["written_points"],
        "heatmap points": stats["compliance_points"],
        "image": f"{rgb.shape[1]}x{rgb.shape[0]}",
        "depth kind": inferred_depth_kind,
        "z range": f"{float(np.nanmin(z_m)):.3g} to {float(np.nanmax(z_m)):.3g}",
        "metric range": f"{vmin:.3g} to {vmax:.3g}",
        "focal px": stats["focal_px"],
        "coordinate frame": stats["coordinate_frame"],
    }
    write_html_viewer(html_path, points, colors, f"{cfg['title']} + Monocular Depth", metadata)

    print(f"RGB frame: {extracted_path}")
    print(f"Raw depth: {out_dir / f'{image_stem}_raw_depth.npy'}")
    print(f"Depth visualization: {out_dir / f'{image_stem}_raw_depth.png'}")
    print(f"Point-cloud Z: {out_dir / f'{image_stem}_pointcloud_z.npy'}")
    print(f"Point-cloud Z visualization: {out_dir / f'{image_stem}_pointcloud_z.png'}")
    print(f"Depth kind: {inferred_depth_kind}")
    print(f"Compliance-colored points written: {stats['compliance_points']}")
    print(f"Total points written: {stats['written_points']}")
    print(f"PLY: {ply_path}")
    print(f"Viewer: {html_path}")


if __name__ == "__main__":
    main()
