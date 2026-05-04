import argparse
import json
import math
from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


NODE_METRIC_COLUMNS = [
    "node_id",
    "depth",
    "root_distance_px",
    "distance_zone",
    "n_frames",
    "motion_rms_px",
    "motion_mean_px",
    "rel_motion_rms_px",
    "rel_motion_mean_px",
    "motion_max_px",
    "rel_motion_max_px",
    "rel_motion_rms_per_root_distance",
    "rel_motion_rms_ci_low",
    "rel_motion_rms_ci_high",
    "track_support_ratio",
]

NODE_SERIES_COLUMNS = [
    "node_id",
    "depth",
    "root_distance_px",
    "distance_zone",
    "frame",
    "motion_px",
    "rel_motion_px",
]

EDGE_METRIC_COLUMNS = [
    "parent",
    "child",
    "depth",
    "root_distance_px",
    "distance_zone",
    "n_frames",
    "rest_length_px",
    "length_mean_px",
    "length_std_px",
    "length_cv",
    "length_cv_ci_low",
    "length_cv_ci_high",
    "length_range_px",
    "motion_gain_proxy",
    "child_rel_motion_rms_px",
    "parent_rel_motion_rms_px",
]

EDGE_SERIES_COLUMNS = ["parent", "child", "frame", "length_px"]

BEND_METRIC_COLUMNS = [
    "grandparent",
    "joint",
    "child",
    "depth",
    "root_distance_px",
    "distance_zone",
    "n_frames",
    "rest_angle_deg",
    "angle_mean_deg",
    "angle_std_deg",
    "angle_range_deg",
    "angle_rms_delta_deg",
    "angle_mean_abs_delta_deg",
    "n_inlier_frames",
    "filtered_angle_std_deg",
    "filtered_angle_rms_delta_deg",
    "filtered_angle_mean_abs_delta_deg",
    "filtered_angle_rms_ci_low",
    "filtered_angle_rms_ci_high",
]
BEND_SERIES_COLUMNS = [
    "grandparent",
    "joint",
    "child",
    "frame",
    "angle_deg",
    "delta_from_rest_deg",
    "is_inlier",
]

RELATIVE_COMPLIANCE_PRIOR_STRENGTH = 12.0
RELATIVE_COMPLIANCE_BOOTSTRAP_SAMPLES = 250


def load_graph(graph_path):
    graph_path = Path(graph_path)
    spec = json.loads(graph_path.read_text())

    if "root" not in spec or "edges" not in spec:
        raise ValueError("Graph JSON must contain 'root' and 'edges'.")

    root = int(spec["root"])
    edges = [tuple(map(int, edge)) for edge in spec["edges"]]

    children = {}
    parent = {}
    nodes = {root}

    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
        children.setdefault(u, []).append(v)
        if v in parent:
            raise ValueError(f"Node {v} has multiple parents; expected a rooted tree.")
        parent[v] = u

    if root in parent:
        raise ValueError("Root node cannot appear as a child in edges.")

    missing = nodes - ({root} | set(parent.keys()))
    if missing:
        raise ValueError(
            "Some nodes are disconnected from the rooted tree: "
            + ", ".join(map(str, sorted(missing)))
        )

    depth = {root: 0}
    q = deque([root])
    while q:
        node = q.popleft()
        for child in children.get(node, []):
            depth[child] = depth[node] + 1
            q.append(child)

    if len(depth) != len(nodes):
        raise ValueError("Graph traversal from root did not reach every node.")

    bends = []
    for joint, joint_children in children.items():
        if joint == root:
            continue
        parent_node = parent[joint]
        for child in joint_children:
            bends.append((parent_node, joint, child))

    return {
        "root": root,
        "nodes": sorted(nodes),
        "edges": edges,
        "children": children,
        "parent": parent,
        "depth": depth,
        "bends": bends,
    }


def load_tracks(csv_path, nodes):
    df = pd.read_csv(csv_path)
    required = {"point_id", "frame", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    df = df[df["point_id"].isin(nodes)].copy()
    if df.empty:
        raise ValueError("No rows matched the point ids defined in the graph JSON.")

    frames = sorted(df["frame"].unique())
    coord_lookup = {}
    for pid, group in df.groupby("point_id"):
        coord_lookup[int(pid)] = {
            int(frame): np.array([x, y], dtype=float)
            for frame, x, y in group.sort_values("frame")[["frame", "x", "y"]].itertuples(index=False)
        }

    missing_nodes = sorted(set(nodes) - set(coord_lookup.keys()))
    if missing_nodes:
        raise ValueError(
            "The graph JSON references point ids missing from the CSV: "
            + ", ".join(map(str, missing_nodes))
        )

    return df, frames, coord_lookup


def count_all_csv_nodes(csv_path):
    df = pd.read_csv(csv_path, usecols=["point_id"])
    return int(df["point_id"].nunique())


def common_frames_for_nodes(coord_lookup, node_ids):
    frame_sets = [set(coord_lookup[node].keys()) for node in node_ids]
    if not frame_sets:
        return []
    return sorted(set.intersection(*frame_sets))


def point_at(coord_lookup, node_id, frame):
    return coord_lookup[node_id][frame]


def moving_average_1d(values, window):
    if window <= 1 or len(values) == 0:
        return values.copy()
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def smooth_coord_lookup(coord_lookup, window=5):
    smoothed = {}
    for node_id, frame_map in coord_lookup.items():
        frames = np.array(sorted(frame_map.keys()), dtype=int)
        xy = np.array([frame_map[int(frame)] for frame in frames], dtype=float)
        xs = moving_average_1d(xy[:, 0], window)
        ys = moving_average_1d(xy[:, 1], window)
        smoothed[node_id] = {
            int(frame): np.array([x, y], dtype=float)
            for frame, x, y in zip(frames, xs, ys)
        }
    return smoothed


def coord_lookup_to_dataframe(coord_lookup):
    rows = []
    for node_id, frame_map in coord_lookup.items():
        for frame, xy in frame_map.items():
            rows.append(
                {
                    "point_id": int(node_id),
                    "frame": int(frame),
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )
    return pd.DataFrame(rows).sort_values(["point_id", "frame"]).reset_index(drop=True)


def factor_graph_style_smoother(
    graph,
    coord_lookup,
    measurement_weight=1.0,
    temporal_weight=0.35,
    acceleration_weight=0.2,
    edge_weight=0.45,
):
    node_ids = sorted(graph["nodes"])
    max_frame = max(max(frame_map.keys()) for frame_map in coord_lookup.values())
    n_frames = max_frame + 1
    n_nodes = len(node_ids)
    node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

    def var_index(node_id, frame):
        return node_to_idx[node_id] * n_frames + int(frame)

    n_vars = n_nodes * n_frames
    rows = []
    cols = []
    data = []
    bx = []
    by = []
    row_counter = 0

    def add_factor(entries, target_x, target_y):
        nonlocal row_counter
        for col, value in entries:
            rows.append(row_counter)
            cols.append(col)
            data.append(float(value))
        bx.append(float(target_x))
        by.append(float(target_y))
        row_counter += 1

    # Measurement factors
    meas_w = math.sqrt(max(measurement_weight, 1e-9))
    for node_id in node_ids:
        for frame, xy in coord_lookup[node_id].items():
            add_factor([(var_index(node_id, frame), meas_w)], meas_w * xy[0], meas_w * xy[1])

    # Temporal smoothness factors
    temp_w = math.sqrt(max(temporal_weight, 0.0))
    if temp_w > 0:
        for node_id in node_ids:
            for frame in range(n_frames - 1):
                add_factor(
                    [
                        (var_index(node_id, frame + 1), temp_w),
                        (var_index(node_id, frame), -temp_w),
                    ],
                    0.0,
                    0.0,
                )

    # Second-difference acceleration factors
    accel_w = math.sqrt(max(acceleration_weight, 0.0))
    if accel_w > 0:
        for node_id in node_ids:
            for frame in range(1, n_frames - 1):
                add_factor(
                    [
                        (var_index(node_id, frame - 1), accel_w),
                        (var_index(node_id, frame), -2.0 * accel_w),
                        (var_index(node_id, frame + 1), accel_w),
                    ],
                    0.0,
                    0.0,
                )

    # Edge-shape factors based on first common-frame offset; this is a linearized shape prior
    edge_w = math.sqrt(max(edge_weight, 0.0))
    if edge_w > 0:
        for parent, child in graph["edges"]:
            common = common_frames_for_nodes(coord_lookup, [parent, child])
            if not common:
                continue
            frame0 = common[0]
            rest_vec = point_at(coord_lookup, child, frame0) - point_at(coord_lookup, parent, frame0)
            for frame in range(n_frames):
                add_factor(
                    [
                        (var_index(child, frame), edge_w),
                        (var_index(parent, frame), -edge_w),
                    ],
                    edge_w * rest_vec[0],
                    edge_w * rest_vec[1],
                )

    A = sparse.csr_matrix((data, (rows, cols)), shape=(row_counter, n_vars))
    x_sol = lsqr(A, np.asarray(bx, dtype=float), atol=1e-6, btol=1e-6, iter_lim=2000)[0]
    y_sol = lsqr(A, np.asarray(by, dtype=float), atol=1e-6, btol=1e-6, iter_lim=2000)[0]

    smoothed = {}
    for node_id in node_ids:
        smoothed[node_id] = {}
        observed_frames = sorted(coord_lookup[node_id].keys())
        for frame in observed_frames:
            idx = var_index(node_id, frame)
            smoothed[node_id][frame] = np.array([x_sol[idx], y_sol[idx]], dtype=float)
    return smoothed


def summarize_metric_table(df, id_cols, value_cols, suffix):
    out = df[list(id_cols) + list(value_cols)].copy()
    rename_map = {col: f"{col}_{suffix}" for col in value_cols}
    return out.rename(columns=rename_map)


def build_smoothing_comparison(node_raw, node_smooth, edge_raw, edge_smooth, bend_raw, bend_smooth):
    node_cmp = summarize_metric_table(
        node_raw,
        ["node_id", "depth", "root_distance_px", "distance_zone"],
        ["rel_motion_rms_px", "rel_motion_rms_per_root_distance"],
        "raw",
    ).merge(
        summarize_metric_table(
            node_smooth,
            ["node_id", "depth", "root_distance_px", "distance_zone"],
            ["rel_motion_rms_px", "rel_motion_rms_per_root_distance"],
            "smoothed",
        ),
        on=["node_id", "depth", "root_distance_px", "distance_zone"],
        how="outer",
    )
    edge_cmp = summarize_metric_table(
        edge_raw,
        ["parent", "child", "depth", "root_distance_px", "distance_zone"],
        ["length_cv", "length_std_px"],
        "raw",
    ).merge(
        summarize_metric_table(
            edge_smooth,
            ["parent", "child", "depth", "root_distance_px", "distance_zone"],
            ["length_cv", "length_std_px"],
            "smoothed",
        ),
        on=["parent", "child", "depth", "root_distance_px", "distance_zone"],
        how="outer",
    )
    bend_cmp = summarize_metric_table(
        bend_raw,
        ["grandparent", "joint", "child", "depth", "root_distance_px", "distance_zone"],
        ["filtered_angle_rms_delta_deg", "filtered_angle_std_deg"],
        "raw",
    ).merge(
        summarize_metric_table(
            bend_smooth,
            ["grandparent", "joint", "child", "depth", "root_distance_px", "distance_zone"],
            ["filtered_angle_rms_delta_deg", "filtered_angle_std_deg"],
            "smoothed",
        ),
        on=["grandparent", "joint", "child", "depth", "root_distance_px", "distance_zone"],
        how="outer",
    )
    return node_cmp, edge_cmp, bend_cmp


def plot_smoothing_comparison(node_cmp, edge_cmp, bend_cmp, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=False)
    if not node_cmp.empty:
        axes[0].scatter(node_cmp["root_distance_px"], node_cmp["rel_motion_rms_px_raw"], color="0.65", label="raw", s=35)
        axes[0].scatter(node_cmp["root_distance_px"], node_cmp["rel_motion_rms_px_smoothed"], color="tab:green", label="smoothed", s=35)
        axes[0].set_ylabel("Node motion RMS")
        axes[0].set_title("Raw vs smoothed metrics")
        axes[0].legend()
        axes[0].grid(alpha=0.2)
    if not edge_cmp.empty:
        axes[1].scatter(edge_cmp["root_distance_px"], edge_cmp["length_cv_raw"], color="0.65", label="raw", s=35)
        axes[1].scatter(edge_cmp["root_distance_px"], edge_cmp["length_cv_smoothed"], color="tab:blue", label="smoothed", s=35)
        axes[1].set_ylabel("Edge length CV")
        axes[1].grid(alpha=0.2)
    if not bend_cmp.empty:
        axes[2].scatter(bend_cmp["root_distance_px"], bend_cmp["filtered_angle_rms_delta_deg_raw"], color="0.65", label="raw", s=35)
        axes[2].scatter(bend_cmp["root_distance_px"], bend_cmp["filtered_angle_rms_delta_deg_smoothed"], color="tab:red", label="smoothed", s=35)
        axes[2].set_xlabel("Root distance (px)")
        axes[2].set_ylabel("Filtered angle RMS")
        axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "smoothing_comparison.png", dpi=180)
    plt.close(fig)


def compute_root_distances(graph, coord_lookup):
    root = graph["root"]
    root_dist = {root: 0.0}
    edge_rest = {}

    q = deque([root])
    while q:
        parent = q.popleft()
        for child in graph["children"].get(parent, []):
            frames = common_frames_for_nodes(coord_lookup, [parent, child])
            if not frames:
                raise ValueError(f"No overlapping frames for edge {parent}->{child}.")
            rest_length = float(
                np.linalg.norm(
                    point_at(coord_lookup, child, frames[0]) - point_at(coord_lookup, parent, frames[0])
                )
            )
            edge_rest[(parent, child)] = rest_length
            root_dist[child] = root_dist[parent] + rest_length
            q.append(child)

    return root_dist, edge_rest


def assign_distance_zones(root_distance):
    max_dist = max(root_distance.values()) if root_distance else 0.0
    zones = {}
    for node_id, dist in root_distance.items():
        if max_dist <= 1e-9:
            zones[node_id] = "root"
            continue
        t = dist / max_dist
        if t <= 1.0 / 3.0:
            zones[node_id] = "proximal"
        elif t <= 2.0 / 3.0:
            zones[node_id] = "mid"
        else:
            zones[node_id] = "distal"
    return zones


def moving_block_bootstrap(values, metric_fn, block_size=25, n_boot=300, rng=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return np.nan, np.nan
    n = values.size
    block_size = int(max(3, min(block_size, n)))
    if rng is None:
        rng = np.random.default_rng(0)
    starts = np.arange(0, n - block_size + 1)
    stats = []
    for _ in range(n_boot):
        sampled = []
        while len(sampled) < n:
            start = int(rng.choice(starts))
            sampled.extend(values[start:start + block_size])
        sampled = np.asarray(sampled[:n], dtype=float)
        stat = metric_fn(sampled)
        if np.isfinite(stat):
            stats.append(stat)
    if not stats:
        return np.nan, np.nan
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def angle_between_deg(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return np.nan
    cos_theta = np.dot(a, b) / (na * nb)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def robust_inlier_mask(values, z_thresh=3.5):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return finite
    med = np.nanmedian(values[finite])
    mad = np.nanmedian(np.abs(values[finite] - med))
    if mad <= 1e-9:
        return finite
    robust_z = 0.6745 * np.abs(values - med) / mad
    return finite & (robust_z <= z_thresh)


def _prepare_relative_compliance_matrix(bend_metrics, bend_series):
    if bend_metrics.empty or bend_series.empty:
        return [], np.asarray([], dtype=int), np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=bool)

    bend_metrics = bend_metrics.sort_values(["root_distance_px", "joint", "child"]).reset_index(drop=True)
    edge_keys = [(int(row["joint"]), int(row["child"])) for _, row in bend_metrics.iterrows()]
    key_to_idx = {key: idx for idx, key in enumerate(edge_keys)}
    frames = np.array(sorted(bend_series["frame"].unique()), dtype=int)
    frame_to_idx = {int(frame): idx for idx, frame in enumerate(frames)}

    values = np.full((len(edge_keys), len(frames)), np.nan, dtype=float)
    valid_mask = np.zeros_like(values, dtype=bool)

    for _, row in bend_series.iterrows():
        key = (int(row["joint"]), int(row["child"]))
        if key not in key_to_idx:
            continue
        frame = int(row["frame"])
        if frame not in frame_to_idx:
            continue
        if not bool(row["is_inlier"]):
            continue
        delta = abs(float(row["delta_from_rest_deg"]))
        if not np.isfinite(delta):
            continue
        edge_idx = key_to_idx[key]
        frame_idx = frame_to_idx[frame]
        values[edge_idx, frame_idx] = delta
        valid_mask[edge_idx, frame_idx] = True

    return edge_keys, frames, values, valid_mask


def _fit_relative_compliance_matrix(values, valid_mask, prior_strength=RELATIVE_COMPLIANCE_PRIOR_STRENGTH, n_iter=60):
    n_edges, n_frames = values.shape
    if n_edges == 0 or n_frames == 0:
        return {
            "kappa": np.asarray([], dtype=float),
            "excitation": np.asarray([], dtype=float),
            "predicted": np.empty((0, 0), dtype=float),
            "support": np.asarray([], dtype=int),
            "rmse": np.asarray([], dtype=float),
            "r2": np.asarray([], dtype=float),
        }

    masked_values = np.where(valid_mask, values, np.nan)
    excitation = np.nanmedian(masked_values, axis=0)
    if not np.isfinite(excitation).any() or np.nanmax(excitation) <= 1e-9:
        excitation = np.nanmean(masked_values, axis=0)
    excitation = np.where(np.isfinite(excitation), excitation, 0.0)
    if np.nanmax(excitation) <= 1e-9:
        excitation = np.ones(n_frames, dtype=float)

    kappa = np.ones(n_edges, dtype=float)
    eps = 1e-6

    for _ in range(max(1, int(n_iter))):
        prev_kappa = kappa.copy()

        for edge_idx in range(n_edges):
            mask = valid_mask[edge_idx] & np.isfinite(excitation) & (excitation > eps)
            support = int(np.sum(mask))
            if support <= 0:
                kappa[edge_idx] = 1.0
                continue
            z = excitation[mask]
            y = values[edge_idx, mask]
            raw = float(np.dot(z, y) / max(np.dot(z, z), eps))
            raw = max(raw, eps)
            shrink = support / (support + max(float(prior_strength), 0.0))
            kappa[edge_idx] = math.exp(shrink * math.log(raw))

        finite_kappa = np.isfinite(kappa) & (kappa > eps)
        if np.any(finite_kappa):
            geom_mean = math.exp(float(np.mean(np.log(kappa[finite_kappa]))))
            if geom_mean > eps:
                kappa[finite_kappa] /= geom_mean

        for frame_idx in range(n_frames):
            mask = valid_mask[:, frame_idx] & np.isfinite(kappa) & (kappa > eps)
            if not np.any(mask):
                excitation[frame_idx] = 0.0
                continue
            k = kappa[mask]
            y = values[mask, frame_idx]
            excitation[frame_idx] = max(0.0, float(np.dot(k, y) / max(np.dot(k, k), eps)))

        delta = np.max(np.abs(np.log(np.clip(kappa, eps, None) / np.clip(prev_kappa, eps, None))))
        if delta < 1e-4:
            break

    predicted = kappa[:, None] * excitation[None, :]
    support = np.sum(valid_mask, axis=1).astype(int)
    rmse = np.full(n_edges, np.nan, dtype=float)
    r2 = np.full(n_edges, np.nan, dtype=float)

    for edge_idx in range(n_edges):
        mask = valid_mask[edge_idx]
        if not np.any(mask):
            continue
        observed = values[edge_idx, mask]
        fitted = predicted[edge_idx, mask]
        rmse[edge_idx] = float(np.sqrt(np.mean((observed - fitted) ** 2)))
        ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
        if ss_tot > eps:
            ss_res = float(np.sum((observed - fitted) ** 2))
            r2[edge_idx] = 1.0 - (ss_res / ss_tot)

    return {
        "kappa": kappa,
        "excitation": excitation,
        "predicted": predicted,
        "support": support,
        "rmse": rmse,
        "r2": r2,
    }


def estimate_relative_compliance(
    graph,
    bend_metrics,
    bend_series,
    seed=0,
    prior_strength=RELATIVE_COMPLIANCE_PRIOR_STRENGTH,
    n_boot=RELATIVE_COMPLIANCE_BOOTSTRAP_SAMPLES,
    block_size=25,
):
    edge_keys, frames, values, valid_mask = _prepare_relative_compliance_matrix(bend_metrics, bend_series)
    base_fit = _fit_relative_compliance_matrix(values, valid_mask, prior_strength=prior_strength)

    columns = [
        "parent",
        "child",
        "relative_compliance_kappa",
        "relative_compliance_ci_low",
        "relative_compliance_ci_high",
        "relative_stiffness_inv_kappa",
        "relative_stiffness_ci_low",
        "relative_stiffness_ci_high",
        "compliance_fit_rmse_deg",
        "compliance_fit_r2",
        "compliance_support_ratio",
        "compliance_n_frames",
        "excitation_rms_deg",
    ]
    if not edge_keys:
        return pd.DataFrame(columns=columns), pd.DataFrame(
            columns=["frame", "excitation_proxy_deg"]
        )

    ci_low = np.full(len(edge_keys), np.nan, dtype=float)
    ci_high = np.full(len(edge_keys), np.nan, dtype=float)
    if n_boot > 0 and len(frames) >= 8:
        rng = np.random.default_rng(seed)
        block_size = int(max(3, min(int(block_size), len(frames))))
        starts = np.arange(0, len(frames) - block_size + 1)
        boot_samples = [[] for _ in edge_keys]
        for _ in range(int(n_boot)):
            sampled = []
            while len(sampled) < len(frames):
                start = int(rng.choice(starts))
                sampled.extend(range(start, start + block_size))
            sampled = np.asarray(sampled[: len(frames)], dtype=int)
            fit = _fit_relative_compliance_matrix(
                values[:, sampled],
                valid_mask[:, sampled],
                prior_strength=prior_strength,
            )
            for edge_idx, kval in enumerate(fit["kappa"]):
                if np.isfinite(kval):
                    boot_samples[edge_idx].append(float(kval))
        for edge_idx, samples in enumerate(boot_samples):
            if samples:
                ci_low[edge_idx] = float(np.percentile(samples, 2.5))
                ci_high[edge_idx] = float(np.percentile(samples, 97.5))

    excitation_rms = float(np.sqrt(np.mean(base_fit["excitation"] ** 2))) if base_fit["excitation"].size else np.nan
    rows = []
    for edge_idx, (parent, child) in enumerate(edge_keys):
        kappa = float(base_fit["kappa"][edge_idx])
        kappa_lo = float(ci_low[edge_idx]) if np.isfinite(ci_low[edge_idx]) else np.nan
        kappa_hi = float(ci_high[edge_idx]) if np.isfinite(ci_high[edge_idx]) else np.nan
        rows.append(
            {
                "parent": parent,
                "child": child,
                "relative_compliance_kappa": kappa,
                "relative_compliance_ci_low": kappa_lo,
                "relative_compliance_ci_high": kappa_hi,
                "relative_stiffness_inv_kappa": (1.0 / kappa) if kappa > 1e-9 else np.nan,
                "relative_stiffness_ci_low": (1.0 / kappa_hi) if kappa_hi > 1e-9 else np.nan,
                "relative_stiffness_ci_high": (1.0 / kappa_lo) if kappa_lo > 1e-9 else np.nan,
                "compliance_fit_rmse_deg": float(base_fit["rmse"][edge_idx]),
                "compliance_fit_r2": float(base_fit["r2"][edge_idx]),
                "compliance_support_ratio": float(base_fit["support"][edge_idx] / max(len(frames), 1)),
                "compliance_n_frames": int(base_fit["support"][edge_idx]),
                "excitation_rms_deg": excitation_rms,
            }
        )

    excitation_series = pd.DataFrame(
        {
            "frame": frames.astype(int),
            "excitation_proxy_deg": base_fit["excitation"].astype(float),
        }
    )
    return pd.DataFrame(rows, columns=columns), excitation_series


def compute_node_metrics(graph, coord_lookup, root_distance, distance_zones):
    root = graph["root"]
    root_frames = set(coord_lookup[root].keys())
    node_rows = []
    node_series_rows = []

    for node in graph["nodes"]:
        frames = common_frames_for_nodes(coord_lookup, [root, node])
        if not frames:
            continue

        root0 = point_at(coord_lookup, root, frames[0])
        node0 = point_at(coord_lookup, node, frames[0])
        rel0 = node0 - root0

        abs_displacements = []
        rel_displacements = []
        for frame in frames:
            root_pt = point_at(coord_lookup, root, frame)
            node_pt = point_at(coord_lookup, node, frame)
            abs_disp = np.linalg.norm(node_pt - node0)
            rel_disp = np.linalg.norm((node_pt - root_pt) - rel0)
            abs_displacements.append(abs_disp)
            rel_displacements.append(rel_disp)
            node_series_rows.append(
                {
                    "node_id": node,
                    "depth": graph["depth"][node],
                    "root_distance_px": float(root_distance[node]),
                    "distance_zone": distance_zones[node],
                    "frame": frame,
                    "motion_px": float(abs_disp),
                    "rel_motion_px": float(rel_disp),
                }
            )

        abs_displacements = np.asarray(abs_displacements, dtype=float)
        rel_displacements = np.asarray(rel_displacements, dtype=float)

        node_rows.append(
            {
                "node_id": node,
                "depth": graph["depth"][node],
                "root_distance_px": float(root_distance[node]),
                "distance_zone": distance_zones[node],
                "n_frames": len(frames),
                "motion_rms_px": float(np.sqrt(np.mean(abs_displacements ** 2))),
                "motion_mean_px": float(np.mean(abs_displacements)),
                "rel_motion_rms_px": float(np.sqrt(np.mean(rel_displacements ** 2))),
                "rel_motion_mean_px": float(np.mean(rel_displacements)),
                "motion_max_px": float(np.max(abs_displacements)),
                "rel_motion_max_px": float(np.max(rel_displacements)),
                "rel_motion_rms_per_root_distance": (
                    float(np.sqrt(np.mean(rel_displacements ** 2))) / float(root_distance[node])
                    if root_distance[node] > 1e-9
                    else np.nan
                ),
                "rel_motion_rms_ci_low": np.nan,
                "rel_motion_rms_ci_high": np.nan,
                "track_support_ratio": float(len(frames) / max(len(root_frames), 1)),
            }
        )

    node_metrics = pd.DataFrame(node_rows, columns=NODE_METRIC_COLUMNS).sort_values(
        ["root_distance_px", "node_id"]
    ).reset_index(drop=True)
    node_series = pd.DataFrame(node_series_rows, columns=NODE_SERIES_COLUMNS).sort_values(
        ["root_distance_px", "node_id", "frame"]
    ).reset_index(drop=True)
    return node_metrics, node_series


def compute_edge_metrics(graph, coord_lookup, node_metrics, root_distance, distance_zones):
    node_motion = node_metrics.set_index("node_id").to_dict(orient="index")
    edge_rows = []
    edge_series_rows = []

    for parent, child in graph["edges"]:
        frames = common_frames_for_nodes(coord_lookup, [parent, child])
        if not frames:
            continue

        lengths = []
        for frame in frames:
            p0 = point_at(coord_lookup, parent, frame)
            p1 = point_at(coord_lookup, child, frame)
            length = np.linalg.norm(p1 - p0)
            lengths.append(length)
            edge_series_rows.append(
                {
                    "parent": parent,
                    "child": child,
                    "frame": frame,
                    "length_px": float(length),
                }
            )

        lengths = np.asarray(lengths, dtype=float)
        rest_length = lengths[0]
        parent_rel_motion = node_motion.get(parent, {}).get("rel_motion_rms_px", np.nan)
        child_rel_motion = node_motion.get(child, {}).get("rel_motion_rms_px", np.nan)
        motion_gain = np.nan
        if np.isfinite(parent_rel_motion) and parent_rel_motion > 1e-9:
            motion_gain = float(child_rel_motion / parent_rel_motion)
        elif np.isfinite(child_rel_motion):
            motion_gain = float(np.inf if child_rel_motion > 1e-9 else 1.0)

        edge_rows.append(
            {
                "parent": parent,
                "child": child,
                "depth": graph["depth"][child],
                "root_distance_px": float(root_distance[child]),
                "distance_zone": distance_zones[child],
                "n_frames": len(frames),
                "rest_length_px": float(rest_length),
                "length_mean_px": float(np.mean(lengths)),
                "length_std_px": float(np.std(lengths)),
                "length_cv": float(np.std(lengths) / np.mean(lengths)) if np.mean(lengths) > 1e-9 else np.nan,
                "length_cv_ci_low": np.nan,
                "length_cv_ci_high": np.nan,
                "length_range_px": float(np.max(lengths) - np.min(lengths)),
                "motion_gain_proxy": motion_gain,
                "child_rel_motion_rms_px": float(child_rel_motion),
                "parent_rel_motion_rms_px": float(parent_rel_motion),
            }
        )

    edge_metrics = pd.DataFrame(edge_rows, columns=EDGE_METRIC_COLUMNS).sort_values(
        ["root_distance_px", "parent", "child"]
    ).reset_index(drop=True)
    edge_series = pd.DataFrame(edge_series_rows, columns=EDGE_SERIES_COLUMNS).sort_values(
        ["parent", "child", "frame"]
    ).reset_index(drop=True)
    return edge_metrics, edge_series


def compute_bending_metrics(graph, coord_lookup, root_distance, distance_zones):
    bend_rows = []
    bend_series_rows = []

    for grandparent, joint, child in graph["bends"]:
        frames = common_frames_for_nodes(coord_lookup, [grandparent, joint, child])
        if not frames:
            continue

        angles = []
        for frame in frames:
            pg = point_at(coord_lookup, grandparent, frame)
            pj = point_at(coord_lookup, joint, frame)
            pc = point_at(coord_lookup, child, frame)
            parent_vec = pg - pj
            child_vec = pc - pj
            angle = angle_between_deg(parent_vec, child_vec)
            angles.append(angle)

        angles = np.asarray(angles, dtype=float)
        rest_angle = angles[0]
        delta = angles - rest_angle
        inlier_mask = robust_inlier_mask(delta)
        inlier_delta = delta[inlier_mask]
        inlier_angles = angles[inlier_mask]
        for frame, angle, dval, is_inlier in zip(frames, angles, delta, inlier_mask):
            bend_series_rows.append(
                {
                    "grandparent": grandparent,
                    "joint": joint,
                    "child": child,
                    "frame": frame,
                    "angle_deg": float(angle),
                    "delta_from_rest_deg": float(dval),
                    "is_inlier": bool(is_inlier),
                }
            )
        bend_rows.append(
            {
                "grandparent": grandparent,
                "joint": joint,
                "child": child,
                "depth": graph["depth"][child],
                "root_distance_px": float(root_distance[child]),
                "distance_zone": distance_zones[child],
                "n_frames": len(frames),
                "rest_angle_deg": float(rest_angle),
                "angle_mean_deg": float(np.nanmean(angles)),
                "angle_std_deg": float(np.nanstd(angles)),
                "angle_range_deg": float(np.nanmax(angles) - np.nanmin(angles)),
                "angle_rms_delta_deg": float(np.sqrt(np.nanmean(delta ** 2))),
                "angle_mean_abs_delta_deg": float(np.nanmean(np.abs(delta))),
                "n_inlier_frames": int(np.sum(inlier_mask)),
                "filtered_angle_std_deg": float(np.nanstd(inlier_angles)) if inlier_angles.size else np.nan,
                "filtered_angle_rms_delta_deg": (
                    float(np.sqrt(np.nanmean(inlier_delta ** 2))) if inlier_delta.size else np.nan
                ),
                "filtered_angle_mean_abs_delta_deg": (
                    float(np.nanmean(np.abs(inlier_delta))) if inlier_delta.size else np.nan
                ),
                "filtered_angle_rms_ci_low": np.nan,
                "filtered_angle_rms_ci_high": np.nan,
            }
        )

    bend_metrics = pd.DataFrame(bend_rows, columns=BEND_METRIC_COLUMNS).sort_values(
        ["root_distance_px", "joint", "child"]
    ).reset_index(drop=True)
    bend_series = pd.DataFrame(bend_series_rows, columns=BEND_SERIES_COLUMNS).sort_values(
        ["joint", "child", "frame"]
    ).reset_index(drop=True)
    return bend_metrics, bend_series


def combine_compliance(edge_metrics, bend_metrics):
    compliance = edge_metrics.copy()
    compliance["angle_rms_delta_deg"] = np.nan
    compliance["angle_std_deg"] = np.nan
    compliance["angle_compliance_proxy"] = np.nan
    compliance["filtered_angle_rms_delta_deg"] = np.nan
    compliance["filtered_angle_std_deg"] = np.nan
    compliance["filtered_angle_compliance_proxy"] = np.nan
    compliance["motion_per_root_distance"] = np.nan
    compliance["filtered_angle_per_length"] = np.nan
    compliance["filtered_angle_per_root_distance"] = np.nan
    compliance["motion_per_root_distance_ci_low"] = np.nan
    compliance["motion_per_root_distance_ci_high"] = np.nan
    compliance["filtered_angle_per_length_ci_low"] = np.nan
    compliance["filtered_angle_per_length_ci_high"] = np.nan
    compliance["filtered_angle_per_root_distance_ci_low"] = np.nan
    compliance["filtered_angle_per_root_distance_ci_high"] = np.nan
    compliance["relative_compliance_kappa"] = np.nan
    compliance["relative_compliance_ci_low"] = np.nan
    compliance["relative_compliance_ci_high"] = np.nan
    compliance["relative_stiffness_inv_kappa"] = np.nan
    compliance["relative_stiffness_ci_low"] = np.nan
    compliance["relative_stiffness_ci_high"] = np.nan
    compliance["compliance_fit_rmse_deg"] = np.nan
    compliance["compliance_fit_r2"] = np.nan
    compliance["compliance_support_ratio"] = np.nan
    compliance["compliance_n_frames"] = np.nan
    compliance["excitation_rms_deg"] = np.nan

    bend_lookup = {
        (int(row["joint"]), int(row["child"])): row
        for _, row in bend_metrics.iterrows()
    }

    for idx, row in compliance.iterrows():
        key = (int(row["parent"]), int(row["child"]))
        bend = bend_lookup.get(key)
        if bend is None:
            continue
        compliance.at[idx, "angle_rms_delta_deg"] = float(bend["angle_rms_delta_deg"])
        compliance.at[idx, "angle_std_deg"] = float(bend["angle_std_deg"])
        compliance.at[idx, "angle_compliance_proxy"] = float(bend["angle_rms_delta_deg"])
        compliance.at[idx, "filtered_angle_rms_delta_deg"] = float(bend["filtered_angle_rms_delta_deg"])
        compliance.at[idx, "filtered_angle_std_deg"] = float(bend["filtered_angle_std_deg"])
        compliance.at[idx, "filtered_angle_compliance_proxy"] = float(bend["filtered_angle_rms_delta_deg"])
        root_dist = float(row["root_distance_px"])
        rest_length = float(row["rest_length_px"])
        child_motion = float(row["child_rel_motion_rms_px"])
        compliance.at[idx, "motion_per_root_distance"] = (
            child_motion / root_dist if root_dist > 1e-9 else np.nan
        )
        compliance.at[idx, "filtered_angle_per_length"] = (
            float(bend["filtered_angle_rms_delta_deg"]) / rest_length if rest_length > 1e-9 else np.nan
        )
        compliance.at[idx, "filtered_angle_per_root_distance"] = (
            float(bend["filtered_angle_rms_delta_deg"]) / root_dist if root_dist > 1e-9 else np.nan
        )

    return compliance.sort_values(["root_distance_px", "parent", "child"]).reset_index(drop=True)


def add_relative_compliance_estimates(
    graph,
    bend_metrics,
    bend_series,
    compliance,
    seed=0,
    prior_strength=RELATIVE_COMPLIANCE_PRIOR_STRENGTH,
    n_boot=RELATIVE_COMPLIANCE_BOOTSTRAP_SAMPLES,
):
    rel_comp, excitation_series = estimate_relative_compliance(
        graph,
        bend_metrics,
        bend_series,
        seed=seed,
        prior_strength=prior_strength,
        n_boot=n_boot,
    )
    if rel_comp.empty:
        return compliance, excitation_series

    merge_cols = [
        "relative_compliance_kappa",
        "relative_compliance_ci_low",
        "relative_compliance_ci_high",
        "relative_stiffness_inv_kappa",
        "relative_stiffness_ci_low",
        "relative_stiffness_ci_high",
        "compliance_fit_rmse_deg",
        "compliance_fit_r2",
        "compliance_support_ratio",
        "compliance_n_frames",
        "excitation_rms_deg",
    ]
    compliance = compliance.merge(rel_comp, on=["parent", "child"], how="left", suffixes=("", "_rel"))
    for col in merge_cols:
        rel_col = f"{col}_rel"
        if rel_col in compliance.columns:
            compliance[col] = compliance[rel_col].combine_first(compliance[col])
            compliance = compliance.drop(columns=[rel_col])
    return compliance, excitation_series


def add_uncertainty_estimates(node_metrics, node_series, edge_metrics, edge_series, bend_metrics, bend_series, compliance, seed=0):
    rng = np.random.default_rng(seed)

    for idx, row in node_metrics.iterrows():
        node_id = int(row["node_id"])
        vals = node_series[node_series["node_id"] == node_id]["rel_motion_px"].to_numpy(dtype=float)
        low, high = moving_block_bootstrap(vals, lambda v: np.sqrt(np.mean(v ** 2)), rng=rng)
        node_metrics.at[idx, "rel_motion_rms_ci_low"] = low
        node_metrics.at[idx, "rel_motion_rms_ci_high"] = high

    for idx, row in edge_metrics.iterrows():
        parent = int(row["parent"])
        child = int(row["child"])
        vals = edge_series[(edge_series["parent"] == parent) & (edge_series["child"] == child)]["length_px"].to_numpy(dtype=float)
        low, high = moving_block_bootstrap(
            vals,
            lambda v: (np.std(v) / np.mean(v)) if np.mean(v) > 1e-9 else np.nan,
            rng=rng,
        )
        edge_metrics.at[idx, "length_cv_ci_low"] = low
        edge_metrics.at[idx, "length_cv_ci_high"] = high

    for idx, row in bend_metrics.iterrows():
        gp = int(row["grandparent"])
        joint = int(row["joint"])
        child = int(row["child"])
        group = bend_series[
            (bend_series["grandparent"] == gp)
            & (bend_series["joint"] == joint)
            & (bend_series["child"] == child)
            & (bend_series["is_inlier"] == True)
        ]
        vals = group["delta_from_rest_deg"].to_numpy(dtype=float)
        low, high = moving_block_bootstrap(vals, lambda v: np.sqrt(np.mean(v ** 2)), rng=rng)
        bend_metrics.at[idx, "filtered_angle_rms_ci_low"] = low
        bend_metrics.at[idx, "filtered_angle_rms_ci_high"] = high

    bend_ci_lookup = {
        (int(r["joint"]), int(r["child"])): (float(r["filtered_angle_rms_ci_low"]), float(r["filtered_angle_rms_ci_high"]))
        for _, r in bend_metrics.iterrows()
    }
    node_ci_lookup = {
        int(r["node_id"]): (float(r["rel_motion_rms_ci_low"]), float(r["rel_motion_rms_ci_high"]))
        for _, r in node_metrics.iterrows()
    }
    for idx, row in compliance.iterrows():
        child = int(row["child"])
        root_dist = float(row["root_distance_px"])
        rest_length = float(row["rest_length_px"])
        motion_low, motion_high = node_ci_lookup.get(child, (np.nan, np.nan))
        if root_dist > 1e-9:
            compliance.at[idx, "motion_per_root_distance_ci_low"] = motion_low / root_dist
            compliance.at[idx, "motion_per_root_distance_ci_high"] = motion_high / root_dist
        angle_low, angle_high = bend_ci_lookup.get((int(row["parent"]), child), (np.nan, np.nan))
        if rest_length > 1e-9:
            compliance.at[idx, "filtered_angle_per_length_ci_low"] = angle_low / rest_length
            compliance.at[idx, "filtered_angle_per_length_ci_high"] = angle_high / rest_length
        if root_dist > 1e-9:
            compliance.at[idx, "filtered_angle_per_root_distance_ci_low"] = angle_low / root_dist
            compliance.at[idx, "filtered_angle_per_root_distance_ci_high"] = angle_high / root_dist

    return node_metrics, edge_metrics, bend_metrics, compliance


def summarize_by_zone(node_metrics, edge_metrics, bend_metrics, compliance):
    rows = []
    zone_order = ["proximal", "mid", "distal", "root"]
    zones = [z for z in zone_order if z in set(node_metrics["distance_zone"]) | set(edge_metrics["distance_zone"]) | set(bend_metrics["distance_zone"])]
    for zone in zones:
        node_group = node_metrics[node_metrics["distance_zone"] == zone]
        edge_group = edge_metrics[edge_metrics["distance_zone"] == zone]
        bend_group = bend_metrics[bend_metrics["distance_zone"] == zone]
        comp_group = compliance[compliance["distance_zone"] == zone]
        rows.append(
            {
                "distance_zone": zone,
                "n_nodes": int(len(node_group)),
                "n_edges": int(len(edge_group)),
                "n_bends": int(len(bend_group)),
                "node_rel_motion_rms_mean": float(node_group["rel_motion_rms_px"].mean()) if not node_group.empty else np.nan,
                "node_rel_motion_rms_std": float(node_group["rel_motion_rms_px"].std()) if len(node_group) > 1 else np.nan,
                "edge_length_cv_mean": float(edge_group["length_cv"].mean()) if not edge_group.empty else np.nan,
                "filtered_angle_rms_mean": float(bend_group["filtered_angle_rms_delta_deg"].mean()) if not bend_group.empty else np.nan,
                "motion_per_root_distance_mean": float(comp_group["motion_per_root_distance"].mean()) if not comp_group.empty else np.nan,
                "filtered_angle_per_length_mean": float(comp_group["filtered_angle_per_length"].mean()) if not comp_group.empty else np.nan,
                "relative_compliance_kappa_mean": float(comp_group["relative_compliance_kappa"].mean()) if not comp_group.empty else np.nan,
                "relative_stiffness_inv_kappa_mean": float(comp_group["relative_stiffness_inv_kappa"].mean()) if not comp_group.empty else np.nan,
                "compliance_fit_r2_mean": float(comp_group["compliance_fit_r2"].mean()) if not comp_group.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subplot_grid(n_panels, ncols=3):
    if n_panels <= 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        return fig, np.array([ax]), 1, 1
    ncols = min(max(1, ncols), n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    return fig, axes.ravel(), nrows, ncols


def hide_unused_axes(axes, used_count):
    for ax in axes[used_count:]:
        ax.set_visible(False)


def plot_depth_vs_compliance(compliance_df, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    motion = compliance_df[np.isfinite(compliance_df["motion_gain_proxy"])]
    if not motion.empty:
        axes[0].scatter(
            motion["depth"],
            motion["motion_gain_proxy"],
            color="tab:blue",
            s=45,
        )
        motion_mean = motion.groupby("depth")["motion_gain_proxy"].mean().reset_index()
        axes[0].plot(motion_mean["depth"], motion_mean["motion_gain_proxy"], color="tab:blue", alpha=0.7)
        for _, row in motion.iterrows():
            axes[0].annotate(
                f"{int(row['parent'])}->{int(row['child'])}",
                (row["depth"], row["motion_gain_proxy"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="tab:blue",
            )
    axes[0].set_ylabel("Motion gain proxy")
    axes[0].set_title("Depth vs motion gain proxy")
    axes[0].grid(alpha=0.2)

    rel = compliance_df[np.isfinite(compliance_df["relative_compliance_kappa"])]
    if not rel.empty:
        axes[1].scatter(
            rel["depth"],
            rel["relative_compliance_kappa"],
            color="tab:red",
            s=45,
            marker="s",
        )
        rel_mean = rel.groupby("depth")["relative_compliance_kappa"].mean().reset_index()
        axes[1].plot(rel_mean["depth"], rel_mean["relative_compliance_kappa"], color="tab:red", alpha=0.7)
        for _, row in rel.iterrows():
            axes[1].annotate(
                f"{int(row['parent'])}->{int(row['child'])}",
                (row["depth"], row["relative_compliance_kappa"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )
    axes[1].set_ylabel("Relative compliance $\\kappa_e$")
    axes[1].set_title("Depth vs relative compliance estimate")
    axes[1].grid(alpha=0.2)

    angle = compliance_df[np.isfinite(compliance_df["filtered_angle_per_length"])]
    if not angle.empty:
        axes[2].scatter(
            angle["depth"],
            angle["filtered_angle_per_length"],
            color="tab:purple",
            s=45,
            marker="^",
        )
        angle_mean = angle.groupby("depth")["filtered_angle_per_length"].mean().reset_index()
        axes[2].plot(angle_mean["depth"], angle_mean["filtered_angle_per_length"], color="tab:purple", alpha=0.7)
        for _, row in angle.iterrows():
            axes[2].annotate(
                f"{int(row['parent'])}->{int(row['child'])}",
                (row["depth"], row["filtered_angle_per_length"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="tab:purple",
            )
    axes[2].set_xlabel("Edge depth")
    axes[2].set_ylabel("Filtered angle RMS / length")
    axes[2].set_title("Depth vs normalized bending proxy")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_root_distance_vs_compliance(compliance_df, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)

    cols = [
        ("relative_compliance_kappa", "tab:red", "Relative compliance $\\kappa_e$"),
        ("motion_per_root_distance", "tab:blue", "Motion / root distance"),
        ("filtered_angle_rms_delta_deg", "tab:orange", "Filtered angle RMS delta (deg)"),
        ("filtered_angle_per_length", "tab:purple", "Filtered angle RMS / edge length"),
    ]
    for ax, (col, color, ylabel) in zip(axes, cols):
        rows = compliance_df[np.isfinite(compliance_df[col])]
        if not rows.empty:
            ax.scatter(rows["root_distance_px"], rows[col], color=color, s=45)
            ordered = rows.sort_values("root_distance_px")
            ax.plot(ordered["root_distance_px"], ordered[col], color=color, alpha=0.6)
            for _, row in rows.iterrows():
                ax.annotate(
                    f"{int(row['parent'])}->{int(row['child'])}",
                    (row["root_distance_px"], row[col]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_title("Root distance vs relative compliance and proxies")
    axes[-1].set_xlabel("Root-to-edge distance (px)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_root_distance_vs_compliance_uncertainty(compliance_df, out_path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)
    specs = [
        ("relative_compliance_kappa", "relative_compliance_ci_low", "relative_compliance_ci_high", "tab:red", "Relative compliance $\\kappa_e$"),
        ("motion_per_root_distance", "motion_per_root_distance_ci_low", "motion_per_root_distance_ci_high", "tab:blue", "Motion / root distance"),
        ("filtered_angle_per_length", "filtered_angle_per_length_ci_low", "filtered_angle_per_length_ci_high", "tab:purple", "Filtered angle RMS / edge length"),
        ("filtered_angle_per_root_distance", "filtered_angle_per_root_distance_ci_low", "filtered_angle_per_root_distance_ci_high", "tab:green", "Filtered angle RMS / root distance"),
    ]
    for ax, (col, lo_col, hi_col, color, ylabel) in zip(axes, specs):
        rows = compliance_df[np.isfinite(compliance_df[col])]
        if not rows.empty:
            yerr = np.vstack([
                np.maximum(0.0, rows[col].to_numpy(dtype=float) - rows[lo_col].to_numpy(dtype=float)),
                np.maximum(0.0, rows[hi_col].to_numpy(dtype=float) - rows[col].to_numpy(dtype=float)),
            ])
            ax.errorbar(
                rows["root_distance_px"],
                rows[col],
                yerr=yerr,
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1,
                capsize=3,
                alpha=0.8,
            )
            for _, row in rows.iterrows():
                ax.annotate(
                    f"{int(row['parent'])}->{int(row['child'])}",
                    (row["root_distance_px"], row[col]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[0].set_title("Relative compliance and proxies with bootstrap uncertainty")
    axes[-1].set_xlabel("Root-to-edge distance (px)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_zone_summary(zone_summary, out_path):
    if zone_summary.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    x = np.arange(len(zone_summary))
    labels = zone_summary["distance_zone"].tolist()
    axes[0].bar(x, zone_summary["node_rel_motion_rms_mean"], color="tab:green")
    axes[0].set_ylabel("Node motion RMS mean")
    axes[0].set_title("Zone summary")
    axes[1].bar(x, zone_summary["edge_length_cv_mean"], color="tab:orange")
    axes[1].set_ylabel("Edge length CV mean")
    axes[2].bar(x, zone_summary["filtered_angle_per_length_mean"], color="tab:purple")
    axes[2].set_ylabel("Filtered angle / length mean")
    axes[3].bar(x, zone_summary["relative_compliance_kappa_mean"], color="tab:red")
    axes[3].set_ylabel("Mean relative $\\kappa_e$")
    axes[3].set_xticks(x, labels)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_edge_lengths(edge_series, out_path):
    groups = list(edge_series.groupby(["parent", "child"]))
    fig, axes, _, _ = subplot_grid(len(groups), ncols=3)
    for idx, ((parent, child), group) in enumerate(groups):
        ax = axes[idx]
        ax.plot(group["frame"], group["length_px"], linewidth=1.4, color="tab:blue")
        ax.set_title(f"Edge {parent}->{child}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Length (px)")
        ax.grid(alpha=0.2)
    hide_unused_axes(axes, len(groups))
    fig.suptitle("Edge lengths over time", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_edge_length_summary(edge_metrics, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for _, row in edge_metrics.iterrows():
        label = f"{int(row['parent'])}->{int(row['child'])}"
        axes[0].scatter(row["root_distance_px"], row["length_std_px"], color="tab:blue", s=45)
        axes[0].annotate(label, (row["root_distance_px"], row["length_std_px"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axes[1].scatter(row["root_distance_px"], row["length_cv"], color="tab:orange", s=45)
        axes[1].annotate(label, (row["root_distance_px"], row["length_cv"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[0].set_ylabel("Length std (px)")
    axes[0].set_title("Edge length variability vs root distance")
    axes[0].grid(alpha=0.2)
    axes[1].set_xlabel("Root-to-edge distance (px)")
    axes[1].set_ylabel("Length CV")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bending_angles(bend_series, out_path):
    groups = list(bend_series.groupby(["joint", "child"]))
    fig, axes, _, _ = subplot_grid(len(groups), ncols=3)
    for idx, ((joint, child), group) in enumerate(groups):
        grandparent = int(group["grandparent"].iloc[0])
        ax = axes[idx]
        ax.plot(group["frame"], group["angle_deg"], linewidth=1.4, color="tab:red")
        ax.set_title(f"Bend {grandparent}->{joint}->{child}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (deg)")
        ax.grid(alpha=0.2)
    hide_unused_axes(axes, len(groups))
    fig.suptitle("Bending angles over time", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bending_summary(bend_metrics, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    for _, row in bend_metrics.iterrows():
        label = f"{int(row['grandparent'])}->{int(row['joint'])}->{int(row['child'])}"
        axes[0].scatter(row["root_distance_px"], row["angle_std_deg"], color="tab:red", s=45)
        axes[0].annotate(label, (row["root_distance_px"], row["angle_std_deg"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axes[1].scatter(row["root_distance_px"], row["angle_rms_delta_deg"], color="tab:purple", s=45)
        axes[1].annotate(label, (row["root_distance_px"], row["angle_rms_delta_deg"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        axes[2].scatter(row["root_distance_px"], row["filtered_angle_rms_delta_deg"], color="tab:green", s=45)
        axes[2].annotate(label, (row["root_distance_px"], row["filtered_angle_rms_delta_deg"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axes[0].set_ylabel("Angle std (deg)")
    axes[0].set_title("Bending variability vs root distance")
    axes[0].grid(alpha=0.2)
    axes[1].set_ylabel("Angle RMS delta (deg)")
    axes[1].grid(alpha=0.2)
    axes[2].set_xlabel("Root-to-bend distance (px)")
    axes[2].set_ylabel("Filtered angle RMS delta (deg)")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_node_motion_timeseries(node_series, out_path):
    groups = list(node_series.groupby("node_id"))
    fig, axes, _, _ = subplot_grid(len(groups), ncols=3)
    for idx, (node_id, group) in enumerate(groups):
        depth = int(group["depth"].iloc[0])
        ax = axes[idx]
        ax.plot(group["frame"], group["rel_motion_px"], linewidth=1.4, color="tab:green")
        ax.set_title(f"Node {node_id} (depth {depth})")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Root-relative motion (px)")
        ax.grid(alpha=0.2)
    hide_unused_axes(axes, len(groups))
    fig.suptitle("Node root-relative motion over time", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_node_motion(node_metrics, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    axes[0].scatter(node_metrics["root_distance_px"], node_metrics["rel_motion_rms_px"], color="tab:green", s=45)
    ordered = node_metrics.sort_values("root_distance_px")
    axes[0].plot(ordered["root_distance_px"], ordered["rel_motion_rms_px"], color="tab:green", alpha=0.7)
    for _, row in node_metrics.iterrows():
        axes[0].annotate(
            f"{int(row['node_id'])}",
            (row["root_distance_px"], row["rel_motion_rms_px"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="tab:green",
        )
    axes[0].set_ylabel("Root-relative motion RMS (px)")
    axes[0].set_title("Node motion vs root distance")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(node_metrics["root_distance_px"], node_metrics["rel_motion_max_px"], color="tab:olive", s=45)
    axes[1].plot(ordered["root_distance_px"], ordered["rel_motion_max_px"], color="tab:olive", alpha=0.7)
    for _, row in node_metrics.iterrows():
        axes[1].annotate(
            f"{int(row['node_id'])}",
            (row["root_distance_px"], row["rel_motion_max_px"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="tab:olive",
        )
    axes[1].set_ylabel("Root-relative motion max (px)")
    axes[1].set_title("Node max motion vs root distance")
    axes[1].grid(alpha=0.2)

    rows = node_metrics[np.isfinite(node_metrics["rel_motion_rms_per_root_distance"])]
    axes[2].scatter(rows["root_distance_px"], rows["rel_motion_rms_per_root_distance"], color="tab:cyan", s=45)
    ordered_rows = rows.sort_values("root_distance_px")
    axes[2].plot(ordered_rows["root_distance_px"], ordered_rows["rel_motion_rms_per_root_distance"], color="tab:cyan", alpha=0.7)
    for _, row in rows.iterrows():
        axes[2].annotate(
            f"{int(row['node_id'])}",
            (row["root_distance_px"], row["rel_motion_rms_per_root_distance"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color="tab:cyan",
        )
    axes[2].set_xlabel("Root-to-node distance (px)")
    axes[2].set_ylabel("Motion RMS / root distance")
    axes[2].set_title("Length-normalized node motion")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


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


def annotate_video(video_path, out_path, graph, coord_lookup):
    video_path = Path(video_path)
    out_path = Path(out_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    colors = build_color_map(graph["nodes"])
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_pts = {
            node_id: coords[frame_idx]
            for node_id, coords in coord_lookup.items()
            if frame_idx in coords
        }

        for parent, child in graph["edges"]:
            if parent in current_pts and child in current_pts:
                p0 = tuple(current_pts[parent].astype(int))
                p1 = tuple(current_pts[child].astype(int))
                cv2.line(frame, p0, p1, (220, 220, 220), 1, lineType=cv2.LINE_AA)

        for node_id, pt in current_pts.items():
            p = tuple(pt.astype(int))
            color = colors[node_id]
            radius = 5 if node_id == graph["root"] else 4
            cv2.circle(frame, p, radius, color, thickness=-1, lineType=cv2.LINE_AA)
            if node_id == graph["root"]:
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

        cv2.putText(
            frame,
            f"frame {frame_idx}",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def save_graph_copy(graph_path, out_dir):
    out_path = out_dir / "graph_spec_used.json"
    out_path.write_text(Path(graph_path).read_text())


def analyze(
    csv_path,
    graph_path,
    out_dir=None,
    video_path=None,
    smooth_window=5,
    use_factor_graph_smoother=False,
    fg_measurement_weight=1.0,
    fg_temporal_weight=0.35,
    fg_acceleration_weight=0.2,
    fg_edge_weight=0.45,
    compliance_prior_strength=RELATIVE_COMPLIANCE_PRIOR_STRENGTH,
    compliance_bootstrap_samples=RELATIVE_COMPLIANCE_BOOTSTRAP_SAMPLES,
):
    csv_path = Path(csv_path)
    graph_path = Path(graph_path)
    if out_dir is None:
        out_dir = csv_path.parent / "branch_compliance_analysis"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_path)
    total_csv_nodes = count_all_csv_nodes(csv_path)
    _, _, raw_coord_lookup = load_tracks(csv_path, graph["nodes"])
    baseline_coord_lookup = smooth_coord_lookup(raw_coord_lookup, window=smooth_window)
    coord_lookup = baseline_coord_lookup
    if use_factor_graph_smoother:
        coord_lookup = factor_graph_style_smoother(
            graph,
            baseline_coord_lookup,
            measurement_weight=fg_measurement_weight,
            temporal_weight=fg_temporal_weight,
            acceleration_weight=fg_acceleration_weight,
            edge_weight=fg_edge_weight,
        )

    root_distance, _ = compute_root_distances(graph, coord_lookup)
    distance_zones = assign_distance_zones(root_distance)

    node_metrics, node_series = compute_node_metrics(graph, coord_lookup, root_distance, distance_zones)
    edge_metrics, edge_series = compute_edge_metrics(graph, coord_lookup, node_metrics, root_distance, distance_zones)
    bend_metrics, bend_series = compute_bending_metrics(graph, coord_lookup, root_distance, distance_zones)
    compliance = combine_compliance(edge_metrics, bend_metrics)
    node_metrics, edge_metrics, bend_metrics, compliance = add_uncertainty_estimates(
        node_metrics,
        node_series,
        edge_metrics,
        edge_series,
        bend_metrics,
        bend_series,
        compliance,
    )
    compliance, excitation_series = add_relative_compliance_estimates(
        graph,
        bend_metrics,
        bend_series,
        compliance,
        seed=0,
        prior_strength=compliance_prior_strength,
        n_boot=compliance_bootstrap_samples,
    )
    zone_summary = summarize_by_zone(node_metrics, edge_metrics, bend_metrics, compliance)

    coord_lookup_to_dataframe(coord_lookup).to_csv(out_dir / "analysis_tracks.csv", index=False)

    node_metrics.to_csv(out_dir / "node_metrics.csv", index=False)
    node_series.to_csv(out_dir / "node_motion_timeseries.csv", index=False)
    edge_metrics.to_csv(out_dir / "edge_metrics.csv", index=False)
    bend_metrics.to_csv(out_dir / "bending_metrics.csv", index=False)
    compliance.to_csv(out_dir / "compliance_summary.csv", index=False)
    zone_summary.to_csv(out_dir / "zone_summary.csv", index=False)
    if not excitation_series.empty:
        excitation_series.to_csv(out_dir / "relative_compliance_excitation_timeseries.csv", index=False)

    if not edge_series.empty:
        edge_series.to_csv(out_dir / "edge_length_timeseries.csv", index=False)
        plot_edge_lengths(edge_series, out_dir / "all_edge_lengths_subplots.png")
    if not edge_metrics.empty:
        plot_edge_length_summary(edge_metrics, out_dir / "edge_length_vs_root_distance.png")

    if not bend_series.empty:
        bend_series.to_csv(out_dir / "bending_angle_timeseries.csv", index=False)
        plot_bending_angles(bend_series, out_dir / "all_bending_angles_subplots.png")
    if not bend_metrics.empty:
        plot_bending_summary(bend_metrics, out_dir / "bending_vs_root_distance.png")

    if not node_metrics.empty:
        plot_node_motion(node_metrics, out_dir / "node_motion_vs_root_distance.png")
    if not node_series.empty:
        plot_node_motion_timeseries(node_series, out_dir / "all_node_motion_subplots.png")

    if not compliance.empty:
        plot_depth_vs_compliance(compliance, out_dir / "depth_vs_compliance.png")
        plot_root_distance_vs_compliance(compliance, out_dir / "root_distance_vs_compliance.png")
        plot_root_distance_vs_compliance_uncertainty(compliance, out_dir / "root_distance_vs_compliance_uncertainty.png")
    if not zone_summary.empty:
        plot_zone_summary(zone_summary, out_dir / "zone_summary.png")

    if video_path is not None:
        annotate_video(video_path, out_dir / "annotated_nodes.mp4", graph, raw_coord_lookup)

    if use_factor_graph_smoother:
        raw_root_distance, _ = compute_root_distances(graph, baseline_coord_lookup)
        raw_distance_zones = assign_distance_zones(raw_root_distance)
        raw_node_metrics, raw_node_series = compute_node_metrics(graph, baseline_coord_lookup, raw_root_distance, raw_distance_zones)
        raw_edge_metrics, raw_edge_series = compute_edge_metrics(graph, baseline_coord_lookup, raw_node_metrics, raw_root_distance, raw_distance_zones)
        raw_bend_metrics, raw_bend_series = compute_bending_metrics(graph, baseline_coord_lookup, raw_root_distance, raw_distance_zones)
        raw_compliance = combine_compliance(raw_edge_metrics, raw_bend_metrics)
        raw_node_metrics, raw_edge_metrics, raw_bend_metrics, raw_compliance = add_uncertainty_estimates(
            raw_node_metrics,
            raw_node_series,
            raw_edge_metrics,
            raw_edge_series,
            raw_bend_metrics,
            raw_bend_series,
            raw_compliance,
            seed=1,
        )
        raw_compliance, _ = add_relative_compliance_estimates(
            graph,
            raw_bend_metrics,
            raw_bend_series,
            raw_compliance,
            seed=1,
            prior_strength=compliance_prior_strength,
            n_boot=max(50, compliance_bootstrap_samples // 3),
        )
        node_cmp, edge_cmp, bend_cmp = build_smoothing_comparison(
            raw_node_metrics,
            node_metrics,
            raw_edge_metrics,
            edge_metrics,
            raw_bend_metrics,
            bend_metrics,
        )
        node_cmp.to_csv(out_dir / "smoothing_node_comparison.csv", index=False)
        edge_cmp.to_csv(out_dir / "smoothing_edge_comparison.csv", index=False)
        bend_cmp.to_csv(out_dir / "smoothing_bending_comparison.csv", index=False)
        plot_smoothing_comparison(node_cmp, edge_cmp, bend_cmp, out_dir)
        coord_lookup_to_dataframe(baseline_coord_lookup).to_csv(out_dir / "baseline_smoothed_tracks.csv", index=False)

    save_graph_copy(graph_path, out_dir)

    print(
        f"Graph covers {len(graph['nodes'])} nodes, while CSV contains {total_csv_nodes} unique tracked nodes."
    )
    print(f"Analysis uses smoothed tracks with moving-average window = {smooth_window} frames.")
    if use_factor_graph_smoother:
        print(
            "Applied factor-graph-style smoother with weights "
            f"(measurement={fg_measurement_weight}, temporal={fg_temporal_weight}, "
            f"acceleration={fg_acceleration_weight}, edge={fg_edge_weight})."
        )
    print(f"Saved branch compliance analysis to: {out_dir}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze TAPIR tracks on a manually defined rooted branch graph. "
            "Computes edge length stability, bending angles, node motion, "
            "and simple compliance proxies."
        )
    )
    parser.add_argument("csv_path", help="Path to TAPIR track CSV with columns point_id,frame,x,y")
    parser.add_argument(
        "graph_json",
        help=(
            "Path to manual rooted graph JSON. Format: "
            '{"root": 0, "edges": [[0, 1], [1, 2], [1, 3]]}'
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for CSV summaries and plots",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional source video path for an annotated node/edge overlay video",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd moving-average window for smoothing TAPIR tracks before analysis",
    )
    parser.add_argument(
        "--use-factor-graph-smoother",
        action="store_true",
        help="Run a quadratic factor-graph-style position smoother after the moving-average preprocessing",
    )
    parser.add_argument("--fg-measurement-weight", type=float, default=1.0, help="Weight for TAPIR measurement factors")
    parser.add_argument("--fg-temporal-weight", type=float, default=0.35, help="Weight for frame-to-frame smoothness factors")
    parser.add_argument("--fg-acceleration-weight", type=float, default=0.2, help="Weight for second-difference temporal factors")
    parser.add_argument("--fg-edge-weight", type=float, default=0.45, help="Weight for soft edge-shape factors")
    parser.add_argument(
        "--compliance-prior-strength",
        type=float,
        default=RELATIVE_COMPLIANCE_PRIOR_STRENGTH,
        help="Shrinkage strength pulling per-edge relative compliance toward kappa=1 when bending support is weak",
    )
    parser.add_argument(
        "--compliance-bootstrap-samples",
        type=int,
        default=RELATIVE_COMPLIANCE_BOOTSTRAP_SAMPLES,
        help="Number of moving-block bootstrap resamples for relative compliance confidence intervals",
    )
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    analyze(
        args.csv_path,
        args.graph_json,
        out_dir=args.out_dir,
        video_path=args.video_path,
        smooth_window=args.smooth_window,
        use_factor_graph_smoother=args.use_factor_graph_smoother,
        fg_measurement_weight=args.fg_measurement_weight,
        fg_temporal_weight=args.fg_temporal_weight,
        fg_acceleration_weight=args.fg_acceleration_weight,
        fg_edge_weight=args.fg_edge_weight,
        compliance_prior_strength=args.compliance_prior_strength,
        compliance_bootstrap_samples=args.compliance_bootstrap_samples,
    )
