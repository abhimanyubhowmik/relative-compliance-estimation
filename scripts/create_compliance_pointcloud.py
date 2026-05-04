#!/usr/bin/env python3
import argparse
import base64
import bz2
import html
import shutil
import struct
import subprocess
from dataclasses import dataclass
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


OP_MSG_DATA = 2
OP_CHUNK = 5
OP_CONNECTION = 7


@dataclass
class ImageFrame:
    topic: str
    bag_stamp: float
    header_stamp: float
    frame_id: str
    encoding: str
    image: np.ndarray


@dataclass
class CameraInfo:
    topic: str
    width: int
    height: int
    k: np.ndarray
    frame_id: str


@dataclass
class Transform:
    topic: str
    translation: np.ndarray
    rotation_xyzw: np.ndarray


def parse_ros_fields(buf):
    fields = {}
    offset = 0
    while offset + 4 <= len(buf):
        field_len = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        field = buf[offset : offset + field_len]
        offset += field_len
        if b"=" in field:
            key, value = field.split(b"=", 1)
            fields[key.decode("ascii", errors="replace")] = value
    return fields


def field_to_int(value):
    if len(value) == 1:
        return value[0]
    if len(value) == 4:
        return struct.unpack("<I", value)[0]
    if len(value) == 8:
        return struct.unpack("<Q", value)[0]
    return int.from_bytes(value, byteorder="little", signed=False)


def field_to_time(value):
    if len(value) != 8:
        return np.nan
    secs, nsecs = struct.unpack("<II", value)
    return float(secs) + float(nsecs) * 1e-9


def iter_bag_file_records(bag_path):
    with open(bag_path, "rb") as f:
        version = f.readline()
        if not version.startswith(b"#ROSBAG V2.0"):
            raise ValueError(f"{bag_path} is not a ROS bag v2.0 file")

        while True:
            header_len_bytes = f.read(4)
            if len(header_len_bytes) < 4:
                break
            header_len = struct.unpack("<I", header_len_bytes)[0]
            header = parse_ros_fields(f.read(header_len))
            data_len = struct.unpack("<I", f.read(4))[0]
            data = f.read(data_len)
            yield header, data


def iter_chunk_records(data):
    offset = 0
    while offset + 4 <= len(data):
        header_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if offset + header_len + 4 > len(data):
            break
        header = parse_ros_fields(data[offset : offset + header_len])
        offset += header_len
        data_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        payload = data[offset : offset + data_len]
        offset += data_len
        yield header, payload


def decompress_chunk(header, data):
    compression = header.get("compression", b"none").decode("ascii", errors="replace")
    if compression == "none":
        return data
    if compression == "bz2":
        return bz2.decompress(data)
    if compression == "lz4":
        try:
            import lz4.frame

            return lz4.frame.decompress(data)
        except ImportError:
            lz4_bin = shutil.which("lz4")
            if not lz4_bin:
                raise RuntimeError(
                    "This bag uses lz4 compression. Install python-lz4 or make the lz4 CLI available on PATH."
                )
            result = subprocess.run(
                [lz4_bin, "-dc"],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
            return result.stdout
    raise RuntimeError(f"Unsupported bag compression: {compression}")


def iter_bag_messages(bag_path, topics=None):
    topics = set(topics) if topics is not None else None
    connections = {}

    for file_header, chunk_payload in iter_bag_file_records(bag_path):
        if field_to_int(file_header.get("op", b"\x00")) != OP_CHUNK:
            continue

        chunk = decompress_chunk(file_header, chunk_payload)
        for record_header, payload in iter_chunk_records(chunk):
            op = field_to_int(record_header.get("op", b"\x00"))
            if op == OP_CONNECTION:
                conn = field_to_int(record_header["conn"])
                topic = record_header.get("topic", b"").decode("utf-8", errors="replace")
                meta = parse_ros_fields(payload)
                msg_type = meta.get("type", b"").decode("utf-8", errors="replace")
                connections[conn] = (topic, msg_type)
                continue

            if op != OP_MSG_DATA:
                continue

            conn = field_to_int(record_header["conn"])
            if conn not in connections:
                continue
            topic, msg_type = connections[conn]
            if topics is not None and topic not in topics:
                continue
            yield topic, msg_type, payload, field_to_time(record_header.get("time", b""))


def read_u32(buf, offset):
    return struct.unpack_from("<I", buf, offset)[0], offset + 4


def read_f32(buf, offset):
    return struct.unpack_from("<f", buf, offset)[0], offset + 4


def read_f64(buf, offset):
    return struct.unpack_from("<d", buf, offset)[0], offset + 8


def read_string(buf, offset):
    length, offset = read_u32(buf, offset)
    value = buf[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def parse_std_header(buf, offset=0):
    _seq, offset = read_u32(buf, offset)
    secs, offset = read_u32(buf, offset)
    nsecs, offset = read_u32(buf, offset)
    frame_id, offset = read_string(buf, offset)
    return float(secs) + float(nsecs) * 1e-9, frame_id, offset


def parse_image(topic, payload, bag_stamp):
    header_stamp, frame_id, offset = parse_std_header(payload)
    height, offset = read_u32(payload, offset)
    width, offset = read_u32(payload, offset)
    encoding, offset = read_string(payload, offset)
    is_bigendian = payload[offset]
    offset += 1
    step, offset = read_u32(payload, offset)
    data_len, offset = read_u32(payload, offset)
    image_bytes = payload[offset : offset + data_len]

    encoding_lower = encoding.lower()
    if encoding_lower in {"rgb8", "bgr8", "8uc3"}:
        channels = 3
        dtype = np.uint8
    elif encoding_lower in {"mono8", "8uc1"}:
        channels = 1
        dtype = np.uint8
    elif encoding_lower in {"mono16", "16uc1", "z16"}:
        channels = 1
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
    elif encoding_lower in {"32fc1"}:
        channels = 1
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
    else:
        raise ValueError(f"Unsupported image encoding on {topic}: {encoding}")

    itemsize = np.dtype(dtype).itemsize
    expected_row_bytes = width * channels * itemsize
    rows = np.frombuffer(image_bytes, dtype=np.uint8).reshape(height, step)
    rows = rows[:, :expected_row_bytes]
    array = rows.reshape(-1).view(dtype).reshape(height, width, channels)
    if channels == 1:
        array = array[:, :, 0]
    else:
        array = array.copy()
        if encoding_lower == "bgr8":
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if array.dtype.byteorder == ">":
        array = array.byteswap().newbyteorder()

    return ImageFrame(
        topic=topic,
        bag_stamp=bag_stamp,
        header_stamp=header_stamp,
        frame_id=frame_id,
        encoding=encoding,
        image=array.copy(),
    )


def parse_camera_info(topic, payload):
    _stamp, frame_id, offset = parse_std_header(payload)
    height, offset = read_u32(payload, offset)
    width, offset = read_u32(payload, offset)
    _distortion_model, offset = read_string(payload, offset)
    d_len, offset = read_u32(payload, offset)
    offset += d_len * 8
    k = []
    for _ in range(9):
        value, offset = read_f64(payload, offset)
        k.append(value)
    return CameraInfo(topic=topic, width=width, height=height, k=np.asarray(k, dtype=np.float64), frame_id=frame_id)


def parse_float32(payload):
    value, _offset = read_f32(payload, 0)
    return float(value)


def parse_transform(topic, payload):
    values = struct.unpack_from("<7d", payload, 0)
    return Transform(
        topic=topic,
        translation=np.asarray(values[:3], dtype=np.float64),
        rotation_xyzw=np.asarray(values[3:], dtype=np.float64),
    )


def pick_topic(topics, requested, includes, suffix=None, msg_type=None):
    if requested:
        if requested not in topics:
            raise ValueError(f"Requested topic not found: {requested}")
        return requested

    matches = []
    for topic, topic_type in topics.items():
        if msg_type is not None and topic_type != msg_type:
            continue
        if suffix is not None and not topic.endswith(suffix):
            continue
        lower = topic.lower()
        if all(term.lower() in lower for term in includes):
            matches.append(topic)

    if not matches:
        return None
    return sorted(matches, key=len)[0]


def discover_topics(bag_path, args):
    topics = {}
    for topic, msg_type, _payload, _stamp in iter_bag_messages(bag_path):
        topics.setdefault(topic, msg_type)
        has_depth = any("depth" in name.lower() and name.endswith("/image/data") for name in topics)
        has_color = any("color" in name.lower() and name.endswith("/image/data") for name in topics)
        has_depth_info = any("depth" in name.lower() and name.endswith("/info/camera_info") for name in topics)
        has_color_info = any("color" in name.lower() and name.endswith("/info/camera_info") for name in topics)
        has_depth_units = any("depth_units" in name.lower() and name.endswith("/value") for name in topics)
        has_color_tf = any("color" in name.lower() and "/tf/" in name.lower() for name in topics)
        if has_depth and has_color and has_depth_info and has_color_info and has_depth_units and has_color_tf:
            break

    depth_topic = pick_topic(topics, args.depth_topic, ["depth"], suffix="/image/data", msg_type="sensor_msgs/Image")
    color_topic = pick_topic(topics, args.color_topic, ["color"], suffix="/image/data", msg_type="sensor_msgs/Image")
    depth_info_topic = pick_topic(
        topics,
        args.depth_camera_info_topic,
        ["depth"],
        suffix="/info/camera_info",
        msg_type="sensor_msgs/CameraInfo",
    )
    color_info_topic = pick_topic(
        topics,
        args.color_camera_info_topic,
        ["color"],
        suffix="/info/camera_info",
        msg_type="sensor_msgs/CameraInfo",
    )
    depth_units_topic = pick_topic(
        topics,
        args.depth_units_topic,
        ["depth_units", "value"],
        msg_type="std_msgs/Float32",
    )
    color_tf_topic = pick_topic(topics, args.color_tf_topic, ["color", "tf"], msg_type="geometry_msgs/Transform")

    required = {
        "depth image": depth_topic,
        "color image": color_topic,
        "depth camera info": depth_info_topic,
        "color camera info": color_info_topic,
    }
    missing = [name for name, topic in required.items() if topic is None]
    if missing:
        available = "\n".join(f"  {name} ({msg_type})" for name, msg_type in sorted(topics.items()))
        raise ValueError(f"Could not auto-detect {', '.join(missing)} topics. Available topics:\n{available}")

    return {
        "depth": depth_topic,
        "color": color_topic,
        "depth_info": depth_info_topic,
        "color_info": color_info_topic,
        "depth_units": depth_units_topic,
        "color_tf": color_tf_topic,
    }


def read_bag_inputs(bag_path, selected_topics, frame_index, max_frames=30):
    wanted = {topic for topic in selected_topics.values() if topic is not None}
    color_frames = []
    depth_frames = []
    depth_info = None
    color_info = None
    depth_scale = None
    color_tf = None

    for topic, msg_type, payload, bag_stamp in iter_bag_messages(bag_path, topics=wanted):
        if topic == selected_topics["color"] and msg_type == "sensor_msgs/Image":
            color_frames.append(parse_image(topic, payload, bag_stamp))
        elif topic == selected_topics["depth"] and msg_type == "sensor_msgs/Image":
            depth_frames.append(parse_image(topic, payload, bag_stamp))
        elif topic == selected_topics["depth_info"] and msg_type == "sensor_msgs/CameraInfo":
            depth_info = parse_camera_info(topic, payload)
        elif topic == selected_topics["color_info"] and msg_type == "sensor_msgs/CameraInfo":
            color_info = parse_camera_info(topic, payload)
        elif selected_topics["depth_units"] and topic == selected_topics["depth_units"] and msg_type == "std_msgs/Float32":
            depth_scale = parse_float32(payload)
        elif selected_topics["color_tf"] and topic == selected_topics["color_tf"] and msg_type == "geometry_msgs/Transform":
            color_tf = parse_transform(topic, payload)

        if (
            len(color_frames) > frame_index
            and len(depth_frames) >= min(max_frames, frame_index + 5)
            and depth_info is not None
            and color_info is not None
            and (selected_topics["depth_units"] is None or depth_scale is not None)
            and (selected_topics["color_tf"] is None or color_tf is not None)
        ):
            break

        if len(color_frames) >= max_frames and len(depth_frames) >= max_frames and depth_info and color_info:
            break

    if len(color_frames) <= frame_index:
        raise ValueError(f"Bag has only {len(color_frames)} color frames before the scan stopped")
    if not depth_frames:
        raise ValueError("No depth frames were read from the selected depth topic")
    if depth_info is None or color_info is None:
        raise ValueError("Missing camera info for selected depth/color topics")

    color_frame = sorted(color_frames, key=lambda f: f.header_stamp)[frame_index]
    depth_frame = min(depth_frames, key=lambda f: abs(f.header_stamp - color_frame.header_stamp))

    if depth_scale is None:
        depth_scale = 0.001 if depth_frame.image.dtype.kind in {"u", "i"} else 1.0

    return color_frame, depth_frame, color_info, depth_info, float(depth_scale), color_tf


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


def build_compliance_images(image_shape, coords, edges, rows_by_edge, mode, thickness):
    height, width = image_shape[:2]
    cfg = metric_config(mode)
    values = [metric_value(row, mode) for row in rows_by_edge.values()]
    vmin, vmax = normalize(values)
    color_fn = cfg["color_fn"]

    color_bgr = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    edge_id = np.full((height, width), -1, dtype=np.int32)

    for idx, (parent, child) in enumerate(edges):
        if parent not in coords or child not in coords:
            continue

        row = rows_by_edge.get((parent, child))
        value = metric_value(row, mode) if row is not None else np.nan
        if np.isfinite(value):
            t = (value - vmin) / max(vmax - vmin, 1e-9)
            bgr = color_fn(t)
        else:
            bgr = (180, 180, 180)

        p0 = tuple(np.round(coords[parent]).astype(int))
        p1 = tuple(np.round(coords[child]).astype(int))
        edge_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(edge_mask, p0, p1, 255, thickness, cv2.LINE_AA)
        color_bgr[edge_mask > 0] = bgr
        mask = np.maximum(mask, edge_mask)
        edge_id[edge_mask > 0] = idx

    color_rgb = color_bgr[:, :, ::-1].copy()
    return color_rgb, mask, edge_id, vmin, vmax


def quaternion_to_matrix(q_xyzw):
    q = np.asarray(q_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_depth_to_color(points_depth, color_tf, tf_direction):
    if color_tf is None:
        return None

    rotation = quaternion_to_matrix(color_tf.rotation_xyzw)
    translation = color_tf.translation.astype(np.float64)

    if tf_direction == "depth-to-color":
        return points_depth @ rotation.T + translation
    if tf_direction == "color-to-depth":
        return (points_depth - translation) @ rotation
    raise ValueError(f"Unknown tf direction: {tf_direction}")


def project_points(points, camera_info):
    k = camera_info.k.reshape(3, 3)
    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]
    z = points[:, 2]
    u = fx * (points[:, 0] / z) + cx
    v = fy * (points[:, 1] / z) + cy
    return u, v


def decode_depth_meters(depth_image, encoding, depth_scale):
    if depth_image.dtype.kind in {"u", "i"}:
        return depth_image.astype(np.float32) * float(depth_scale)
    return depth_image.astype(np.float32) * float(depth_scale)


def sample_rgb(image_rgb, u, v, fallback=(145, 145, 145)):
    h, w = image_rgb.shape[:2]
    colors = np.empty((u.shape[0], 3), dtype=np.uint8)
    colors[:] = np.asarray(fallback, dtype=np.uint8)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    colors[valid] = image_rgb[vi[valid], ui[valid]]
    return colors, ui, vi, valid


def orient_points_for_output(points, coordinate_frame):
    points_out = points.copy()
    if coordinate_frame == "camera":
        return points_out
    if coordinate_frame == "legacy":
        points_out[:, 1] *= -1.0
        return points_out
    if coordinate_frame == "right-handed":
        points_out[:, 0] *= -1.0
        points_out[:, 1] *= -1.0
        return points_out
    raise ValueError(f"Unknown coordinate frame: {coordinate_frame}")


def make_compliance_pointcloud(
    depth_frame,
    color_frame,
    depth_info,
    color_info,
    color_tf,
    depth_scale,
    compliance_rgb,
    compliance_mask,
    stride,
    min_depth,
    max_depth,
    alignment,
    tf_direction,
    background,
    max_points,
    coordinate_frame,
):
    depth_m = decode_depth_meters(depth_frame.image, depth_frame.encoding, depth_scale)
    depth_h, depth_w = depth_m.shape[:2]
    ys = np.arange(0, depth_h, stride, dtype=np.int32)
    xs = np.arange(0, depth_w, stride, dtype=np.int32)
    uu, vv = np.meshgrid(xs, ys)
    z = depth_m[vv, uu]

    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    uu = uu[valid].astype(np.float32)
    vv = vv[valid].astype(np.float32)
    z = z[valid].astype(np.float32)

    k_depth = depth_info.k.reshape(3, 3)
    x = (uu - k_depth[0, 2]) * z / k_depth[0, 0]
    y = (vv - k_depth[1, 2]) * z / k_depth[1, 1]
    points_depth = np.column_stack([x, y, z]).astype(np.float32)

    if alignment == "extrinsics" and color_tf is not None:
        points_for_color = transform_depth_to_color(points_depth.astype(np.float64), color_tf, tf_direction)
        positive_z = points_for_color[:, 2] > 1e-6
        u_color = np.full(points_depth.shape[0], -1.0, dtype=np.float32)
        v_color = np.full(points_depth.shape[0], -1.0, dtype=np.float32)
        projected_u, projected_v = project_points(points_for_color[positive_z], color_info)
        u_color[positive_z] = projected_u.astype(np.float32)
        v_color[positive_z] = projected_v.astype(np.float32)
    else:
        color_h, color_w = color_frame.image.shape[:2]
        u_color = uu * (float(color_w) / float(depth_w))
        v_color = vv * (float(color_h) / float(depth_h))

    if background == "rgb":
        colors, ui, vi, projection_valid = sample_rgb(color_frame.image, u_color, v_color)
    elif background == "black":
        colors = np.zeros((points_depth.shape[0], 3), dtype=np.uint8)
        _unused, ui, vi, projection_valid = sample_rgb(color_frame.image, u_color, v_color)
    else:
        colors = np.full((points_depth.shape[0], 3), 145, dtype=np.uint8)
        _unused, ui, vi, projection_valid = sample_rgb(color_frame.image, u_color, v_color)

    compliance_hits = np.zeros(points_depth.shape[0], dtype=bool)
    in_compliance_image = projection_valid.copy()
    compliance_hits[in_compliance_image] = compliance_mask[vi[in_compliance_image], ui[in_compliance_image]] > 0
    colors[compliance_hits] = compliance_rgb[vi[compliance_hits], ui[compliance_hits]]

    if background == "hide":
        keep = compliance_hits
    else:
        keep = np.ones(points_depth.shape[0], dtype=bool)

    points_out = orient_points_for_output(points_depth, coordinate_frame)
    points_out = points_out[keep]
    colors_out = colors[keep]
    compliance_out = compliance_hits[keep]

    if max_points is not None and points_out.shape[0] > max_points:
        rng = np.random.default_rng(0)
        branch_idx = np.flatnonzero(compliance_out)
        background_idx = np.flatnonzero(~compliance_out)
        if branch_idx.size >= max_points:
            keep_idx = np.sort(rng.choice(branch_idx, size=max_points, replace=False))
        else:
            remaining = max_points - branch_idx.size
            sampled_bg = rng.choice(background_idx, size=remaining, replace=False) if remaining > 0 else np.array([], dtype=int)
            keep_idx = np.sort(np.concatenate([branch_idx, sampled_bg]))
        points_out = points_out[keep_idx]
        colors_out = colors_out[keep_idx]
        compliance_out = compliance_out[keep_idx]

    stats = {
        "valid_depth_points": int(points_depth.shape[0]),
        "written_points": int(points_out.shape[0]),
        "compliance_points": int(np.count_nonzero(compliance_out)),
        "alignment": alignment if color_tf is not None else "resize",
        "coordinate_frame": coordinate_frame,
    }
    return points_out, colors_out, stats


def write_binary_ply(path, points, colors):
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0].astype(np.float32)
    vertices["y"] = points[:, 1].astype(np.float32)
    vertices["z"] = points[:, 2].astype(np.float32)
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertices.tofile(f)


def bounds_for_html(points):
    if points.size == 0:
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0
    else:
        pmin = points.min(axis=0)
        pmax = points.max(axis=0)
        center = 0.5 * (pmin + pmax)
        radius = float(np.linalg.norm(pmax - pmin) * 0.55)
        radius = max(radius, 0.1)
    return center.tolist(), radius


def write_html_viewer(path, points, colors, title, metadata):
    points_f32 = np.ascontiguousarray(points.astype("<f4"))
    colors_u8 = np.ascontiguousarray(colors.astype(np.uint8))
    points_b64 = base64.b64encode(points_f32.tobytes()).decode("ascii")
    colors_b64 = base64.b64encode(colors_u8.tobytes()).decode("ascii")
    center, radius = bounds_for_html(points_f32)
    metadata_lines = [f"{key}: {value}" for key, value in metadata.items()]
    metadata_html = html.escape(" | ".join(metadata_lines))
    title_html = html.escape(title)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_html}</title>
<style>
  html, body {{
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #ffffff;
    color: #1f2328;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  canvas {{
    display: block;
    width: 100vw;
    height: 100vh;
    cursor: grab;
  }}
  canvas:active {{
    cursor: grabbing;
  }}
  .hud {{
    position: fixed;
    left: 14px;
    top: 12px;
    max-width: min(720px, calc(100vw - 28px));
    padding: 10px 12px;
    border: 1px solid rgba(31, 35, 40, 0.16);
    background: rgba(255, 255, 255, 0.78);
    backdrop-filter: blur(10px);
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(31, 35, 40, 0.12);
    pointer-events: none;
  }}
  .title {{
    font-size: 14px;
    font-weight: 650;
    line-height: 1.25;
  }}
  .meta {{
    margin-top: 4px;
    color: #4b5563;
    font-size: 12px;
    line-height: 1.35;
  }}
</style>
</head>
<body>
<canvas id="view"></canvas>
<div class="hud">
  <div class="title">{title_html}</div>
  <div class="meta">{metadata_html}</div>
</div>
<script>
const POINT_COUNT = {points_f32.shape[0]};
const POINTS_B64 = "{points_b64}";
const COLORS_B64 = "{colors_b64}";
const CENTER = new Float32Array({center});
const RADIUS = {radius:.8f};

function decodeBase64(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}}

const positions = new Float32Array(decodeBase64(POINTS_B64));
const colors = new Uint8Array(decodeBase64(COLORS_B64));
const canvas = document.getElementById("view");
const gl = canvas.getContext("webgl", {{ antialias: true, alpha: false }});
if (!gl) {{
  document.body.innerHTML = "<p style='padding:24px'>WebGL is not available in this browser.</p>";
}}

const vertexSource = `
attribute vec3 aPosition;
attribute vec3 aColor;
uniform mat4 uViewProj;
uniform float uPointSize;
varying vec3 vColor;
void main() {{
  gl_Position = uViewProj * vec4(aPosition, 1.0);
  gl_PointSize = uPointSize;
  vColor = aColor;
}}
`;
const fragmentSource = `
precision mediump float;
varying vec3 vColor;
void main() {{
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;
  gl_FragColor = vec4(vColor, 1.0);
}}
`;

function compileShader(type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
    throw new Error(gl.getShaderInfoLog(shader));
  }}
  return shader;
}}

const program = gl.createProgram();
gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSource));
gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSource));
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
  throw new Error(gl.getProgramInfoLog(program));
}}
gl.useProgram(program);

const positionBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
const positionLocation = gl.getAttribLocation(program, "aPosition");
gl.enableVertexAttribArray(positionLocation);
gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);

const colorBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
const colorLocation = gl.getAttribLocation(program, "aColor");
gl.enableVertexAttribArray(colorLocation);
gl.vertexAttribPointer(colorLocation, 3, gl.UNSIGNED_BYTE, true, 0, 0);

const uViewProj = gl.getUniformLocation(program, "uViewProj");
const uPointSize = gl.getUniformLocation(program, "uPointSize");

let yaw = 0.0;
let pitch = 0.22;
let distance = Math.max(RADIUS * 2.6, 0.4);
let target = new Float32Array(CENTER);
let dragging = false;
let panning = false;
let lastX = 0;
let lastY = 0;

function normalize(v) {{
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}}

function cross(a, b) {{
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}}

function dot(a, b) {{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}}

function lookAt(eye, center, up) {{
  const z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0,
  ]);
}}

function multiply(a, b) {{
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++) {{
    for (let r = 0; r < 4; r++) {{
      out[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }}
  }}
  return out;
}}

function resize() {{
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  gl.viewport(0, 0, width, height);
}}

function render() {{
  resize();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(1.0, 1.0, 1.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const cp = Math.cos(pitch);
  const eye = [
    target[0] + Math.sin(yaw) * cp * distance,
    target[1] + Math.sin(pitch) * distance,
    target[2] + Math.cos(yaw) * cp * distance,
  ];
  const view = lookAt(eye, target, [0, 1, 0]);
  const proj = perspective(Math.PI / 4, canvas.width / Math.max(canvas.height, 1), Math.max(distance / 500, 0.001), distance + RADIUS * 8 + 1);
  gl.uniformMatrix4fv(uViewProj, false, multiply(proj, view));
  gl.uniform1f(uPointSize, Math.max(1.5, Math.min(7.0, 3.2 * (window.devicePixelRatio || 1))));
  gl.drawArrays(gl.POINTS, 0, POINT_COUNT);
}}

canvas.addEventListener("pointerdown", (event) => {{
  dragging = true;
  panning = event.button === 2 || event.shiftKey;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
}});
canvas.addEventListener("pointermove", (event) => {{
  if (!dragging) return;
  const dx = event.clientX - lastX;
  const dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;
  if (panning) {{
    const scale = distance * 0.0015;
    const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
    const up = [0, 1, 0];
    target[0] -= right[0] * dx * scale - up[0] * dy * scale;
    target[1] -= right[1] * dx * scale - up[1] * dy * scale;
    target[2] -= right[2] * dx * scale - up[2] * dy * scale;
  }} else {{
    yaw -= dx * 0.006;
    pitch = Math.max(-1.45, Math.min(1.45, pitch - dy * 0.006));
  }}
  render();
}});
canvas.addEventListener("pointerup", (event) => {{
  dragging = false;
  canvas.releasePointerCapture(event.pointerId);
}});
canvas.addEventListener("wheel", (event) => {{
  event.preventDefault();
  distance *= Math.exp(event.deltaY * 0.001);
  distance = Math.max(0.03, Math.min(distance, RADIUS * 30 + 2));
  render();
}}, {{ passive: false }});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
window.addEventListener("resize", render);
render();
</script>
</body>
</html>
"""
    path.write_text(document)


def make_parser():
    parser = argparse.ArgumentParser(
        description="Create a depth point cloud colored by first-frame relative compliance heatmap values."
    )
    parser.add_argument("analysis_dir", help="Directory with compliance_summary.csv, analysis_tracks.csv, graph_spec_used.json")
    parser.add_argument("bag_path", help="ROS bag containing RealSense color/depth image streams")
    parser.add_argument("--out-dir", default=None, help="Output directory; defaults to the analysis directory")
    parser.add_argument("--mode", default="compliance", choices=[
        "compliance",
        "uncertainty",
        "compliance_per_length",
        "compliance_per_uncertainty",
        "compliance_per_length_variation",
    ])
    parser.add_argument("--frame-index", type=int, default=0, help="Color frame index to use from the bag")
    parser.add_argument("--edge-thickness", type=int, default=16, help="2D compliance line thickness in color pixels")
    parser.add_argument("--stride", type=int, default=2, help="Depth pixel stride for point cloud generation")
    parser.add_argument("--max-points", type=int, default=180000, help="Maximum points to write after sampling")
    parser.add_argument("--min-depth", type=float, default=0.15, help="Minimum depth in meters")
    parser.add_argument("--max-depth", type=float, default=4.0, help="Maximum depth in meters")
    parser.add_argument("--background", choices=["rgb", "gray", "black", "hide"], default="rgb")
    parser.add_argument("--alignment", choices=["extrinsics", "resize"], default="extrinsics")
    parser.add_argument("--tf-direction", choices=["color-to-depth", "depth-to-color"], default="color-to-depth")
    parser.add_argument(
        "--coordinate-frame",
        choices=["right-handed", "camera", "legacy"],
        default="right-handed",
        help=(
            "Point coordinate convention. right-handed mirrors X and flips Y for a natural browser/OpenGL view; "
            "camera keeps raw camera coordinates; legacy keeps the previous X-right/Y-up output."
        ),
    )
    parser.add_argument("--track-image-size", type=parse_size, default=None, help="Original track coordinate size, e.g. 640x480")
    parser.add_argument("--depth-scale", type=float, default=None, help="Override depth scale for integer depth images")
    parser.add_argument("--color-topic", default=None)
    parser.add_argument("--depth-topic", default=None)
    parser.add_argument("--color-camera-info-topic", default=None)
    parser.add_argument("--depth-camera-info-topic", default=None)
    parser.add_argument("--depth-units-topic", default=None)
    parser.add_argument("--color-tf-topic", default=None)
    return parser


def main():
    args = make_parser().parse_args()
    analysis_dir = Path(args.analysis_dir)
    bag_path = Path(args.bag_path)
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    graph_json = analysis_dir / "graph_spec_used.json"
    tracks_csv = analysis_dir / "analysis_tracks.csv"
    compliance_csv = analysis_dir / "compliance_summary.csv"

    _root_id, edges = load_graph(graph_json)
    coords = load_first_frame_coords(tracks_csv)
    rows = load_compliance_rows(compliance_csv)
    rows_by_edge = {(row["parent"], row["child"]): row for row in rows}

    selected = discover_topics(bag_path, args)
    color_frame, depth_frame, color_info, depth_info, depth_scale, color_tf = read_bag_inputs(
        bag_path, selected, frame_index=args.frame_index
    )
    if args.depth_scale is not None:
        depth_scale = args.depth_scale

    coords = scaled_coords(coords, args.track_image_size, color_frame.image.shape[1::-1])
    compliance_rgb, compliance_mask, _edge_id, vmin, vmax = build_compliance_images(
        color_frame.image.shape,
        coords,
        edges,
        rows_by_edge,
        mode=args.mode,
        thickness=args.edge_thickness,
    )

    points, colors, stats = make_compliance_pointcloud(
        depth_frame=depth_frame,
        color_frame=color_frame,
        depth_info=depth_info,
        color_info=color_info,
        color_tf=color_tf,
        depth_scale=depth_scale,
        compliance_rgb=compliance_rgb,
        compliance_mask=compliance_mask,
        stride=max(args.stride, 1),
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        alignment=args.alignment,
        tf_direction=args.tf_direction,
        background=args.background,
        max_points=args.max_points,
        coordinate_frame=args.coordinate_frame,
    )

    stem = f"relative_{args.mode}_pointcloud_{stats['alignment']}_frame_{args.frame_index:04d}"
    if args.background == "hide":
        stem += "_branch_only"
    ply_path = out_dir / f"{stem}.ply"
    html_path = out_dir / f"{stem}.html"
    write_binary_ply(ply_path, points, colors)

    cfg = metric_config(args.mode)
    metadata = {
        "points": stats["written_points"],
        "heatmap points": stats["compliance_points"],
        "depth frame": f"{depth_frame.image.shape[1]}x{depth_frame.image.shape[0]}",
        "color frame": f"{color_frame.image.shape[1]}x{color_frame.image.shape[0]}",
        "metric range": f"{vmin:.3g} to {vmax:.3g}",
        "alignment": stats["alignment"],
        "coordinate frame": stats["coordinate_frame"],
    }
    write_html_viewer(html_path, points, colors, cfg["title"], metadata)

    print(f"Color topic: {selected['color']}")
    print(f"Depth topic: {selected['depth']}")
    if selected["color_tf"]:
        print(f"Color transform topic: {selected['color_tf']} ({args.tf_direction})")
    print(f"Depth scale: {depth_scale:g} m/unit")
    print(f"Color/depth timestamp delta: {abs(color_frame.header_stamp - depth_frame.header_stamp):.6f} s")
    print(f"Valid depth points before final sampling: {stats['valid_depth_points']}")
    print(f"Compliance-colored points written: {stats['compliance_points']}")
    print(f"Total points written: {stats['written_points']}")
    print(f"PLY: {ply_path}")
    print(f"Viewer: {html_path}")


if __name__ == "__main__":
    main()
