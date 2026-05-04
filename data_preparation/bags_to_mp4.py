#!/usr/bin/env python3
"""
Convert all ROS1 bag files in a directory to MP4 videos.
Uses rosbags (pure Python, no ROS installation needed).
Extracts the RealSense color stream from /device_0/sensor_1/Color_0/image/data.
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader

COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"


def parse_image(rawdata: bytes) -> np.ndarray | None:
    """Parse a raw sensor_msgs/Image payload into an BGR numpy array."""
    try:
        offset = 0
        # Header: seq(4), stamp secs(4)+nsecs(4), frame_id string(4+n)
        offset += 4 + 4 + 4
        fid_len = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        offset += fid_len
        # Image fields
        height = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        width  = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        enc_len = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        encoding = rawdata[offset:offset + enc_len].decode(); offset += enc_len
        offset += 1  # is_bigendian
        offset += 4  # step
        data_len = struct.unpack_from("<I", rawdata, offset)[0]; offset += 4
        pixels = np.frombuffer(rawdata[offset:offset + data_len], dtype=np.uint8)
        img = pixels.reshape(height, width, -1)
        if encoding == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    except Exception as e:
        print(f"  Warning: could not parse frame ({e})", file=sys.stderr)
        return None


def bag_to_mp4(bag_path: Path, out_path: Path) -> int:
    """Extract color frames from bag_path and write to out_path. Returns frame count."""
    print(f"Reading {bag_path.name} ...")
    writer = None
    frame_count = 0

    with Reader(str(bag_path)) as reader:
        conns = [c for c in reader.connections if c.topic == COLOR_TOPIC]
        if not conns:
            print(f"  No color topic found in {bag_path.name}, skipping.")
            return 0

        for _conn, _ts, rawdata in reader.messages(connections=conns):
            img = parse_image(rawdata)
            if img is None:
                continue
            if writer is None:
                h, w = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))
                print(f"  Resolution {w}x{h}, writing to {out_path.name}")
            writer.write(img)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  {frame_count} frames processed...")

    if writer is not None:
        writer.release()
    return frame_count


def main():
    repo_root = Path(__file__).resolve().parent.parent
    bags_dir  = repo_root / "Data" / "Bags"
    out_dir   = repo_root / "Data"

    bag_files = sorted(bags_dir.glob("*.bag"))
    if not bag_files:
        print(f"No .bag files found in {bags_dir}")
        sys.exit(1)

    print(f"Found {len(bag_files)} bag file(s) in {bags_dir}\n")
    for bag_path in bag_files:
        out_path = out_dir / f"{bag_path.stem}_video.mp4"
        if out_path.exists():
            print(f"Skipping {bag_path.name} (output already exists: {out_path.name})")
            continue
        n = bag_to_mp4(bag_path, out_path)
        if n:
            print(f"  Done — {n} frames -> {out_path}\n")
        else:
            print()


if __name__ == "__main__":
    main()
