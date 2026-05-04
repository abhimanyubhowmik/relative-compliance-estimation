"""
Pure Python ROS bag file reader - no rosbag dependency needed.
Reads images from ROS bag files without requiring Python 3.9 or rosbag library.
"""

import io
import struct
import tarfile
import zlib
from pathlib import Path
from collections import defaultdict
import numpy as np


class BagMessage:
    """Minimal ROS message representation"""
    def __init__(self, topic, data, timestamp, msg_type):
        self.topic = topic
        self.data = data
        self.timestamp = timestamp
        self.type = msg_type


class ImageMessage:
    """Mimics ROS sensor_msgs/Image message"""
    def __init__(self, data_bytes, height, width, encoding):
        self.data = data_bytes
        self.height = height
        self.width = width
        self.encoding = encoding


def read_string(f):
    """Read ROS string field (4-byte length + data)"""
    length_bytes = f.read(4)
    if len(length_bytes) < 4:
        return None
    length = struct.unpack('<I', length_bytes)[0]
    if length > 1e7:  # sanity check: >100MB strings are invalid
        return None
    return f.read(length)


def read_uint32(f):
    """Read ROS uint32"""
    data = f.read(4)
    if len(data) < 4:
        return None
    return struct.unpack('<I', data)[0]


def read_uint64(f):
    """Read ROS uint64"""
    data = f.read(8)
    if len(data) < 8:
        return None
    return struct.unpack('<Q', data)[0]


def parse_image_message(data_bytes, msg_type):
    """
    Parse a sensor_msgs/Image message from raw bytes.
    Returns ImageMessage object or None on error.
    """
    try:
        f = io.BytesIO(data_bytes)
        
        # Header
        seq = read_uint32(f)  # sequence
        if seq is None:
            return None
            
        # timestamp (2x uint32: secs, nsecs)
        secs = read_uint32(f)
        nsecs = read_uint32(f)
        
        # frame_id (string)
        frame_id = read_string(f)
        
        # height, width
        height = read_uint32(f)
        width = read_uint32(f)
        
        if height is None or width is None or height == 0 or width == 0:
            return None
            
        # encoding (string)
        encoding_bytes = read_string(f)
        encoding = encoding_bytes.decode('utf-8', errors='ignore') if encoding_bytes else 'rgb8'
        
        # is_bigendian
        is_bigendian_byte = f.read(1)
        if len(is_bigendian_byte) < 1:
            return None
        is_bigendian = struct.unpack('B', is_bigendian_byte)[0]
        
        # step
        step = read_uint32(f)
        
        # data (byte array)
        data_len_bytes = f.read(4)
        if len(data_len_bytes) < 4:
            return None
        data_len = struct.unpack('<I', data_len_bytes)[0]
        image_data = f.read(data_len)
        
        if len(image_data) < data_len:
            return None
            
        return ImageMessage(image_data, height, width, encoding)
    except Exception as e:
        print(f"Error parsing image message: {e}")
        return None


def read_bag(bag_path):
    """
    Generator that yields (topic, ImageMessage, timestamp) tuples from a bag file.
    Works with Python 3.10+ without rosbag library.
    """
    bag_path = Path(bag_path)
    
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    
    try:
        with tarfile.open(bag_path, 'r:*') as tar:
            # Extract all chunk files
            chunks = {}
            for member in tar.getmembers():
                if member.name.endswith('.bag_index'):
                    continue
                if 'chunk' in member.name:
                    chunk_file = tar.extractfile(member)
                    if chunk_file:
                        chunks[member.name] = chunk_file.read()
            
            # Try to read index if available
            index_data = None
            for member in tar.getmembers():
                if member.name.endswith('.bag_index'):
                    index_file = tar.extractfile(member)
                    if index_file:
                        index_data = index_file.read()
                        break
            
            # Parse chunks for messages
            for chunk_name, chunk_data in chunks.items():
                try:
                    f = io.BytesIO(chunk_data)
                    
                    while True:
                        # Read message type field (4 bytes length + data)
                        length_bytes = f.read(4)
                        if len(length_bytes) < 4:
                            break
                            
                        msg_length = struct.unpack('<I', length_bytes)[0]
                        if msg_length > 1e7:  # sanity check
                            break
                            
                        msg_data = f.read(msg_length)
                        if len(msg_data) < msg_length:
                            break
                        
                        # Parse connection header (ROS bag message format)
                        header_f = io.BytesIO(msg_data)
                        
                        # Topic name
                        topic_len_bytes = header_f.read(4)
                        if len(topic_len_bytes) < 4:
                            continue
                        topic_len = struct.unpack('<I', topic_len_bytes)[0]
                        topic = header_f.read(topic_len).decode('utf-8', errors='ignore')
                        
                        # Timestamp (nanoseconds since epoch)
                        timestamp_bytes = header_f.read(8)
                        if len(timestamp_bytes) < 8:
                            continue
                        timestamp = struct.unpack('<Q', timestamp_bytes)[0]
                        
                        # Message type (string)
                        msg_type_len_bytes = header_f.read(4)
                        if len(msg_type_len_bytes) < 4:
                            continue
                        msg_type_len = struct.unpack('<I', msg_type_len_bytes)[0]
                        msg_type = header_f.read(msg_type_len).decode('utf-8', errors='ignore')
                        
                        # Payload
                        payload = header_f.read()
                        
                        # If it's an image message, parse it
                        if 'Image' in msg_type:
                            img = parse_image_message(payload, msg_type)
                            if img:
                                yield (topic, img, timestamp / 1e9)  # Convert to seconds
                
                except Exception as e:
                    print(f"Error parsing chunk {chunk_name}: {e}")
                    continue
    
    except tarfile.ReadError:
        raise ValueError(f"Invalid ROS bag file: {bag_path}")


def get_image_topics(bag_path):
    """Get list of image topics in a bag file"""
    topics = set()
    for topic, msg, ts in read_bag(bag_path):
        topics.add(topic)
        if len(topics) > 50:  # Avoid reading entire file
            break
    return sorted(topics)
