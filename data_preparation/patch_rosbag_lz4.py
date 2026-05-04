#!/usr/bin/env python3
"""
Patch rosbag to add lz4 support by monkey-patching the decompression.
"""

import lz4.frame
import rosbag
from rosbag import bag

# Monkey-patch rosbag to support lz4
def patched_decompress(self, datastr, compression_type):
    """Override decompression to support lz4"""
    if compression_type == rosbag.Compression.LZ4:
        try:
            return lz4.frame.decompress(datastr)
        except Exception as e:
            raise rosbag.bag.ROSBagException(f"Failed to decompress lz4 data: {e}")
    # Fall back to original for other compression types
    return self._original_decompress(datastr, compression_type)

# Apply the monkey patch
if not hasattr(bag.Bag, '_original_decompress'):
    bag.Bag._original_decompress = bag.Bag._decompress
    bag.Bag._decompress = patched_decompress

print("Rosbag lz4 support patched!")
