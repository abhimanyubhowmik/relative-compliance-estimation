#!/usr/bin/env python3
"""
Decompress a rosbag file by reading and rewriting without compression.
This works around lz4 support issues.
"""

import sys
import os

try:
    import rosbag
except ImportError:
    print("rosbag not found, trying to use bagpy...")
    sys.exit(1)

def decompress_bag(input_bag, output_bag=None):
    """Decompress a rosbag file by reading and rewriting it."""
    
    if output_bag is None:
        output_bag = input_bag.replace('.bag', '_uncompressed.bag')
    
    if os.path.exists(output_bag):
        print(f"Output file {output_bag} already exists. Removing...")
        os.remove(output_bag)
    
    print(f"Decompressing {input_bag} to {output_bag}...")
    
    try:
        with rosbag.Bag(input_bag, 'r') as input_file:
            with rosbag.Bag(output_bag, 'w', compression=rosbag.Compression.NONE) as output_file:
                msg_count = 0
                for topic, msg, t in input_file.read_messages():
                    output_file.write(topic, msg, t)
                    msg_count += 1
                    if msg_count % 1000 == 0:
                        print(f"  Processed {msg_count} messages...")
        
        print(f"Done! Decompressed {msg_count} messages.")
        print(f"Output saved to: {output_bag}")
        return output_bag
    
    except Exception as e:
        print(f"Error decompressing bag: {e}")
        if os.path.exists(output_bag):
            os.remove(output_bag)
        raise

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python decompress_bag.py <input.bag> [output.bag]")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2] if len(sys.argv) > 2 else None
    
    decompress_bag(input_bag, output_bag)
