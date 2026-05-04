"""
Interactive point picker.

Usage:

    python3 tests/select_points.py path/to/image.png        # writes ./points.json
    python3 tests/select_points.py img.jpg -o /tmp/pts.json

Left‑click to add a point, press `s` to save the current list to the output
JSON, `q` to quit without saving.  If no GUI backend is available a
Matplotlib figure with ginput() is used instead.
"""
import argparse
import json
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

pts = []  # list of (x,y) tuples
win = None
img = None


def _on_mouse(event, x, y, flags, param):
    global pts, img, win
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow(win, img)


def _try_cv_window(name):
    """Return True if a named window was created successfully."""
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(name, _on_mouse)
        return True
    except cv2.error:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Click points on an image and save them to JSON."
    )
    parser.add_argument("image", help="path to an image file")
    parser.add_argument(
        "-o",
        "--output",
        default="points.json",
        help="output JSON file (default: %(default)s)",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        # Try treating it as a video and grabbing the first frame
        cap = cv2.VideoCapture(args.image)
        ret, img = cap.read()
        cap.release()
    if img is None:
        print(f"could not open image '{args.image}'", file=sys.stderr)
        sys.exit(1)

    use_cv = _try_cv_window("select points – left click to add, 's' save, 'q' quit")

    if use_cv:
        win = "select points – left click to add, 's' save, 'q' quit"
        while True:
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
            elif key == ord("q"):
                pts = []
                break
        cv2.destroyAllWindows()
    else:
        # no GUI available – fallback to matplotlib
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(rgb)
        ax.set_title("click points, close window when done")
        pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close(fig)

    # write output if we have any points
    if pts:
        with open(args.output, "w") as f:
            json.dump({"points": [(float(x), float(y)) for x, y in pts]}, f)
        print(f"saved {len(pts)} points to {args.output}")
    else:
        print("no points selected; nothing written")