"""Record and replay synchronized frames + pose."""
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np


class Recorder:
    """Saves timestamped JPEG frames and pose CSV to a dated folder."""

    def __init__(self, base_dir="recordings"):
        self._base_dir = base_dir
        self._recording = False
        self._dir = None
        self._t0 = 0.0
        self._frame_idx = 0
        self._pose_file = None
        self._pose_writer = None

    @property
    def recording(self):
        return self._recording

    def start(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dir = os.path.join(self._base_dir, f"rec_{stamp}")
        os.makedirs(os.path.join(self._dir, "frames"), exist_ok=True)
        self._t0 = time.monotonic()
        self._frame_idx = 0
        # Pose CSV
        self._pose_file = open(os.path.join(self._dir, "pose.csv"), "w", newline="")
        self._pose_writer = csv.writer(self._pose_file)
        self._pose_writer.writerow(["t", "x", "y", "z", "yaw"])
        # Frame index CSV
        self._frame_csv = open(os.path.join(self._dir, "frames.csv"), "w", newline="")
        self._frame_writer = csv.writer(self._frame_csv)
        self._frame_writer.writerow(["t", "filename"])
        self._recording = True
        print(f"[Recorder] Started → {self._dir}")

    def stop(self):
        self._recording = False
        if self._pose_file:
            self._pose_file.close()
            self._pose_file = None
        if hasattr(self, "_frame_csv") and self._frame_csv:
            self._frame_csv.close()
            self._frame_csv = None
        print(f"[Recorder] Stopped. {self._frame_idx} frames saved.")

    def save_frame(self, rgb_img):
        if not self._recording:
            return
        t = time.monotonic() - self._t0
        fname = f"{self._frame_idx:06d}.jpg"
        path = os.path.join(self._dir, "frames", fname)
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        self._frame_writer.writerow([f"{t:.4f}", fname])
        self._frame_idx += 1

    def save_pose(self, x, y, z, yaw):
        if not self._recording:
            return
        t = time.monotonic() - self._t0
        self._pose_writer.writerow([f"{t:.4f}", f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{yaw:.4f}"])
