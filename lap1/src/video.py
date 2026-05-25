"""Video sources: UDP AI-deck stream and local video file playback."""
import os
import socket
import struct
import warnings
import contextlib

import cv2
import numpy as np
from PyQt6 import QtCore

from .constants import (
    AIDECK_IP, AIDECK_PORT, LOCAL_PORT, START_MAGIC,
    CPX_HEADER_SIZE, IMG_HEADER_MAGIC, IMG_HEADER_SIZE, MIN_JPEG_BYTES,
)


@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


class UdpVideoThread(QtCore.QThread):
    """Receives JPEG frames from the AI-deck over UDP."""
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    size_detected = QtCore.pyqtSignal(int, int)
    connection_failed = QtCore.pyqtSignal(str)

    def __init__(self, expected_w, expected_h, parent=None):
        super().__init__(parent)
        self._expected_w = expected_w
        self._expected_h = expected_h
        self._size_emitted = False

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        # Wait for first response — if no reply, dongle is not reachable
        sock.settimeout(5.0)
        try:
            data, _ = sock.recvfrom(2048)
        except socket.timeout:
            warnings.warn(
                f"AI-deck dongle not found — no response from "
                f"{AIDECK_IP}:{AIDECK_PORT}. Check Wi-Fi connection.")
            self.connection_failed.emit(
                "AI-deck dongle not found. Check Wi-Fi connection.")
            sock.close()
            return
        sock.settimeout(None)

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            if len(data) >= CPX_HEADER_SIZE:
                payload = data[CPX_HEADER_SIZE:]

                if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                    _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                    if 0 < w and 0 < h and 0 < size < 65536:
                        self._expected_w = int(w)
                        self._expected_h = int(h)
                        if not self._size_emitted:
                            self.size_detected.emit(self._expected_w, self._expected_h)
                            self._size_emitted = True
                        expected_size = size
                        buffer = bytearray()
                        receiving = True
                elif receiving:
                    buffer.extend(payload)
                    if len(buffer) >= expected_size:
                        self._decode_and_emit(buffer)
                        receiving = False

            data, _ = sock.recvfrom(2048)

    def _decode_and_emit(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        if img.shape[:2] != (self._expected_h, self._expected_w):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(img)


class VideoFileThread(QtCore.QThread):
    """Plays back a local video file, emitting frames at ~original FPS."""
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    size_detected = QtCore.pyqtSignal(int, int)

    def __init__(self, path, loop=True, parent=None):
        super().__init__(parent)
        self._path = path
        self._loop = loop

    def run(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            print(f"Cannot open video: {self._path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = int(1000.0 / fps)
        size_emitted = False

        while True:
            ret, frame = cap.read()
            if not ret:
                if self._loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if not size_emitted:
                h, w = frame.shape[:2]
                self.size_detected.emit(w, h)
                size_emitted = True

            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(frame)
            self.msleep(delay_ms)

        cap.release()


class ReplayThread(QtCore.QThread):
    """Replays a recorded session (frames + pose) in sync.

    Emits frames at their original timestamps and pose updates at ~50Hz,
    interpolated from the pose log. The controller receives pose via
    pose_ready and frames via frame_ready, just like the live sources.
    """
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    size_detected = QtCore.pyqtSignal(int, int)
    pose_ready = QtCore.pyqtSignal(float, float, float, float)  # x, y, z, yaw

    POSE_TICK_MS = 20  # 50 Hz pose emission

    def __init__(self, rec_dir, loop=False, parent=None):
        super().__init__(parent)
        self._rec_dir = rec_dir
        self._loop = loop

    def run(self):
        import csv as _csv
        import time as _time

        rec = self._rec_dir

        # Load frames index
        frames = []
        with open(os.path.join(rec, "frames.csv")) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                frames.append((float(row["t"]), row["filename"]))
        if not frames:
            print(f"[Replay] No frames in {rec}")
            return

        # Load pose log
        poses = []
        with open(os.path.join(rec, "pose.csv")) as f:
            reader = _csv.DictReader(f)
            for row in reader:
                poses.append((float(row["t"]), float(row["x"]),
                              float(row["y"]), float(row["z"]), float(row["yaw"])))

        pose_times = np.array([p[0] for p in poses])
        pose_data = np.array([[p[1], p[2], p[3], p[4]] for p in poses])

        # Emit size from first frame
        first_path = os.path.join(rec, "frames", frames[0][1])
        first_img = cv2.imread(first_path)
        if first_img is not None:
            h, w = first_img.shape[:2]
            self.size_detected.emit(w, h)

        total_duration = max(frames[-1][0], pose_times[-1] if len(pose_times) else 0)

        while True:
            t0_real = _time.monotonic()
            frame_idx = 0
            next_frame_t = frames[0][0] if frames else float('inf')

            t_sim = 0.0
            while t_sim <= total_duration:
                # Emit pose (interpolated)
                if len(pose_times) > 0:
                    xyzw = self._interp_pose(t_sim, pose_times, pose_data)
                    self.pose_ready.emit(xyzw[0], xyzw[1], xyzw[2], xyzw[3])

                # Emit frame if its timestamp has arrived
                while frame_idx < len(frames) and t_sim >= frames[frame_idx][0]:
                    fpath = os.path.join(rec, "frames", frames[frame_idx][1])
                    img = cv2.imread(fpath)
                    if img is not None:
                        if img.ndim == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.frame_ready.emit(img)
                    frame_idx += 1

                # Wait real-time tick
                self.msleep(self.POSE_TICK_MS)
                t_sim = _time.monotonic() - t0_real

            if not self._loop:
                break

    @staticmethod
    def _interp_pose(t, times, data):
        """Linearly interpolate pose at time t."""
        if t <= times[0]:
            return data[0]
        if t >= times[-1]:
            return data[-1]
        idx = np.searchsorted(times, t, side='right') - 1
        idx = min(idx, len(times) - 2)
        dt = times[idx + 1] - times[idx]
        if dt < 1e-9:
            return data[idx]
        alpha = (t - times[idx]) / dt
        return data[idx] * (1 - alpha) + data[idx + 1] * alpha
