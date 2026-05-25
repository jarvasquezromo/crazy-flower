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
