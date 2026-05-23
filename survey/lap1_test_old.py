#!/usr/bin/env python3
import contextlib
import logging
import os
import socket
import struct
import sys
import threading
import warnings
import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from PyQt6 import QtCore, QtWidgets, QtGui
import cv2
import time

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore', message='.*TYPE_HOVER_LEGACY.*')
warnings.filterwarnings('ignore', message='.*supervisor subsystem requires CRTP.*')

URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E705')
AIDECK_IP = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b'FER'
SPEED = 0.6

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
IMG_WIDTH = 324
IMG_HEIGHT = 244
MIN_JPEG_BYTES = 5000

# --- Gate detection / control tuning ---
# HSV ranges for "green" can vary a lot with exposure/white balance.
# Start with this, then adjust if needed.
GREEN_HSV_LO = np.array([35, 50, 50], dtype=np.uint8)
GREEN_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)
GREEN_MIN_V = 240            # set lower (e.g. 120) if detection is too strict

MIN_GREEN_AREA_FRAC = 0.01   # fraction of image area
CENTER_TOL_X = 0.10          # normalized (0..1) horizontal tolerance
CENTER_TOL_Y = 0.12          # normalized (0..1) vertical tolerance

SEARCH_YAWRATE = -20.0       # deg/s, negative chosen as "turn left"
MAX_YAWRATE = 70.0           # deg/s
K_YAW = 80.0                 # deg/s per normalized x error

FORWARD_SPEED = 0.35         # m/s in body X, during push-through
PUSH_DURATION_S = 1.0
DEFAULT_HEIGHT = 0.8         # meters
MAX_GATES = 4

MIN_HEIGHT = 0.2             # meters (safety clamp)
MAX_HEIGHT = 2.0             # meters (safety clamp)
K_HEIGHT = 1.2               # (m/s) per normalized vertical error
MAX_DH_PER_S = 0.6           # max height change rate

MORPH_KERNEL = np.ones((5, 5), np.uint8)


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


def _detect_green_gate(rgb_img):
    """Return detection dict {found, cx, cy, area, bbox, ex, ey} from an RGB image."""
    """
    It works this way : 
    1. Convert the RGB image to HSV color space.
    2. Create a binary mask where the green pixels are white and the rest are black
    3. Apply morphological operations to clean up the mask.
    4. Find contours in the mask and select the largest one as the detected gate.
    5. Calculate the center of the detected gate and the error from the image center.
    6. Return a dictionary with the detection results.
    """
    
    h, w = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)
    if GREEN_MIN_V is not None and GREEN_MIN_V > 0:
        v = hsv[:, :, 2]
        v_mask = np.where(v >= GREEN_MIN_V, np.uint8(255), np.uint8(0))
        mask = cv2.bitwise_and(mask, v_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "mask": mask}

    img_area = float(h * w)
    best = None
    best_area = 0.0
    for cnt in contours:
        a = float(cv2.contourArea(cnt))
        if a > best_area:
            best_area = a
            best = cnt

    if best is None or best_area < (MIN_GREEN_AREA_FRAC * img_area):
        return {"found": False, "mask": mask}

    x, y, bw, bh = cv2.boundingRect(best)
    m = cv2.moments(best)
    if abs(m.get("m00", 0.0)) < 1e-6:
        cx = x + 0.5 * bw
        cy = y + 0.5 * bh
    else:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

    ex = (cx - 0.5 * w) / max(0.5 * w, 1.0)
    ey = (cy - 0.5 * h) / max(0.5 * h, 1.0)

    return {
        "found": True,
        "cx": cx,
        "cy": cy,
        "area": best_area,
        "bbox": (x, y, bw, bh),
        "ex": ex,
        "ey": ey,
        "mask": mask,
    }


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                self._decode_and_emit(buffer)
                receiving = False

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
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(img)


class FPVWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crazyflie FPV')

        self.image_label = QtWidgets.QLabel()
        self.status_label = QtWidgets.QLabel('Connecting...')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': 0.0}

        # Simple autonomy state machine.
        self._gate_state = "SEARCH"   # SEARCH -> CENTER -> PUSH
        self._state_t0 = time.monotonic()
        self._gates_passed = 0
        self._vision_lock = threading.Lock()
        self._vision = {"found": False, "ex": 0.0, "ey": 0.0, "bbox": None, "area": 0.0}

        self._last_ctrl_time = time.monotonic()

        # Default height to something safe; keep manual W/S working by incrementing.
        self.hover["height"] = DEFAULT_HEIGHT

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.open_link(URI)

        if not self.cf.link:
            print('Could not connect')
            sys.exit(1)

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._update_image)
        self.video.start()

        self.cf.supervisor.send_arming_request(True)

        self._timer = QtCore.QTimer()
        #self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(100)
        self._timer.start()

    def _update_image(self, img):
        if img.ndim == 2:
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            color = img

        det = _detect_green_gate(color)
        with self._vision_lock:
            self._vision = {
                "found": bool(det.get("found", False)),
                "ex": float(det.get("ex", 0.0)),
                "ey": float(det.get("ey", 0.0)),
                "bbox": det.get("bbox", None),
                "area": float(det.get("area", 0.0)),
            }

        # Debug overlay on the RGB image.
        disp = color.copy()
        if det.get("found", False) and det.get("bbox") is not None:
            x, y, bw, bh = det["bbox"]
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cx, cy = int(round(det["cx"])), int(round(det["cy"]))
            cv2.circle(disp, (cx, cy), 4, (255, 0, 0), -1)

        # Crosshair
        h, w = disp.shape[:2]
        cv2.drawMarker(disp, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(
            disp,
            f"state={self._gate_state} gates={self._gates_passed} found={int(det.get('found', False))}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        h, w, ch = disp.shape
        q = QtGui.QImage(disp.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    def _send_setpoint(self):
        now = time.monotonic()
        dt = float(np.clip(now - self._last_ctrl_time, 0.02, 0.3))
        self._last_ctrl_time = now
        with self._vision_lock:
            found = bool(self._vision.get("found", False))
            ex = float(self._vision.get("ex", 0.0))
            ey = float(self._vision.get("ey", 0.0))
            bbox = self._vision.get("bbox", None)

        # Autonomy: if you touch the keyboard, it will override values via self.hover.
        # This loop only writes x/y/yaw; it keeps height as a setpoint.
        x_cmd = 0.0
        y_cmd = 0.0
        yaw_cmd = 0.0

        if self._gate_state == "DONE":
            return

        if self._gate_state == "SEARCH":
            # Slowly rotate left until we see a sufficiently large green blob.
            yaw_cmd = SEARCH_YAWRATE
            if found and bbox is not None:
                self._gate_state = "CENTER"
                self._state_t0 = now

        elif self._gate_state == "CENTER":
            if not found:
                # Lost it: go back to searching.
                self._gate_state = "SEARCH"
                self._state_t0 = now
            else:
                yaw_cmd = float(np.clip(-K_YAW * ex, -MAX_YAWRATE, MAX_YAWRATE))
                # Optional tiny lateral correction; keep small to avoid oscillations.
                y_cmd = float(np.clip(-0.15 * ex, -0.2, 0.2))

                # Height control: move up/down to center the gate vertically.
                # ey > 0 means gate is below image center -> drone likely too high -> go down.
                dh = float(np.clip(-K_HEIGHT * ey, -MAX_DH_PER_S, MAX_DH_PER_S))
                self.hover["height"] = float(
                    np.clip(self.hover["height"] + dh * dt, MIN_HEIGHT, MAX_HEIGHT)
                )

                centered = (abs(ex) <= CENTER_TOL_X) and (abs(ey) <= CENTER_TOL_Y)
                if centered:
                    self._gate_state = "PUSH"
                    self._state_t0 = now

        elif self._gate_state == "PUSH":
            # Commit forward for a fixed time.
            x_cmd = FORWARD_SPEED
            yaw_cmd = 0.0
            if (now - self._state_t0) >= PUSH_DURATION_S:
                self._gates_passed += 1
                if self._gates_passed >= MAX_GATES:
                    self._gate_state = "DONE"
                    self.cf.commander.send_stop_setpoint()
                    self._timer.stop()
                    return
                self._gate_state = "SEARCH"
                self._state_t0 = now

        # Apply computed commands.
        self.hover["x"] = x_cmd
        self.hover["y"] = y_cmd
        self.hover["yaw"] = yaw_cmd

        self.cf.commander.send_hover_setpoint(
            self.hover['x'], self.hover['y'], self.hover['yaw'], self.hover['height'])

    def _set_hover(self, key, value):
        if key == 'height':
            self.hover[key] += value
        else:
            self.hover[key] = value * SPEED

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Up:       self._set_hover('x',   1)
        if k == QtCore.Qt.Key.Key_Down:     self._set_hover('x',  -1)
        if k == QtCore.Qt.Key.Key_Left:     self._set_hover('y',   1)
        if k == QtCore.Qt.Key.Key_Right:    self._set_hover('y',  -1)
        if k == QtCore.Qt.Key.Key_A:        self._set_hover('yaw', -70)
        if k == QtCore.Qt.Key.Key_D:        self._set_hover('yaw',  70)
        if k == QtCore.Qt.Key.Key_W:        self._set_hover('height',  0.1)
        if k == QtCore.Qt.Key.Key_S:        self._set_hover('height', -0.1)
        if k == QtCore.Qt.Key.Key_Space:
            self._gate_state = "DONE"
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):    self._set_hover('x', 0)
        if k in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):  self._set_hover('y', 0)
        if k in (QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_D):         self._set_hover('yaw', 0)
        if k in (QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_S):         self._set_hover('height', 0)

    def _connected(self, uri):
        self.status_label.setText(f'Connected to {uri}')

    def _disconnected(self, uri):
        print('Disconnected')
        sys.exit(1)

    def closeEvent(self, event):
        self._timer.stop()
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()