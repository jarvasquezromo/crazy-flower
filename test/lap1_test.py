#!/usr/bin/env python3
import contextlib
import json
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
from cflib.crazyflie.log import LogConfig
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
MIN_JPEG_BYTES = 5000
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), 'calibration.json')


def _load_calibration():
    with open(CALIBRATION_PATH, 'r', encoding='utf-8') as f:
        calib = json.load(f)
    required = ('img_w', 'img_h', 'fx', 'fy', 'cx', 'cy', 'dist_coeffs')
    missing = [key for key in required if key not in calib]
    if missing:
        raise ValueError(f"Missing calibration keys in {CALIBRATION_PATH}: {missing}")
    return calib


def _scaled_calibration(calib, img_w, img_h):
    sx = float(img_w) / float(calib['img_w'])
    sy = float(img_h) / float(calib['img_h'])
    scaled = dict(calib)
    scaled['img_w'] = int(img_w)
    scaled['img_h'] = int(img_h)
    scaled['fx'] = float(calib['fx']) * sx
    scaled['fy'] = float(calib['fy']) * sy
    scaled['cx'] = float(calib['cx']) * sx
    scaled['cy'] = float(calib['cy']) * sy
    scaled['fov_h_deg'] = float(np.degrees(2.0 * np.arctan(img_w / (2.0 * scaled['fx']))))
    scaled['fov_v_deg'] = float(np.degrees(2.0 * np.arctan(img_h / (2.0 * scaled['fy']))))
    return scaled


BASE_CALIBRATION = _load_calibration()
CAMERA_CALIBRATION = dict(BASE_CALIBRATION)
IMG_WIDTH = int(CAMERA_CALIBRATION['img_w'])
IMG_HEIGHT = int(CAMERA_CALIBRATION['img_h'])
CAMERA_FX = float(CAMERA_CALIBRATION['fx'])
CAMERA_FY = float(CAMERA_CALIBRATION['fy'])
CAMERA_CX = float(CAMERA_CALIBRATION['cx'])
CAMERA_CY = float(CAMERA_CALIBRATION['cy'])
DIST_COEFFS = np.array(CAMERA_CALIBRATION['dist_coeffs'], dtype=np.float64)


def _set_runtime_calibration(img_w, img_h):
    global CAMERA_CALIBRATION, IMG_WIDTH, IMG_HEIGHT, CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, DIST_COEFFS

    if img_w == int(BASE_CALIBRATION['img_w']) and img_h == int(BASE_CALIBRATION['img_h']):
        CAMERA_CALIBRATION = dict(BASE_CALIBRATION)
        print(f"Radio image size OK: {img_w}x{img_h} matches calibration.json")
    else:
        CAMERA_CALIBRATION = _scaled_calibration(BASE_CALIBRATION, img_w, img_h)
        print(
            f"Radio image size {img_w}x{img_h} differs from calibration.json "
            f"{BASE_CALIBRATION['img_w']}x{BASE_CALIBRATION['img_h']}; scaled calibration in memory"
        )

    IMG_WIDTH = int(CAMERA_CALIBRATION['img_w'])
    IMG_HEIGHT = int(CAMERA_CALIBRATION['img_h'])
    CAMERA_FX = float(CAMERA_CALIBRATION['fx'])
    CAMERA_FY = float(CAMERA_CALIBRATION['fy'])
    CAMERA_CX = float(CAMERA_CALIBRATION['cx'])
    CAMERA_CY = float(CAMERA_CALIBRATION['cy'])
    DIST_COEFFS = np.array(CAMERA_CALIBRATION['dist_coeffs'], dtype=np.float64)


def _camera_matrix():
    return np.array(
        [[CAMERA_FX, 0.0, CAMERA_CX], [0.0, CAMERA_FY, CAMERA_CY], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _undistort_image(rgb_img):
    if DIST_COEFFS.size == 0 or np.allclose(DIST_COEFFS, 0.0):
        return rgb_img
    return cv2.undistort(rgb_img, _camera_matrix(), DIST_COEFFS)

# --- Gate detection / control tuning ---
# HSV ranges for "green" can vary a lot with exposure/white balance.
# Start with this, then adjust if needed.
GREEN_HSV_LO = np.array([35, 50, 50], dtype=np.uint8)
GREEN_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)
GREEN_MIN_V = 240            # set lower (e.g. 120) if detection is too strict

MIN_GREEN_AREA_FRAC = 0.01   # fraction of image area
CENTER_TOL_X = 0.10          # normalized (0..1) horizontal tolerance
CENTER_TOL_Y = 0.12          # normalized (0..1) vertical tolerance

SEARCH_YAWRATE = -20.0      # deg/s, negative = turn left (full scan in ~18 s)
MAX_YAWRATE = 70.0           # deg/s
K_YAW = 80.0                 # deg/s per normalized x error

FORWARD_SPEED = 0.35         # m/s in body X, during push-through
PUSH_DURATION_S = 1.0
SEARCH_HEIGHT = 0.8          # meters, target height for search/center
TAKEOFF_START_HEIGHT = 0.1   # meters, initial setpoint at takeoff
TAKEOFF_RATE = 0.4           # m/s climb rate during takeoff ramp
MAX_GATES = 4

MIN_HEIGHT = 0.2             # meters (safety clamp)
MAX_HEIGHT = 2.0             # meters (safety clamp)
K_HEIGHT = 1.2               # (m/s) per normalized vertical error
MAX_DH_PER_S = 0.6           # max height change rate

MORPH_KERNEL = np.ones((9, 9), np.uint8)

# --- Camera / world-frame projection ---
GATE_PHYS_W    = 0.8                 # metres, physical gate width for distance estimate
TRAJ_N_STEPS   = 5                   # number of waypoints in interpolated trajectory
TRAJ_OVERSHOOT = 0.30                # metres past gate centre (to fly through cleanly)
WAYPOINT_TOL   = 0.15                # metres, advance to next waypoint within this radius
PASS_AREA_FRAC = 0.25                # gate bbox / image area threshold → gate passed
CHASE_TIMEOUT  = 8.0                 # seconds before giving up and returning to SEARCH
GATE_EMA_ALPHA = 0.35                # weight for EMA update of locked gate position


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
    1. Convert the RGB image to grayscale.
    2. Create a binary mask where bright pixels are white and the rest are black.
    3. Apply morphological close to clean up the mask.
    4. Find contours in the mask and select the largest one as the detected gate.
    5. Calculate the center of the detected gate and the error from the image center.
    6. Return a dictionary with the detection results.
    """
    
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    threshold = max(int(GREEN_HSV_LO[2]), int(GREEN_MIN_V) if GREEN_MIN_V is not None else 0)
    mask = np.where(gray >= threshold, np.uint8(255), np.uint8(0))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)

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

    ex = (cx - CAMERA_CX) / max(0.5 * w, 1.0)
    ey = (cy - CAMERA_CY) / max(0.5 * h, 1.0)

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expected_w = IMG_WIDTH
        self._expected_h = IMG_HEIGHT
        self._printed_size = False

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
                if 0 < w and 0 < h and 0 < size < 65536:
                    self._expected_w = int(w)
                    self._expected_h = int(h)
                    if not self._printed_size:
                        _set_runtime_calibration(self._expected_w, self._expected_h)
                        self._printed_size = True
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
        if img is None:
            return
        if img.shape[:2] != (self._expected_h, self._expected_w):
            print(
                f"Decoded radio image has wrong size: got {img.shape[1]}x{img.shape[0]}, "
                f"expected {self._expected_w}x{self._expected_h}"
            )
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

        # Commanded world-frame position sent to the flight controller each tick.
        self._pos = {'x': 0.0, 'y': 0.0, 'z': TAKEOFF_START_HEIGHT, 'yaw': 0.0}
        # State estimator readout (updated by log callback at 50 Hz).
        self._est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._pos_lock = threading.Lock()

        # Gate tracking in world frame.
        self._gate_world = None   # (gx, gy, gz) locked estimate
        self._traj       = []     # list of (x, y, z) waypoints
        self._traj_idx   = 0

        # Simple autonomy state machine.
        self._gate_state = "WAIT"  # WAIT -> TAKEOFF -> SEARCH -> CHASE -> (repeat)
        self._log_ready  = False
        self._state_t0 = time.monotonic()
        self._gates_passed = 0
        self._vision_lock = threading.Lock()
        self._vision = {"found": False, "ex": 0.0, "ey": 0.0, "bbox": None, "area": 0.0}

        self._last_ctrl_time = time.monotonic()

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
        color = _undistort_image(color)

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
            ex    = float(self._vision.get("ex", 0.0))
            ey    = float(self._vision.get("ey", 0.0))
            bbox  = self._vision.get("bbox", None)
            area  = float(self._vision.get("area", 0.0))
        with self._pos_lock:
            est = dict(self._est)

        img_area = float(IMG_WIDTH * IMG_HEIGHT)

        if self._gate_state == "DONE":
            return

        if not self._log_ready:
            # Kalman filter not yet converged — hold still and wait.
            return

        if self._gate_state == "WAIT":
            self._gate_state = "TAKEOFF"
            self._state_t0   = now

        if self._gate_state == "TAKEOFF":
            self._pos['z'] = float(min(self._pos['z'] + TAKEOFF_RATE * dt, SEARCH_HEIGHT))
            if self._pos['z'] >= SEARCH_HEIGHT - 1e-3:
                self._gate_state = "SEARCH"
                self._state_t0 = now

        elif self._gate_state == "SEARCH":
            # Yaw in place; x/y/z hold their current commanded values.
            self._pos['yaw'] += SEARCH_YAWRATE * dt
            if found and bbox is not None:
                gw = self._gate_to_world(ex, ey, bbox)
                self._gate_world = gw
                self._traj     = self._build_traj(gw)
                self._traj_idx = 0
                self._gate_state = "CHASE"
                self._state_t0 = now

        elif self._gate_state == "CHASE":
            if (now - self._state_t0) > CHASE_TIMEOUT:
                # Missed the gate — give up and search again.
                self._gate_world = None
                self._traj       = []
                self._traj_idx   = 0
                self._gate_state = "SEARCH"
                self._state_t0   = now
            else:
                # Continuously refine gate world position while visible.
                if found and bbox is not None:
                    new_gw = self._gate_to_world(ex, ey, bbox)
                    ox, oy, oz = self._gate_world
                    nx, ny, nz = new_gw
                    self._gate_world = (
                        ox + GATE_EMA_ALPHA * (nx - ox),
                        oy + GATE_EMA_ALPHA * (ny - oy),
                        oz + GATE_EMA_ALPHA * (nz - oz),
                    )
                    if self._traj_idx == 0:
                        self._traj = self._build_traj(self._gate_world)

                # Gate passed: bbox fills a large fraction of the frame.
                if area / img_area > PASS_AREA_FRAC:
                    self._gates_passed += 1
                    self._gate_world = None
                    self._traj       = []
                    self._traj_idx   = 0
                    if self._gates_passed >= MAX_GATES:
                        self._gate_state = "DONE"
                        self.cf.commander.send_stop_setpoint()
                        self._timer.stop()
                        return
                    self._gate_state = "SEARCH"
                    self._state_t0   = now
                elif self._traj:
                    # Command next waypoint; aim yaw toward locked gate.
                    wp = self._traj[self._traj_idx]
                    self._pos['x'], self._pos['y'], self._pos['z'] = wp
                    gx, gy, _ = self._gate_world
                    self._pos['yaw'] = float(np.degrees(
                        np.arctan2(gy - est['y'], gx - est['x'])))
                    dist_wp = np.hypot(est['x'] - wp[0], est['y'] - wp[1])
                    if dist_wp < WAYPOINT_TOL and self._traj_idx < len(self._traj) - 1:
                        self._traj_idx += 1

        self._pos['z'] = float(np.clip(self._pos['z'], MIN_HEIGHT, MAX_HEIGHT))
        self.cf.commander.send_position_setpoint(
            self._pos['x'], self._pos['y'], self._pos['z'], self._pos['yaw'])

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Up:    self._pos['x'] += 0.2
        if k == QtCore.Qt.Key.Key_Down:  self._pos['x'] -= 0.2
        if k == QtCore.Qt.Key.Key_Left:  self._pos['y'] += 0.2
        if k == QtCore.Qt.Key.Key_Right: self._pos['y'] -= 0.2
        if k == QtCore.Qt.Key.Key_W:     self._pos['z'] += 0.1
        if k == QtCore.Qt.Key.Key_S:     self._pos['z'] -= 0.1
        if k == QtCore.Qt.Key.Key_A:     self._pos['yaw'] -= 15.0
        if k == QtCore.Qt.Key.Key_D:     self._pos['yaw'] += 15.0
        if k == QtCore.Qt.Key.Key_Space:
            self._gate_state = "DONE"
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    def _setup_log(self):
        lc = LogConfig('StateEst', period_in_ms=20)
        lc.add_variable('stateEstimate.x', 'float')
        lc.add_variable('stateEstimate.y', 'float')
        lc.add_variable('stateEstimate.z', 'float')
        lc.add_variable('stabilizer.yaw', 'float')
        self.cf.log.add_config(lc)
        lc.data_received_cb.add_callback(self._on_log)
        lc.error_cb.add_callback(lambda _conf, msg: print('Log error:', msg))
        lc.start()
        self._log_cfg = lc

    def _on_log(self, _ts, data, _lc):
        with self._pos_lock:
            self._est['x']   = data['stateEstimate.x']
            self._est['y']   = data['stateEstimate.y']
            self._est['z']   = data['stateEstimate.z']
            self._est['yaw'] = data['stabilizer.yaw']
            if not self._log_ready:
                # Seed commanded position from first real estimate so the drone
                # holds its current position rather than jumping to world-origin.
                self._pos['x']   = self._est['x']
                self._pos['y']   = self._est['y']
                self._pos['z']   = self._est['z']
                self._pos['yaw'] = self._est['yaw']
                self._log_ready  = True

    def _gate_to_world(self, ex, ey, bbox):
        """Project image-plane gate centre + bbox width to a world-frame point."""
        bw = bbox[2]
        # Pinhole distance estimate: dist = (physical_width * focal_px) / bbox_px
        dist = max((GATE_PHYS_W * CAMERA_FX) / max(bw, 1), 0.3)
        x_err_px = ex * max(0.5 * IMG_WIDTH, 1.0)
        y_err_px = ey * max(0.5 * IMG_HEIGHT, 1.0)
        # Body-frame offsets (camera looks along +body_x)
        dx_b =  dist
        dy_b = -x_err_px * dist / CAMERA_FX
        dz_b = -y_err_px * dist / CAMERA_FY
        with self._pos_lock:
            yaw_r = np.deg2rad(self._est['yaw'])
            ox, oy, oz = self._est['x'], self._est['y'], self._est['z']
        gx = ox + dx_b * np.cos(yaw_r) - dy_b * np.sin(yaw_r)
        gy = oy + dx_b * np.sin(yaw_r) + dy_b * np.cos(yaw_r)
        gz = float(np.clip(oz + dz_b, MIN_HEIGHT, MAX_HEIGHT))
        return gx, gy, gz

    def _build_traj(self, gate):
        """Linear trajectory from current drone position to gate + overshoot."""
        with self._pos_lock:
            x0, y0, z0 = self._est['x'], self._est['y'], self._est['z']
        gx, gy, gz = gate
        dx, dy = gx - x0, gy - y0
        mag = np.hypot(dx, dy) + 1e-6
        end_x = gx + TRAJ_OVERSHOOT * dx / mag
        end_y = gy + TRAJ_OVERSHOOT * dy / mag
        ts = np.linspace(0.0, 1.0, TRAJ_N_STEPS + 1)[1:]
        return [(x0 + (end_x - x0) * t,
                 y0 + (end_y - y0) * t,
                 z0 + (gz    - z0) * t) for t in ts]

    def _set_status(self, text):
        """Thread-safe status label update."""
        QtCore.QMetaObject.invokeMethod(
            self.status_label, 'setText',
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text))

    def _connected(self, uri):
        self._set_status(f'Connected to {uri} — resetting Kalman filter…')
        # Reset the Kalman filter so the Lighthouse geometry is used as the
        # reference frame from a clean state, then wait for it to converge.
        try:
            self.cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            self.cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(1.5)
        except Exception as e:
            print(f'Kalman reset failed: {e}')
        self._set_status(f'Connected to {uri}')
        self._setup_log()

    def _disconnected(self, uri):
        print('Disconnected')
        sys.exit(1)

    def closeEvent(self, event):
        self._timer.stop()
        if hasattr(self, '_log_cfg'):
            self._log_cfg.stop()
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()
