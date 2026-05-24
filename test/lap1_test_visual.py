#!/usr/bin/env python3
"""Autonomous gate-flying lap for the Crazyflie + AI-deck — *visual control*.

This is a variant of ``lap1_test.py``. It keeps the same gate detection and the
same zone-validation logic (the vision estimate must fall inside the expected
gate's angular zone before the gate is chased), but it does **not** fly to a
world-frame trajectory. Instead the drone is flown straight through the gate by
*visual servoing* on the image errors:

  * yaw rate     <- horizontal centroid error (ex)
  * height       <- vertical centroid error   (ey)
  * small lateral nudge <- horizontal error   (ex)
  * forward speed -> approach while the gate is reasonably centred

The world-frame projection (`_gate_to_world`) is still computed, but only to
validate/snap the detection into the expected gate's zone and to drive the
top-down map — never to position the drone. Commands are body-frame velocity +
absolute-height setpoints (`send_hover_setpoint`), exactly like ``lap1.py``.

State machine (`_send_setpoint`, runs at ~50 Hz)
------------------------------------------------
  WAIT     -> wait for the Kalman filter to converge, then arm the sequence.
  TAKEOFF  -> ramp the height setpoint up to SEARCH_HEIGHT.
  SEARCH   -> yaw slowly in place until a gate is detected *and* its world
              estimate snaps into the expected gate's zone; then switch to
              CHASE. Stray bright blobs outside the zone are ignored.
  CHASE    -> visually servo the drone toward the gate (yaw/height/lateral) and
              creep forward. Keep validating the detection against the zone for
              the map. Times out back to SEARCH if the gate is lost for
              CHASE_TIMEOUT seconds. When the gate fills the frame
              (area fraction > PASS_AREA_FRAC), commit to a push.
  PUSH     -> drive straight forward for PUSH_DURATION_S, then count the gate
              and return to SEARCH (or DONE once MAX_GATES gates have passed).
  DONE     -> stop setpoints and halt.
  STOP     -> safety brake: continuously command zero velocity. Entered with
              the Escape key.

  Arrow keys / WASD allow manual nudging, Escape engages the zero-velocity
  safety brake, and Space aborts hard (cut motors + DONE).

Usage
-----
  python3 crazy-flower/test/lap1_test_visual.py
Set the radio URI via the CRAZYFLIE_URI env var (default radio://0/70/2M/...).
"""
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

from gate_map import GateZoneMap, GateMapWidget

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

# --- Visual-servo gains (used in CHASE) ---
K_LATERAL = 0.15             # m/s body-Y per normalized x error
MAX_LATERAL = 0.20           # m/s lateral clamp
CHASE_FORWARD = 0.30         # m/s forward creep while approaching a gate
ALIGN_FALLOFF = 0.60         # |ex| at which forward creep is fully suppressed

FORWARD_SPEED = 0.35         # m/s in body X, during push-through
PUSH_DURATION_S = 1.0
SEARCH_HEIGHT = 0.8          # meters, target height for search/center
TAKEOFF_START_HEIGHT = 0.1   # meters, initial setpoint at takeoff
TAKEOFF_RATE = 0.4           # m/s climb rate during takeoff ramp
MAX_GATES = 5

MIN_HEIGHT = 0.2             # meters (safety clamp)
MAX_HEIGHT = 2.0             # meters (safety clamp)
K_HEIGHT = 1.2               # (m/s) per normalized vertical error
MAX_DH_PER_S = 0.6           # max height change rate

MORPH_KERNEL = np.ones((5, 5), np.uint8)

# --- Gate-shape acceptance thresholds (a gate is a roughly-square quad frame) ---
GATE_MIN_VERTICES = 4      # quad after polygon approximation
GATE_MAX_VERTICES = 8      # allow a few extra vertices from noise / rounded corners
GATE_ASPECT_MIN   = 0.45   # bbox w/h: tolerate perspective foreshortening
GATE_ASPECT_MAX   = 2.2
GATE_MIN_SOLIDITY = 0.80   # area / convex-hull area: frame outline is near-convex
GATE_APPROX_EPS   = 0.04   # approxPolyDP epsilon, fraction of perimeter

# --- Camera / world-frame projection (zone validation + map only) ---
DEBUG_GATE_POSE = True               # print per-stage gate pose values for debugging
GATE_PHYS_H    = 0.4                 # metres, physical gate height — the only fixed dimension
                                     # (gate width varies between gates and foreshortens with yaw);
                                     # depth is derived from this height alone
PASS_AREA_FRAC = 0.15                # gate bbox / image area threshold → gate passed
CHASE_TIMEOUT  = 8.0                 # seconds before giving up and returning to SEARCH


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


def _order_corners(pts):
    """Order 4 image points as [TL, TR, BR, BL] (image y increases downward).

    TL/BR are the corners with the smallest/largest x+y; TR/BL the largest/
    smallest x-y. A consistent corner order lets the side edges (TL-BL, TR-BR)
    be measured for the gate's pixel height.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    ordered = np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmax(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmin(d)],   # BL
    ], dtype=np.float64)
    return ordered


def _quad_corners(cnt, approx):
    """Best 4-corner estimate of a gate quad, ordered TL/TR/BR/BL.

    Uses the polygon approximation directly when it is a clean quad (keeps the
    true perspective of the four corners); otherwise falls back to the rotated
    min-area rectangle. Returns None if the 4 corners are not distinct.
    """
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float64)
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
    ordered = _order_corners(pts)
    if len(np.unique(np.round(ordered, 1), axis=0)) != 4:
        return None
    return ordered


def _gate_pixel_height(corners):
    """Vertical pixel extent of the gate from its two side edges (TL-BL, TR-BR).

    Uses the side edges, not the bounding box, so it stays correct under tilt,
    and uses *height* rather than width because a yaw off head-on foreshortens
    the gate's width but leaves its height intact. Returns None if unusable.
    """
    if corners is None:
        return None
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if c.shape[0] != 4:
        return None
    tl, tr, br, bl = c
    left_h  = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    h_px = 0.5 * (left_h + right_h)
    return h_px if h_px > 1.0 else None


def _gate_candidate(cnt, img_w, img_h):
    """Validate a contour as a gate-shaped polygon.

    Returns a candidate dict if the contour passes the polygon/shape tests,
    else None. The centre is the polygon's own area centroid — never the
    axis-aligned bounding box.
    """
    area = float(cv2.contourArea(cnt))
    if area < (MIN_GREEN_AREA_FRAC * float(img_w * img_h)):
        return None

    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return None

    # Polygon approximation: a gate frame reduces to a quadrilateral.
    approx = cv2.approxPolyDP(cnt, GATE_APPROX_EPS * peri, True)
    n_vert = len(approx)
    if not (GATE_MIN_VERTICES <= n_vert <= GATE_MAX_VERTICES):
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    if bh <= 0:
        return None
    aspect = bw / float(bh)
    if not (GATE_ASPECT_MIN <= aspect <= GATE_ASPECT_MAX):
        return None

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0
    if solidity < GATE_MIN_SOLIDITY:
        return None

    # Centre from the polygon itself (area centroid), with a vertex-mean fallback.
    m = cv2.moments(approx)
    if abs(m.get("m00", 0.0)) < 1e-6:
        pts = approx.reshape(-1, 2).astype(np.float64)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
    else:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

    ex = (cx - CAMERA_CX) / max(0.5 * img_w, 1.0)
    ey = (cy - CAMERA_CY) / max(0.5 * img_h, 1.0)

    return {
        "found": True,
        "cx": cx,
        "cy": cy,
        "area": area,
        "bbox": (x, y, bw, bh),
        "ex": ex,
        "ey": ey,
        "approx": approx,
        "corners": _quad_corners(cnt, approx),  # ordered TL/TR/BR/BL for height measurement
    }


def _detect_green_gate(rgb_img):
    """Detect gate(s) and return a detection dict.

    1. Threshold bright pixels into a binary mask, morphologically close it.
    2. Find contours and validate each as a gate-shaped polygon (vertex count,
       aspect ratio, solidity). The centre is the polygon centroid, not the bbox.
    3. Keep every passing candidate; select the *rightmost* one (largest cx) for
       the controller. Rejected/extra candidates are returned under "candidates".
    """
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    threshold = max(int(GREEN_HSV_LO[2]), int(GREEN_MIN_V) if GREEN_MIN_V is not None else 0)
    mask = np.where(gray >= threshold, np.uint8(255), np.uint8(0))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        cand = _gate_candidate(cnt, w, h)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return {"found": False, "mask": mask, "candidates": []}

    # Selection policy: take the most-right gate (largest centroid x).
    best = dict(max(candidates, key=lambda c: c["cx"]))
    best["mask"] = mask
    best["candidates"] = candidates
    best["n_candidates"] = len(candidates)
    return best


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
        self.setWindowTitle('Crazyflie FPV — visual control')

        self.image_label = QtWidgets.QLabel()
        self.status_label = QtWidgets.QLabel('Connecting...')

        # Course zone map (ground-truth gates from gates_xyz.py) + top-down view.
        self._zone_map = GateZoneMap()
        self.gate_map = GateMapWidget(self._zone_map)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.image_label)
        top.addWidget(self.gate_map, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(top)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # Body-frame velocity + absolute-height command sent each tick.
        #   x   -> forward velocity (m/s)
        #   y   -> left velocity    (m/s)
        #   yaw -> yaw rate         (deg/s)
        #   height -> absolute z setpoint (m)
        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': TAKEOFF_START_HEIGHT}
        # State estimator readout (updated by log callback at 50 Hz) — for the map.
        self._est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._pos_lock = threading.Lock()

        # Gate tracking — world estimate is for zone validation + the map ONLY,
        # never for positioning the drone (positioning is purely visual).
        self._gate_world = None        # (gx, gy, gz) last in-zone estimate
        self._last_est_gate = None     # last raw vision estimate (for the map)
        self._last_est_in_zone = None  # whether it passed the zone check

        # Simple autonomy state machine.
        self._gate_state = "WAIT"  # WAIT -> TAKEOFF -> SEARCH -> CHASE -> PUSH -> SEARCH ...
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
        self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(20)
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
                "corners": det.get("corners", None),
            }

        # Debug overlay on the RGB image: candidates thin/yellow, selected thick/green.
        disp = color.copy()
        sel_bbox = det.get("bbox")
        for cand in det.get("candidates", []):
            is_sel = (cand["bbox"] == sel_bbox)
            ccolor = (0, 255, 0) if is_sel else (255, 255, 0)
            cthick = 2 if is_sel else 1
            cx_b, cy_b, bw_b, bh_b = cand["bbox"]
            cv2.rectangle(disp, (cx_b, cy_b), (cx_b + bw_b, cy_b + bh_b), ccolor, cthick)
            if cand.get("approx") is not None:
                cv2.polylines(disp, [cand["approx"]], True, ccolor, cthick)
        if det.get("found", False) and sel_bbox is not None:
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

        if self._gate_state == "STOP":
            # Safety brake: command zero velocity (hold in place, no motion).
            # Keep streaming so the firmware command watchdog stays satisfied.
            self.cf.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
            return

        with self._vision_lock:
            found   = bool(self._vision.get("found", False))
            ex      = float(self._vision.get("ex", 0.0))
            ey      = float(self._vision.get("ey", 0.0))
            bbox    = self._vision.get("bbox", None)
            area    = float(self._vision.get("area", 0.0))
            corners = self._vision.get("corners", None)
        with self._pos_lock:
            est = dict(self._est)

        # Refresh the top-down map: drone pose, current estimate, target gate.
        target_gate = min(self._gates_passed + 1, MAX_GATES)
        disp_gate = self._gate_world if self._gate_world is not None else self._last_est_gate
        self.gate_map.update_state(
            drone=(est['x'], est['y'], est['yaw']),
            est_gate=disp_gate,
            target_gate=target_gate,
            est_in_zone=self._last_est_in_zone,
        )

        img_area = float(IMG_WIDTH * IMG_HEIGHT)

        if self._gate_state == "DONE":
            return

        if not self._log_ready:
            # Wait for the first Lighthouse position sample so the zone check and
            # the map have a real world pose before we start flying.
            return

        # Body-frame velocity commands computed this tick (height is a setpoint
        # held in self.hover['height'] and only nudged).
        x_cmd = 0.0
        y_cmd = 0.0
        yaw_cmd = 0.0

        if self._gate_state == "WAIT":
            self._gate_state = "TAKEOFF"
            self._state_t0   = now

        if self._gate_state == "TAKEOFF":
            self.hover['height'] = float(
                min(self.hover['height'] + TAKEOFF_RATE * dt, SEARCH_HEIGHT))
            if self.hover['height'] >= SEARCH_HEIGHT - 1e-3:
                self._gate_state = "SEARCH"
                self._state_t0 = now

        elif self._gate_state == "SEARCH":
            # Yaw in place; only lock on if the detection's world estimate snaps
            # into the expected gate's zone (rejects stray bright blobs).
            yaw_cmd = SEARCH_YAWRATE
            if found and bbox is not None:
                gw = self._gate_to_world(ex, ey, bbox, corners)
                gate_idx = self._gates_passed + 1
                self._last_est_gate = gw
                snapped = self._zone_map.validate_and_snap(gw, gate_idx)
                self._last_est_in_zone = snapped is not None
                if snapped is not None:
                    self._gate_world = snapped
                    self._gate_state = "CHASE"
                    self._state_t0 = now

        elif self._gate_state == "CHASE":
            if (now - self._state_t0) > CHASE_TIMEOUT:
                # Lost the gate for too long — give up and search again.
                self._gate_world = None
                self._gate_state = "SEARCH"
                self._state_t0   = now
            elif not found or bbox is None:
                # Momentarily lost: hover and wait for re-detection / timeout.
                pass
            else:
                # Keep validating against the zone for the map (display only).
                gw = self._gate_to_world(ex, ey, bbox, corners)
                gate_idx = self._gates_passed + 1
                self._last_est_gate = gw
                snapped = self._zone_map.validate_and_snap(gw, gate_idx)
                self._last_est_in_zone = snapped is not None
                if snapped is not None:
                    self._gate_world = snapped

                if area / img_area > PASS_AREA_FRAC:
                    # Gate fills the frame: commit to a straight push through it.
                    self._gate_state = "PUSH"
                    self._state_t0   = now
                else:
                    # --- Purely visual servo toward the gate ---
                    # Yaw to centre the gate horizontally.
                    yaw_cmd = float(np.clip(-K_YAW * ex, -MAX_YAWRATE, MAX_YAWRATE))
                    # Small lateral nudge to help recentre (kept small).
                    y_cmd = float(np.clip(-K_LATERAL * ex, -MAX_LATERAL, MAX_LATERAL))
                    # Height: ey > 0 means gate below image centre -> descend.
                    dh = float(np.clip(-K_HEIGHT * ey, -MAX_DH_PER_S, MAX_DH_PER_S))
                    self.hover['height'] = float(
                        np.clip(self.hover['height'] + dh * dt, MIN_HEIGHT, MAX_HEIGHT))
                    # Creep forward, slowing down the more off-centre the gate is.
                    align = max(0.0, 1.0 - abs(ex) / ALIGN_FALLOFF)
                    x_cmd = CHASE_FORWARD * align

        elif self._gate_state == "PUSH":
            # Drive straight forward (yaw held) to clear the gate, then reset.
            x_cmd = FORWARD_SPEED
            if (now - self._state_t0) >= PUSH_DURATION_S:
                self._gates_passed += 1
                self._gate_world = None
                if self._gates_passed >= MAX_GATES:
                    self._gate_state = "DONE"
                    self.cf.commander.send_stop_setpoint()
                    self._timer.stop()
                    return
                self._gate_state = "SEARCH"
                self._state_t0   = now

        # Apply the computed body-frame command + height setpoint.
        self.hover['x'] = x_cmd
        self.hover['y'] = y_cmd
        self.hover['yaw'] = yaw_cmd
        self.hover['height'] = float(np.clip(self.hover['height'], MIN_HEIGHT, MAX_HEIGHT))
        self.cf.commander.send_hover_setpoint(
            self.hover['x'], self.hover['y'], self.hover['yaw'], self.hover['height'])

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_W:     self.hover['height'] += 0.1
        if k == QtCore.Qt.Key.Key_S:     self.hover['height'] -= 0.1
        if k == QtCore.Qt.Key.Key_Escape:
            # Safety: abort autonomy and brake to zero velocity (keep hovering).
            self._gate_state = "STOP"
            self._set_status('STOP — zero-velocity safety brake (Space to cut motors)')
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
        try:
            self.cf.log.add_config(lc)
        except Exception as e:
            print(f'Could not add StateEst log config: {e}')
            return
        lc.data_received_cb.add_callback(self._on_log)
        lc.error_cb.add_callback(lambda _conf, msg: print('Log error:', msg))
        lc.start()
        self._log_cfg = lc
        print('StateEst log started — waiting for first Lighthouse position…')

    def _on_log(self, _ts, data, _lc):
        with self._pos_lock:
            self._est['x']   = data['stateEstimate.x']
            self._est['y']   = data['stateEstimate.y']
            self._est['z']   = data['stateEstimate.z']
            self._est['yaw'] = data['stabilizer.yaw']
            if not self._log_ready:
                # Seed the commanded height from the first Lighthouse estimate so
                # takeoff ramps from the drone's real altitude, then begin.
                self.hover['height'] = float(np.clip(self._est['z'], MIN_HEIGHT, MAX_HEIGHT))
                self._log_ready  = True
                print(
                    f"First Lighthouse position: x={self._est['x']:.2f} "
                    f"y={self._est['y']:.2f} z={self._est['z']:.2f} "
                    f"yaw={self._est['yaw']:.1f} — leaving WAIT"
                )

    def _gate_to_world(self, ex, ey, bbox, corners=None):
        """Project a gate detection to a world-frame point.

        Used ONLY for the zone validation/snap and the top-down map — the drone
        is never commanded toward this point. Depth comes from the gate's *pixel
        height*: a yaw off head-on foreshortens the gate's width but not its
        height, so height is the stable ranging dimension. Pinhole:
        Z = fy * GATE_PHYS_H / h_px. The lateral/vertical offset is the
        calibrated back-projection of the gate centroid at that depth. Falls
        back to the bbox-height pinhole when the corners are unavailable.
        """
        h_px = _gate_pixel_height(corners)
        if h_px is not None:
            dist = max(CAMERA_FY * GATE_PHYS_H / h_px, 0.3)
            src = "height"
        else:
            # Fallback (no corners): use the bbox HEIGHT, not width — gate width
            # varies between gates and foreshortens, but height is fixed/known.
            bh = bbox[3]
            dist = max(CAMERA_FY * GATE_PHYS_H / max(bh, 1), 0.3)
            src = "bbox-h"

        # Direction from the gate centroid offset, back-projected at `dist`.
        # Body frame: x forward, y left, z up (camera looks along +body_x).
        x_err_px = ex * max(0.5 * IMG_WIDTH, 1.0)
        y_err_px = ey * max(0.5 * IMG_HEIGHT, 1.0)
        dx_b =  dist
        dy_b = -x_err_px * dist / CAMERA_FX
        dz_b = -y_err_px * dist / CAMERA_FY

        with self._pos_lock:
            yaw_r = np.deg2rad(self._est['yaw'])
            ox, oy, oz = self._est['x'], self._est['y'], self._est['z']
        gx = ox + dx_b * np.cos(yaw_r) - dy_b * np.sin(yaw_r)
        gy = oy + dx_b * np.sin(yaw_r) + dy_b * np.cos(yaw_r)
        gz = float(np.clip(oz + dz_b, MIN_HEIGHT, MAX_HEIGHT))

        if DEBUG_GATE_POSE:
            rng = float(np.sqrt(dx_b**2 + dy_b**2 + dz_b**2))
            bw_px = bh_px = -1.0
            if corners is not None:
                c = np.asarray(corners, np.float64).reshape(-1, 2)
                bw_px = float(c[:, 0].max() - c[:, 0].min())
                bh_px = float(c[:, 1].max() - c[:, 1].min())
            print(
                f"[gate {src}] px(w={bw_px:.0f} h={bh_px:.0f}) "
                f"body(fwd={dx_b:+.2f} left={dy_b:+.2f} up={dz_b:+.2f}) "
                f"range={rng:.2f}m | drone(x={ox:.2f} y={oy:.2f} z={oz:.2f} "
                f"yaw={np.degrees(yaw_r):.0f}) -> gate(x={gx:.2f} y={gy:.2f} z={gz:.2f})"
            )
        return gx, gy, gz

    def _set_status(self, text):
        """Thread-safe status label update."""
        QtCore.QMetaObject.invokeMethod(
            self.status_label, 'setText',
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text))

    def _connected(self, uri):
        # No Kalman reset: the Lighthouse already provides an absolute, converged
        # position estimate. Just start logging and take off.
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
