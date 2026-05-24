#!/usr/bin/env python3
"""Simple Lap-1 hardware gate follower for Crazyflie + AI-deck.

This version deliberately keeps the same gate detection code from the working
examples, but replaces the approach logic with a small visual-servo state
machine plus an optional lightweight triangulation step:

WAIT -> TAKEOFF -> SCAN -> ACQUIRE -> TRIANGULATE -> APPROACH -> PASS -> RECOVER -> SCAN

Main idea:
- Do NOT trust one image to create a world-frame gate target.
- Use several consecutive detections before approaching.
- Optionally collect multiple bearing rays while sliding slightly sideways,
  then triangulate a gate-centre estimate from those rays.
- While approaching, keep using the live image centre error.
- Move forward only when the gate is reasonably centred.
- If the gate is lost, stop moving forward and go back to scanning.

Run:
    python3 lap1_hardware_simple_visual_servo.py

Keys:
    Esc   hover/stop autonomy
    Space cut motors / stop setpoints
    W/S   adjust commanded height a little
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
DETECTION_LOG_PATH = os.path.join(os.path.dirname(__file__), 'gate_detections.json')


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
SEARCH_HEIGHT = 1.2          # meters, target height for search/center
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

# --- Camera / world-frame projection ---
GATE_PHYS_H    = 0.4                 # metres, physical gate height
TRAJ_N_STEPS   = 5                   # number of waypoints in interpolated trajectory
TRAJ_OVERSHOOT = 0.30                # metres past gate centre (to fly through cleanly)
WAYPOINT_TOL   = 0.15                # metres, advance to next waypoint within this radius
PASS_AREA_FRAC = 0.15                # gate bbox / image area threshold → gate passed
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


def _pixels_to_world(pixels, cam_pos, cam_rot, img_shape, fov=1.5, real_height=GATE_PHYS_H):
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    cam_pos = np.asarray(cam_pos, dtype=float)

    f = (w / 2.0) / np.tan(fov / 2.0)

    rays_world = []
    for (u, v) in pixels:
        x_img = (u - cx) / f
        y_img = (v - cy) / f

        ray_body = np.array([1.0, -x_img, -y_img], dtype=float)
        ray_body /= np.linalg.norm(ray_body)

        ray_world = cam_rot @ ray_body
        ray_world /= np.linalg.norm(ray_world)
        rays_world.append(ray_world)

    if len(rays_world) != 4:
        return None

    r0, r1, r2, r3 = rays_world

    def solve_vertical_pair(ray_top, ray_bottom):
        A = np.column_stack((ray_top, -ray_bottom))
        b = np.array([0.0, 0.0, real_height], dtype=float)

        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 2:
            return None

        if np.any(~np.isfinite(sol)) or np.any(sol <= 0.0):
            return None

        pair_residual = np.linalg.norm(A @ sol - b)
        if pair_residual > 0.05:
            return None

        top_world = cam_pos + sol[0] * ray_top
        bottom_world = cam_pos + sol[1] * ray_bottom
        return top_world, bottom_world

    left_pair = solve_vertical_pair(r0, r3)
    right_pair = solve_vertical_pair(r1, r2)
    if left_pair is None or right_pair is None:
        return None

    p0, p3 = left_pair
    p1, p2 = right_pair
    return np.array([p0, p1, p2, p3], dtype=float)


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

# ---------------------------------------------------------------------------
# Simple visual-servo controller tuning
# ---------------------------------------------------------------------------
# If the drone turns away from the gate instead of toward it, change YAW_SIGN
# from +1.0 to -1.0.
YAW_SIGN = 1.0

# If height correction is backwards, change Z_SIGN from -1.0 to +1.0.
# With normal image coordinates, ey > 0 means the gate is below image centre,
# so the drone should usually go down.
Z_SIGN = -1.0

ACQUIRE_MIN_FRAMES = 6          # require this many stable detections before approach
ACQUIRE_TIMEOUT_S = 3.0
LOST_TO_SCAN_S = 0.45           # how long we tolerate losing the gate while acquiring/approaching
CENTERED_FRAMES_TO_APPROACH = 3

APPROACH_SPEED = 0.22           # m/s body-forward, conservative for hardware
APPROACH_SPEED_SLOW = 0.10      # m/s if almost centered but not perfect
PASS_SPEED = 0.32               # m/s body-forward through the gate
PASS_DURATION_S = 1.15
RECOVER_DURATION_S = 0.55

CENTER_TOL_X_APPROACH = 0.16    # allow slow forward if within this
CENTER_TOL_Y_APPROACH = 0.18
CENTER_TOL_X_PASS = 0.22        # pass if close and roughly centered
CENTER_TOL_Y_PASS = 0.24

PASS_AREA_FRAC = 0.13           # gate bbox/image area for "close enough to push"
PASS_AREA_FRAC_FORCE = 0.20     # push even if not perfectly centered, because gate is very close

STATUS_PERIOD_S = 0.12

# ---------------------------------------------------------------------------
# Lightweight triangulation tuning
# ---------------------------------------------------------------------------
# Triangulation estimates the world-frame gate centre from multiple camera rays.
# It is used as a sanity check / map estimate. The final approach still uses
# visual servoing so the drone does not blindly trust the triangulated point.
USE_TRIANGULATION = True
TRI_MIN_OBS = 5
TRI_MIN_BASELINE = 0.12          # m; need sideways motion, not just one pose
TRI_OBS_MIN_SPACING = 0.035      # m between stored observations
TRI_MAX_MEAN_LINE_DIST = 0.18    # m; reject inconsistent ray intersection
TRI_COLLECT_TIMEOUT_S = 3.2
TRI_SWAY_SPEED = 0.07            # m/s body-left/right while collecting rays
TRI_SWAY_PERIOD_S = 0.9
TRI_COLLECT_TOL_X = 0.35         # only store if gate is not too far off-camera
TRI_COLLECT_TOL_Y = 0.40
TRI_MIN_Z = 0.25
TRI_MAX_Z = 2.20


def _clip_float(value, lo, hi):
    return float(np.clip(float(value), float(lo), float(hi)))


class FPVWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crazyflie Lap 1 - visual servo + triangulation")

        self.image_label = QtWidgets.QLabel("Waiting for AI-deck video...")
        self.status_label = QtWidgets.QLabel("Starting...")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        # State estimator readout (updated by log callback).
        self._est = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self._pos_lock = threading.Lock()
        self._log_ready = False

        # Latest vision result from _detect_green_gate.
        self._vision_lock = threading.Lock()
        self._vision = {
            "found": False,
            "ex": 0.0,
            "ey": 0.0,
            "area": 0.0,
            "bbox": None,
            "cx": 0.0,
            "cy": 0.0,
            "stamp": 0.0,
            "corners": None,
        }

        # Autonomy state.
        self._state = "WAIT"
        self._state_t0 = time.monotonic()
        self._last_ctrl_time = time.monotonic()
        self._last_status_time = 0.0
        self._gates_passed = 0

        # Triangulation memory for the current gate. Each observation is a
        # camera origin p and a unit bearing ray r in world coordinates.
        self._tri_obs = []
        self._last_tri_gate = None
        self._last_tri_residual = None

        # Commanded hover height. Hover setpoints use body-frame vx/vy and yawrate.
        self._height_cmd = TAKEOFF_START_HEIGHT

        # Detection stability counters.
        self._acquire_frames = 0
        self._centered_frames = 0
        self._last_seen_t = 0.0

        # Low-pass filtered image errors to avoid twitchy motion.
        self._ex_f = 0.0
        self._ey_f = 0.0
        self._area_f = 0.0
        self._filter_initialized = False

        # Detection log for debugging after a flight.
        self._detections = []

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache="cache")
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.open_link(URI)

        if not self.cf.link:
            print("Could not connect")
            sys.exit(1)

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._update_image)
        self.video.start()

        self.cf.supervisor.send_arming_request(True)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(20)  # 50 Hz command loop
        self._timer.start()

    # ------------------------------------------------------------------
    # Vision update: gate detection is intentionally the same function.
    # ------------------------------------------------------------------
    def _update_image(self, img):
        if img.ndim == 2:
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            color = img

        color = _undistort_image(color)
        det = _detect_green_gate(color)

        now = time.monotonic()
        found = bool(det.get("found", False))

        if found:
            with self._pos_lock:
                est = dict(self._est)

            corners = det.get("corners", None)
            if corners is not None:
                corners = np.asarray(corners, dtype=float).reshape(-1, 2).tolist()

            self._detections.append({
                "t": time.time(),
                "state": self._state,
                "gate_number": self._gates_passed + 1,
                "gate": {
                    "ex": float(det.get("ex", 0.0)),
                    "ey": float(det.get("ey", 0.0)),
                    "bbox": det.get("bbox", None),
                    "area": float(det.get("area", 0.0)),
                    "cx": float(det.get("cx", 0.0)),
                    "cy": float(det.get("cy", 0.0)),
                    "corners": corners,
                },
                "drone": {
                    "x": float(est.get("x", 0.0)),
                    "y": float(est.get("y", 0.0)),
                    "z": float(est.get("z", 0.0)),
                    "yaw": float(est.get("yaw", 0.0)),
                },
            })

        with self._vision_lock:
            self._vision = {
                "found": found,
                "ex": float(det.get("ex", 0.0)),
                "ey": float(det.get("ey", 0.0)),
                "bbox": det.get("bbox", None),
                "area": float(det.get("area", 0.0)),
                "cx": float(det.get("cx", 0.0)),
                "cy": float(det.get("cy", 0.0)),
                "corners": det.get("corners", None),
                "stamp": now if found else self._vision.get("stamp", 0.0),
            }

        self._draw_debug_overlay(color, det)

    def _draw_debug_overlay(self, color, det):
        disp = color.copy()
        sel_bbox = det.get("bbox")

        # Candidates thin/yellow, selected thick/green.
        for cand in det.get("candidates", []):
            is_sel = (cand["bbox"] == sel_bbox)
            ccolor = (0, 255, 0) if is_sel else (255, 255, 0)
            cthick = 2 if is_sel else 1
            x, y, bw, bh = cand["bbox"]
            cv2.rectangle(disp, (x, y), (x + bw, y + bh), ccolor, cthick)
            if cand.get("approx") is not None:
                cv2.polylines(disp, [cand["approx"]], True, ccolor, cthick)

        if det.get("found", False) and sel_bbox is not None:
            cx, cy = int(round(det["cx"])), int(round(det["cy"]))
            cv2.circle(disp, (cx, cy), 4, (255, 0, 0), -1)

        h, w = disp.shape[:2]
        cv2.drawMarker(disp, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

        text1 = f"state={self._state} gate={self._gates_passed + 1}/{MAX_GATES} found={int(det.get('found', False))}"
        text2 = f"ex={det.get('ex', 0.0):+.2f} ey={det.get('ey', 0.0):+.2f} area={det.get('area', 0.0)/(IMG_WIDTH*IMG_HEIGHT):.3f}"
        cv2.putText(disp, text1, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp, text2, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        h, w, ch = disp.shape
        q = QtGui.QImage(disp.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    # ------------------------------------------------------------------
    # Main state machine
    # ------------------------------------------------------------------
    def _send_setpoint(self):
        now = time.monotonic()
        dt = _clip_float(now - self._last_ctrl_time, 0.02, 0.20)
        self._last_ctrl_time = now

        if self._state == "STOP":
            self._send_hover(0.0, 0.0, 0.0, self._height_cmd)
            self._status("STOP: hovering. Press Space only when safe to cut motors.")
            return

        if self._state == "DONE":
            self._send_hover(0.0, 0.0, 0.0, self._height_cmd)
            self._status(f"DONE: passed {self._gates_passed}/{MAX_GATES}. Hovering.")
            return

        if not self._log_ready:
            self._status("WAIT: waiting for first Lighthouse/Kalman state estimate...")
            return

        vision = self._read_filtered_vision(now)

        if self._state == "WAIT":
            self._change_state("TAKEOFF")
            self._height_cmd = max(TAKEOFF_START_HEIGHT, min(self._height_cmd, SEARCH_HEIGHT))

        if self._state == "TAKEOFF":
            self._height_cmd = min(self._height_cmd + TAKEOFF_RATE * dt, SEARCH_HEIGHT)
            self._send_hover(0.0, 0.0, 0.0, self._height_cmd)
            self._status(f"TAKEOFF: z_cmd={self._height_cmd:.2f} m")

            if self._height_cmd >= SEARCH_HEIGHT - 0.02:
                self._reset_detection_memory()
                self._change_state("SCAN")
            return

        if self._state == "SCAN":
            # Rotate in place until a gate appears. Do not move forward in SCAN.
            self._send_hover(0.0, 0.0, SEARCH_YAWRATE, self._height_cmd)
            self._status(f"SCAN: rotating, passed={self._gates_passed}/{MAX_GATES}")

            if vision["found"]:
                self._reset_detection_memory()
                self._change_state("ACQUIRE")
            return

        if self._state == "ACQUIRE":
            if not vision["recent"]:
                if now - self._state_t0 > ACQUIRE_TIMEOUT_S:
                    self._change_state("SCAN")
                self._send_hover(0.0, 0.0, SEARCH_YAWRATE * 0.5, self._height_cmd)
                self._status("ACQUIRE: lost candidate, slowly scanning")
                return

            yawrate, z_cmd = self._centering_commands(vision, dt)
            self._height_cmd = z_cmd

            centered = self._is_centered(vision, CENTER_TOL_X, CENTER_TOL_Y)
            if centered:
                self._centered_frames += 1
            else:
                self._centered_frames = 0

            # Count stable detection frames. This is the main difference from the
            # previous code: one image is not enough to start flying at the gate.
            self._acquire_frames += 1

            self._send_hover(0.0, 0.0, yawrate, self._height_cmd)
            self._status(
                f"ACQUIRE: frames={self._acquire_frames}/{ACQUIRE_MIN_FRAMES}, "
                f"centered={self._centered_frames}/{CENTERED_FRAMES_TO_APPROACH}"
            )

            if (
                self._acquire_frames >= ACQUIRE_MIN_FRAMES
                and self._centered_frames >= CENTERED_FRAMES_TO_APPROACH
            ):
                self._reset_tri_memory()
                self._change_state("TRIANGULATE" if USE_TRIANGULATION else "APPROACH")
            return

        if self._state == "TRIANGULATE":
            if not vision["recent"]:
                if now - self._last_seen_t > LOST_TO_SCAN_S:
                    self._reset_detection_memory()
                    self._reset_tri_memory()
                    self._change_state("SCAN")
                self._send_hover(0.0, 0.0, SEARCH_YAWRATE * 0.25, self._height_cmd)
                self._status("TRIANGULATE: lost gate, stopping")
                return

            yawrate, z_cmd = self._centering_commands(vision, dt)
            self._height_cmd = z_cmd

            # Store bearing observations while the gate is visible. The drone
            # sways sideways to create baseline; without baseline triangulation
            # is mathematically weak.
            if abs(vision["ex"]) < TRI_COLLECT_TOL_X and abs(vision["ey"]) < TRI_COLLECT_TOL_Y:
                self._store_tri_observation(vision)

            gate, residual = self._triangulate_gate_center()
            if gate is not None:
                self._last_tri_gate = gate
                self._last_tri_residual = residual
                print(
                    f"[TRI] gate {self._gates_passed + 1}: "
                    f"x={gate[0]:.2f}, y={gate[1]:.2f}, z={gate[2]:.2f}, "
                    f"obs={len(self._tri_obs)}, residual={residual:.3f} m",
                    flush=True,
                )
                self._change_state("APPROACH")
                return

            # Gentle left/right sway while maintaining yaw/height centering.
            phase = int((now - self._state_t0) / TRI_SWAY_PERIOD_S)
            vy = TRI_SWAY_SPEED if (phase % 2 == 0) else -TRI_SWAY_SPEED

            if now - self._state_t0 > TRI_COLLECT_TIMEOUT_S:
                # Do not get stuck forever. If triangulation is not valid, continue
                # with visual servo only. This is safer than forcing a bad point.
                print(f"[TRI] no valid triangulation; obs={len(self._tri_obs)} -> visual approach", flush=True)
                self._change_state("APPROACH")
                return

            self._send_hover(0.0, vy, yawrate, self._height_cmd)
            self._status(f"TRIANGULATE: obs={len(self._tri_obs)}/{TRI_MIN_OBS}, sliding vy={vy:+.2f}")
            return

        if self._state == "APPROACH":
            if not vision["recent"]:
                # Safer behavior: do not continue forward blindly.
                if now - self._last_seen_t > LOST_TO_SCAN_S:
                    self._reset_detection_memory()
                    self._change_state("SCAN")
                self._send_hover(0.0, 0.0, SEARCH_YAWRATE * 0.3, self._height_cmd)
                self._status("APPROACH: lost gate, stopping forward motion")
                return

            yawrate, z_cmd = self._centering_commands(vision, dt)
            self._height_cmd = z_cmd

            centered_for_fast = self._is_centered(vision, CENTER_TOL_X, CENTER_TOL_Y)
            centered_for_slow = self._is_centered(vision, CENTER_TOL_X_APPROACH, CENTER_TOL_Y_APPROACH)

            if centered_for_fast:
                vx = APPROACH_SPEED
            elif centered_for_slow:
                vx = APPROACH_SPEED_SLOW
            else:
                vx = 0.0

            area_frac = vision["area_frac"]
            pass_now = (
                area_frac >= PASS_AREA_FRAC and self._is_centered(vision, CENTER_TOL_X_PASS, CENTER_TOL_Y_PASS)
            ) or (area_frac >= PASS_AREA_FRAC_FORCE)

            if pass_now:
                self._change_state("PASS")
                self._send_hover(PASS_SPEED, 0.0, 0.0, self._height_cmd)
                return

            self._send_hover(vx, 0.0, yawrate, self._height_cmd)
            self._status(
                f"APPROACH: vx={vx:.2f}, ex={vision['ex']:+.2f}, "
                f"ey={vision['ey']:+.2f}, area={area_frac:.3f}"
            )
            return

        if self._state == "PASS":
            # Once the gate fills the image, keep pushing straight briefly. This
            # avoids stopping inside the frame or reacting to partial detections.
            self._send_hover(PASS_SPEED, 0.0, 0.0, self._height_cmd)
            self._status("PASS: pushing straight through gate")

            if now - self._state_t0 >= PASS_DURATION_S:
                self._gates_passed += 1
                self._reset_detection_memory()
                self._reset_tri_memory()

                if self._gates_passed >= MAX_GATES:
                    self._change_state("DONE")
                else:
                    self._change_state("RECOVER")
            return

        if self._state == "RECOVER":
            # Small pause after a pass so the same gate is less likely to be
            # immediately reacquired. Then scan for the next gate.
            self._send_hover(0.0, 0.0, SEARCH_YAWRATE * 0.5, self._height_cmd)
            self._status(f"RECOVER: finished gate {self._gates_passed}, preparing next scan")

            if now - self._state_t0 >= RECOVER_DURATION_S:
                self._change_state("SCAN")
            return

        # Fallback: hover if something unexpected happens.
        self._send_hover(0.0, 0.0, 0.0, self._height_cmd)
        self._status(f"Unknown state {self._state}; hovering")

    # ------------------------------------------------------------------
    # Controller helpers
    # ------------------------------------------------------------------
    def _read_filtered_vision(self, now):
        with self._vision_lock:
            v = dict(self._vision)

        found = bool(v.get("found", False))
        stamp = float(v.get("stamp", 0.0))
        recent = found and (now - stamp <= LOST_TO_SCAN_S)

        if found:
            self._last_seen_t = now
            ex = float(v.get("ex", 0.0))
            ey = float(v.get("ey", 0.0))
            area = float(v.get("area", 0.0))

            if not self._filter_initialized:
                self._ex_f = ex
                self._ey_f = ey
                self._area_f = area
                self._filter_initialized = True
            else:
                alpha = 0.35
                self._ex_f = (1.0 - alpha) * self._ex_f + alpha * ex
                self._ey_f = (1.0 - alpha) * self._ey_f + alpha * ey
                self._area_f = (1.0 - alpha) * self._area_f + alpha * area

        area_frac = self._area_f / max(float(IMG_WIDTH * IMG_HEIGHT), 1.0)

        return {
            "found": found,
            "recent": recent,
            "ex": float(self._ex_f),
            "ey": float(self._ey_f),
            "area": float(self._area_f),
            "area_frac": float(area_frac),
            "bbox": v.get("bbox", None),
            "cx": float(v.get("cx", CAMERA_CX)),
            "cy": float(v.get("cy", CAMERA_CY)),
        }

    def _bearing_ray_from_pixel(self, u, v, est):
        """Return a world-frame unit ray from camera centre through pixel (u, v).

        This uses the same camera convention as the original projection helper:
        body +X is camera forward, body +Y is image-left/right with a sign flip,
        and body +Z follows the image vertical sign convention used there.
        Roll/pitch are ignored because the hardware examples only log yaw here.
        """
        x_img = (float(u) - CAMERA_CX) / max(CAMERA_FX, 1e-6)
        y_img = (float(v) - CAMERA_CY) / max(CAMERA_FY, 1e-6)

        ray_body = np.array([1.0, -x_img, -y_img], dtype=float)
        ray_body /= max(np.linalg.norm(ray_body), 1e-9)

        yaw_r = np.deg2rad(float(est["yaw"]))
        R = np.array(
            [
                [np.cos(yaw_r), -np.sin(yaw_r), 0.0],
                [np.sin(yaw_r),  np.cos(yaw_r), 0.0],
                [0.0,            0.0,           1.0],
            ],
            dtype=float,
        )
        ray_world = R @ ray_body
        ray_world /= max(np.linalg.norm(ray_world), 1e-9)
        return ray_world

    def _store_tri_observation(self, vision):
        with self._pos_lock:
            est = dict(self._est)

        p = np.array([est["x"], est["y"], est["z"]], dtype=float)
        r = self._bearing_ray_from_pixel(vision["cx"], vision["cy"], est)

        if self._tri_obs:
            if np.linalg.norm(p - self._tri_obs[-1]["p"]) < TRI_OBS_MIN_SPACING:
                return

        self._tri_obs.append({"p": p, "r": r, "t": time.monotonic()})

    def _triangulate_gate_center(self):
        """Least-squares point closest to all stored bearing rays.

        Each observation gives a line p_i + lambda*r_i. The solution H minimizes
        the summed squared perpendicular distances to all lines. The residual is
        the mean point-to-line distance; large residual means the observations are
        inconsistent, so the estimate is rejected.
        """
        if len(self._tri_obs) < TRI_MIN_OBS:
            return None, None

        ps = np.array([o["p"] for o in self._tri_obs], dtype=float)
        baseline = np.max(np.linalg.norm(ps - ps[0], axis=1))
        if baseline < TRI_MIN_BASELINE:
            return None, None

        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)
        for obs in self._tri_obs:
            p = obs["p"]
            r = obs["r"]
            M = np.eye(3) - np.outer(r, r)
            A += M
            b += M @ p

        try:
            H = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            H = np.linalg.lstsq(A, b, rcond=None)[0]

        if not np.all(np.isfinite(H)):
            return None, None

        dists = []
        for obs in self._tri_obs:
            p = obs["p"]
            r = obs["r"]
            dists.append(np.linalg.norm(np.cross(H - p, r)))
        mean_dist = float(np.mean(dists))

        # Basic sanity checks. A bad calibration/sign can otherwise create a
        # gate estimate behind the drone or far outside a reasonable height.
        if mean_dist > TRI_MAX_MEAN_LINE_DIST:
            return None, mean_dist
        if H[2] < TRI_MIN_Z or H[2] > TRI_MAX_Z:
            return None, mean_dist

        return H.astype(float), mean_dist

    def _centering_commands(self, vision, dt):
        ex = vision["ex"]
        ey = vision["ey"]

        # Body yaw correction from image horizontal error.
        yawrate = YAW_SIGN * K_YAW * ex
        yawrate = _clip_float(yawrate, -MAX_YAWRATE, MAX_YAWRATE)

        # Height correction from image vertical error.
        dz_rate = Z_SIGN * K_HEIGHT * ey
        dz_rate = _clip_float(dz_rate, -MAX_DH_PER_S, MAX_DH_PER_S)
        z_cmd = _clip_float(self._height_cmd + dz_rate * dt, MIN_HEIGHT, MAX_HEIGHT)

        return yawrate, z_cmd

    def _is_centered(self, vision, tol_x, tol_y):
        return abs(vision["ex"]) <= tol_x and abs(vision["ey"]) <= tol_y

    def _send_hover(self, vx, vy, yawrate, z):
        z = _clip_float(z, MIN_HEIGHT, MAX_HEIGHT)
        vx = _clip_float(vx, -0.35, 0.45)
        vy = _clip_float(vy, -0.25, 0.25)
        yawrate = _clip_float(yawrate, -MAX_YAWRATE, MAX_YAWRATE)
        self.cf.commander.send_hover_setpoint(vx, vy, yawrate, z)

    def _change_state(self, new_state):
        if new_state != self._state:
            print(f"[STATE] {self._state} -> {new_state}", flush=True)
        self._state = new_state
        self._state_t0 = time.monotonic()

    def _reset_detection_memory(self):
        self._acquire_frames = 0
        self._centered_frames = 0
        self._filter_initialized = False
        self._ex_f = 0.0
        self._ey_f = 0.0
        self._area_f = 0.0

    def _reset_tri_memory(self):
        self._tri_obs = []
        self._last_tri_gate = None
        self._last_tri_residual = None

    def _status(self, text):
        now = time.monotonic()
        if now - self._last_status_time < STATUS_PERIOD_S:
            return
        self._last_status_time = now
        QtCore.QMetaObject.invokeMethod(
            self.status_label,
            "setText",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

    # ------------------------------------------------------------------
    # Keyboard and Crazyflie callbacks
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        k = event.key()
        if k == QtCore.Qt.Key.Key_W:
            self._height_cmd = _clip_float(self._height_cmd + 0.05, MIN_HEIGHT, MAX_HEIGHT)
        elif k == QtCore.Qt.Key.Key_S:
            self._height_cmd = _clip_float(self._height_cmd - 0.05, MIN_HEIGHT, MAX_HEIGHT)
        elif k == QtCore.Qt.Key.Key_Escape:
            self._state = "STOP"
            self._status("STOP: hovering, autonomy disabled")
        elif k == QtCore.Qt.Key.Key_Space:
            self._state = "DONE"
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()
            self._status("Motors stopped / setpoints stopped")

    def _setup_log(self):
        lc = LogConfig("StateEst", period_in_ms=20)
        lc.add_variable("stateEstimate.x", "float")
        lc.add_variable("stateEstimate.y", "float")
        lc.add_variable("stateEstimate.z", "float")
        lc.add_variable("stabilizer.yaw", "float")

        try:
            self.cf.log.add_config(lc)
        except Exception as e:
            print(f"Could not add StateEst log config: {e}")
            return

        lc.data_received_cb.add_callback(self._on_log)
        lc.error_cb.add_callback(lambda _conf, msg: print("Log error:", msg))
        lc.start()
        self._log_cfg = lc
        print("StateEst log started — waiting for first position estimate...")

    def _on_log(self, _ts, data, _lc):
        with self._pos_lock:
            self._est["x"] = data["stateEstimate.x"]
            self._est["y"] = data["stateEstimate.y"]
            self._est["z"] = data["stateEstimate.z"]
            self._est["yaw"] = data["stabilizer.yaw"]

            if not self._log_ready:
                self._height_cmd = _clip_float(max(self._est["z"], TAKEOFF_START_HEIGHT), MIN_HEIGHT, SEARCH_HEIGHT)
                self._log_ready = True
                print(
                    f"First estimate: x={self._est['x']:.2f}, y={self._est['y']:.2f}, "
                    f"z={self._est['z']:.2f}, yaw={self._est['yaw']:.1f}"
                )

    def _connected(self, uri):
        self._status(f"Connected to {uri}")
        self._setup_log()

    def _disconnected(self, uri):
        print("Disconnected")
        sys.exit(1)

    def closeEvent(self, event):
        self._timer.stop()
        if hasattr(self, "_log_cfg"):
            self._log_cfg.stop()

        try:
            with open(DETECTION_LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(self._detections, f, indent=2)
            print(f"Saved gate detections to {DETECTION_LOG_PATH}")
        except Exception as e:
            print(f"Failed to save gate detections: {e}")

        try:
            self.cf.commander.send_stop_setpoint()
            self.cf.close_link()
        except Exception:
            pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()
