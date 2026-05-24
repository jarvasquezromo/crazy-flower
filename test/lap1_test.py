#!/usr/bin/env python3
"""Autonomous gate-flying lap for the Crazyflie + AI-deck.

The drone streams JPEG frames from the AI-deck over Wi-Fi/UDP, detects green
race gates in each frame, and flies through them one after another using the
Lighthouse/Kalman world-frame estimate for positioning.

Pipeline
--------
1. Video link (`UdpVideoThread`)
   Connects to the AI-deck (192.168.4.1:5000), reassembles the CPX/JPEG image
   stream, decodes each frame to RGB, and emits it to the Qt GUI. The first
   frame's size selects/scales the camera calibration (`calibration.json`).

2. Gate detection (`_detect_green_gate`, `_gate_candidate`)
   - Threshold the (undistorted) image into a binary mask of bright pixels and
     morphologically close it.
   - For every contour, validate it as a *gate-shaped polygon*: approximate it
     with `approxPolyDP` and require a near-quad vertex count, a roughly square
     aspect ratio, and high solidity (near-convex outline). This rejects blobs
     that are bright but not gate-like.
   - The gate centre is the polygon's own area centroid (not the bounding box).
   - All passing candidates are kept; the controller uses the *rightmost* gate
     (largest centroid x) so the lap is taken consistently around the course.
   - The four gate corners feed `_gate_to_world`, which back-projects each
     corner to a viewing ray (camera intrinsics), rotates the rays into the
     world with the drone's full attitude quaternion (so camera pitch/roll are
     accounted for, not just yaw), and recovers metric depth by enforcing that
     each vertical gate edge is a vertical segment of known physical height
     (GATE_PHYS_H). The gate centre is the mean of the four world corners.
     (Ported from petr_assignment.py's Surveyer.pixels_to_world.)

3. State machine (`_send_setpoint`, runs at 10 Hz)
   WAIT     -> wait for the Kalman filter to converge, then arm the sequence.
   TAKEOFF  -> ramp the height setpoint up to SEARCH_HEIGHT.
   SEARCH   -> yaw slowly in place until a gate is detected; on detection, lock
               its world position, build a straight fly-through trajectory
               (with overshoot) and switch to CHASE.
   CHASE    -> follow the trajectory waypoints while continuously refining the
               locked gate position (EMA) and pointing yaw at the gate. Times
               out back to SEARCH if the gate is lost for CHASE_TIMEOUT seconds.
               When the gate fills the frame (area fraction > PASS_AREA_FRAC),
               commit to a push.
   PUSH     -> drive straight through the gate to an overshoot point for
               PUSH_DURATION_S, then count the gate and return to SEARCH
               (or DONE once MAX_GATES gates have been passed).
   DONE     -> stop setpoints and halt.

   STOP     -> safety brake: continuously command zero velocity (hover, no
               motion). Entered with the Escape key.

   Setpoints are sent as world-frame position+yaw commands. Arrow keys / WASD
   allow manual nudging, Escape engages the zero-velocity safety brake, and
   Space aborts hard (cut motors + DONE).

Usage
-----
  python3 crazy-flower/test/lap1_test.py
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

# --- Camera / world-frame projection ---
DEBUG_GATE_POSE = True               # print per-stage gate pose values for debugging
GATE_PHYS_H    = 0.4                 # metres, physical gate height — the only fixed dimension
                                     # (gate width varies between gates and foreshortens with yaw);
                                     # depth is derived from this height alone
GATE_BORDER_MARGIN     = 15          # px; reject gates whose corners touch the frame edge
VERT_PAIR_RESIDUAL_MAX = 0.05        # m; max residual of the vertical-edge metric solve
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


def _order_points(pts):
    """Order 4 image points as [TL, TR, BR, BL] (image y increases downward).

    Sort by y to split into the top and bottom pairs, then sort each pair by x.
    Robust for rotated quads (unlike an x+/-y diagonal sort, which can mis-order
    near-45-degree rotations). Ported from petr_assignment.Surveyer.order_points.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2]
    bottom = pts[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)


def _gate_corners_robust(cnt):
    """Four gate corners from a contour via convex hull + multi-epsilon approx.

    Tries an increasing approxPolyDP tolerance until the hull reduces to a clean
    quadrilateral, which is far more reliable than a single fixed epsilon.
    Returns an unordered (4, 2) float array or None. Ported from
    petr_assignment.Surveyer.get_gate_corners.
    """
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    if peri < 1e-6:
        return None
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(np.float64)
    return None


def _quad_corners(cnt, approx=None):
    """Best 4-corner estimate of a gate quad, ordered TL/TR/BR/BL.

    Uses the robust hull+multi-epsilon finder, falling back to the rotated
    min-area rectangle. Returns None if the 4 corners are not distinct.
    """
    pts = _gate_corners_robust(cnt)
    if pts is None:
        pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
    ordered = _order_points(pts)
    if len(np.unique(np.round(ordered, 1), axis=0)) != 4:
        return None
    return ordered


def _gate_fully_in_frame(corners, img_w, img_h, margin=GATE_BORDER_MARGIN):
    """True if every corner sits at least `margin` px inside the image.

    A gate clipped by the frame edge has corners that no longer mark the real
    opening, so its metric size (hence depth) is unreliable — reject it for
    pose estimation. Ported from Surveyer.is_gate_fully_contained_in_screen.
    """
    c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if np.any(c[:, 0] < margin) or np.any(c[:, 0] > img_w - margin):
        return False
    if np.any(c[:, 1] < margin) or np.any(c[:, 1] > img_h - margin):
        return False
    return True


def _quat_to_rot(qx, qy, qz, qw):
    """Body->world rotation matrix from a (qx, qy, qz, qw) quaternion."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = q / n
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ], dtype=np.float64)


def _pixel_ray_body(u, v):
    """Unit viewing ray (body frame) for image pixel (u, v).

    Body frame: x forward (optical axis), y left, z up. Uses the calibrated
    intrinsics rather than a single fov-derived focal length.
    """
    x_img = (u - CAMERA_CX) / CAMERA_FX
    y_img = (v - CAMERA_CY) / CAMERA_FY
    ray = np.array([1.0, -x_img, -y_img], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def _solve_vertical_pair(ray_top, ray_bottom, cam_pos, real_height):
    """Metric depth of a vertical gate edge from its top/bottom corner rays.

    Solves for the two ray scales (depths) such that the world points differ
    only in z by exactly `real_height` (i.e. the edge is a vertical segment of
    known length). Returns (top_world, bottom_world) or None if the solve is
    rank-deficient, gives a non-positive depth, or fits poorly. Ported from
    Surveyer.pixels_to_world.solve_vertical_pair.
    """
    A = np.column_stack((ray_top, -ray_bottom))
    b = np.array([0.0, 0.0, real_height], dtype=np.float64)
    sol, _residuals, rank, _sv = np.linalg.lstsq(A, b, rcond=None)
    if rank < 2 or np.any(~np.isfinite(sol)) or np.any(sol <= 0.0):
        return None
    if np.linalg.norm(A @ sol - b) > VERT_PAIR_RESIDUAL_MAX:
        return None
    top_world = cam_pos + sol[0] * ray_top
    bottom_world = cam_pos + sol[1] * ray_bottom
    return top_world, bottom_world


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
        self.setWindowTitle('Crazyflie FPV')

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

        # Commanded world-frame position sent to the flight controller each tick.
        self._pos = {'x': 0.0, 'y': 0.0, 'z': TAKEOFF_START_HEIGHT, 'yaw': 0.0}
        # State estimator readout (updated by log callback at 50 Hz).
        self._est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        # Full attitude quaternion (qx, qy, qz, qw) for the corner-ray gate
        # projection; updated by a second log config. Identity until first sample.
        self._quat = (0.0, 0.0, 0.0, 1.0)
        self._pos_lock = threading.Lock()

        # Gate tracking in world frame.
        self._gate_world = None   # (gx, gy, gz) locked estimate
        self._traj       = []     # list of (x, y, z) waypoints
        self._traj_idx   = 0
        self._last_est_gate = None     # last raw vision estimate (for the map)
        self._last_est_in_zone = None  # whether it passed the zone check

        # Simple autonomy state machine.
        self._gate_state = "WAIT"  # WAIT -> TAKEOFF -> SEARCH -> CHASE -> PUSH -> SEARCH ...
        self._push_target = (0.0, 0.0, TAKEOFF_START_HEIGHT)  # world point for PUSH state
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
            # Wait for the first Lighthouse position sample so the commanded
            # position is seeded from the real estimate (not world-origin).
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
                gw = self._gate_to_world(corners)
                gate_idx = self._gates_passed + 1
                snapped = self._zone_map.validate_and_snap(gw, gate_idx) if gw is not None else None
                if gw is not None:
                    self._last_est_gate = gw
                    self._last_est_in_zone = snapped is not None
                # Only lock and chase if the estimate falls in the expected
                # gate's zone; otherwise keep yawing (a stray bright blob).
                if snapped is not None:
                    self._gate_world = snapped
                    self._traj     = self._build_traj(snapped)
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
                    new_gw = self._gate_to_world(corners)
                    gate_idx = self._gates_passed + 1
                    snapped = self._zone_map.validate_and_snap(new_gw, gate_idx) if new_gw is not None else None
                    if new_gw is not None:
                        self._last_est_gate = new_gw
                        self._last_est_in_zone = snapped is not None
                    # Ignore refinements that drift out of the expected zone.
                    if snapped is not None:
                        ox, oy, oz = self._gate_world
                        nx, ny, nz = snapped
                        self._gate_world = (
                            ox + GATE_EMA_ALPHA * (nx - ox),
                            oy + GATE_EMA_ALPHA * (ny - oy),
                            oz + GATE_EMA_ALPHA * (nz - oz),
                        )
                        if self._traj_idx == 0:
                            self._traj = self._build_traj(self._gate_world)

                # Gate close enough (bbox fills a large fraction of the frame):
                # commit to a straight push through it before searching again.
                if area / img_area > PASS_AREA_FRAC:
                    yaw_r = np.deg2rad(est['yaw'])
                    push_dist = FORWARD_SPEED * PUSH_DURATION_S + TRAJ_OVERSHOOT
                    self._push_target = (
                        est['x'] + push_dist * np.cos(yaw_r),
                        est['y'] + push_dist * np.sin(yaw_r),
                        self._pos['z'],
                    )
                    self._gate_state = "PUSH"
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

        elif self._gate_state == "PUSH":
            # Drive straight to the overshoot point (yaw held), then reset to SEARCH.
            self._pos['x'], self._pos['y'], self._pos['z'] = self._push_target
            if (now - self._state_t0) >= PUSH_DURATION_S:
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

        self._pos['z'] = float(np.clip(self._pos['z'], MIN_HEIGHT, MAX_HEIGHT))
        #self.cf.commander.send_position_setpoint(
        #    self._pos['x'], self._pos['y'], self._pos['z'], self._pos['yaw'])

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

        # Second config: full attitude quaternion for the gate corner-ray
        # projection. Kept separate so neither config exceeds the log packet
        # size limit (StateEst already carries 4 floats).
        qc = LogConfig('Quat', period_in_ms=20)
        qc.add_variable('stateEstimate.qx', 'float')
        qc.add_variable('stateEstimate.qy', 'float')
        qc.add_variable('stateEstimate.qz', 'float')
        qc.add_variable('stateEstimate.qw', 'float')
        try:
            self.cf.log.add_config(qc)
        except Exception as e:
            print(f'Could not add Quat log config: {e}')
        else:
            qc.data_received_cb.add_callback(self._on_log_quat)
            qc.error_cb.add_callback(lambda _conf, msg: print('Quat log error:', msg))
            qc.start()
            self._log_quat_cfg = qc

    def _on_log_quat(self, _ts, data, _lc):
        with self._pos_lock:
            self._quat = (
                data['stateEstimate.qx'],
                data['stateEstimate.qy'],
                data['stateEstimate.qz'],
                data['stateEstimate.qw'],
            )

    def _on_log(self, _ts, data, _lc):
        with self._pos_lock:
            self._est['x']   = data['stateEstimate.x']
            self._est['y']   = data['stateEstimate.y']
            self._est['z']   = data['stateEstimate.z']
            self._est['yaw'] = data['stabilizer.yaw']
            if not self._log_ready:
                # Seed commanded position from the first Lighthouse estimate so
                # the drone holds its current position rather than jumping to
                # world-origin, then takeoff can begin immediately.
                self._pos['x']   = self._est['x']
                self._pos['y']   = self._est['y']
                self._pos['z']   = self._est['z']
                self._pos['yaw'] = self._est['yaw']
                self._log_ready  = True
                print(
                    f"First Lighthouse position: x={self._est['x']:.2f} "
                    f"y={self._est['y']:.2f} z={self._est['z']:.2f} "
                    f"yaw={self._est['yaw']:.1f} — leaving WAIT"
                )

    def _gate_to_world(self, corners):
        """Project the four gate corners to a world-frame gate centre.

        Each corner is back-projected to a viewing ray (calibrated intrinsics),
        rotated into the world with the drone's full attitude quaternion (so
        camera pitch/roll are handled, not just yaw), and metric depth is
        recovered by enforcing that each vertical gate edge is a vertical
        segment of length GATE_PHYS_H. Returns the (gx, gy, gz) mean of the four
        world corners, or None if the gate is clipped by the frame or the
        geometry does not solve cleanly. Ported from
        petr_assignment.Surveyer.pixels_to_world.
        """
        if corners is None:
            return None
        c = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if c.shape[0] != 4:
            return None
        # A clipped gate's corners no longer mark the real opening -> bad scale.
        if not _gate_fully_in_frame(c, IMG_WIDTH, IMG_HEIGHT):
            return None

        with self._pos_lock:
            cam_pos = np.array(
                [self._est['x'], self._est['y'], self._est['z']], dtype=np.float64)
            quat = self._quat
            yaw_dbg = self._est['yaw']
        rot = _quat_to_rot(*quat)  # body -> world

        # Viewing rays for TL, TR, BR, BL in the world frame.
        rays = []
        for (u, v) in c:
            rw = rot @ _pixel_ray_body(u, v)
            rays.append(rw / np.linalg.norm(rw))
        r_tl, r_tr, r_br, r_bl = rays

        # Solve each vertical edge (left: TL-BL, right: TR-BR) for metric depth.
        left = _solve_vertical_pair(r_tl, r_bl, cam_pos, GATE_PHYS_H)
        right = _solve_vertical_pair(r_tr, r_br, cam_pos, GATE_PHYS_H)
        if left is None or right is None:
            return None
        p_tl, p_bl = left
        p_tr, p_br = right
        world_corners = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float64)

        gx, gy, gz = world_corners.mean(axis=0)
        gz = float(np.clip(gz, MIN_HEIGHT, MAX_HEIGHT))

        if DEBUG_GATE_POSE:
            gate_left = 0.5 * (p_tl + p_bl)
            gate_right = 0.5 * (p_tr + p_br)
            ang = float(np.degrees(np.arctan2(
                gate_right[1] - gate_left[1], gate_right[0] - gate_left[0])))
            rng = float(np.linalg.norm(world_corners.mean(axis=0) - cam_pos))
            print(
                f"[gate ray] range={rng:.2f}m angle={ang:+.0f} | "
                f"drone(x={cam_pos[0]:.2f} y={cam_pos[1]:.2f} z={cam_pos[2]:.2f} "
                f"yaw={yaw_dbg:.0f}) -> gate(x={gx:.2f} y={gy:.2f} z={gz:.2f})"
            )
        return float(gx), float(gy), gz

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
        # No Kalman reset: the Lighthouse already provides an absolute, converged
        # position estimate. Resetting would discard it and force a reconvergence
        # wait. Just start logging and take off using the current estimate.
        self._set_status(f'Connected to {uri}')
        self._setup_log()

    def _disconnected(self, uri):
        print('Disconnected')
        sys.exit(1)

    def closeEvent(self, event):
        self._timer.stop()
        if hasattr(self, '_log_cfg'):
            self._log_cfg.stop()
        if hasattr(self, '_log_quat_cfg'):
            self._log_quat_cfg.stop()
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()