#!/usr/bin/env python3
"""Very simple lap-1 gate follower for Crazyflie + AI-deck.

Behavior:
- search by yawing slowly counterclockwise in place,
- when a gate is detected, lock it and fly toward a point 20 cm after the gate,
- keep altitude constant,
- once the target point is reached, hold position and keep turning slowly CCW,
- then continue searching for the next gate.

This file intentionally avoids the more complex mapping / triangulation logic.
"""

import contextlib
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
import warnings

import cv2
import cflib.crtp
import numpy as np
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt6 import QtCore, QtGui, QtWidgets

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore", message=".*TYPE_HOVER_LEGACY.*")
warnings.filterwarnings("ignore", message=".*supervisor subsystem requires CRTP.*")

URI = uri_helper.uri_from_env(default="radio://0/70/2M/E7E7E7E705")
AIDECK_IP = "192.168.4.1"
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b"FER"

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
MIN_JPEG_BYTES = 5000
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

GREEN_MIN_V = 240
GREEN_HSV_LO = np.array([35, 50, 50], dtype=np.uint8)
GREEN_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)
MORPH_KERNEL = np.ones((5, 5), np.uint8)

MIN_GREEN_AREA_FRAC = 0.01
GATE_MIN_VERTICES = 4
GATE_MAX_VERTICES = 8
GATE_ASPECT_MIN = 0.45
GATE_ASPECT_MAX = 2.2
GATE_MIN_SOLIDITY = 0.80
GATE_APPROX_EPS = 0.04

SEARCH_YAWRATE = 10.0
SEARCH_SWEEP_LIMIT_DEG = 90.0
START_BACKUP_DISTANCE_M = 0.35
START_BACKUP_MAX_SPEED = 0.12
RECOVER_YAWRATE = 12.0
APPROACH_TIMEOUT_S = 4.0
RECOVER_DURATION_S = 1.0
PASS_OFFSET_M = 0.20
TARGET_TOL_M = 0.12
MAX_GATES = 5

MAX_XY_SPEED = 0.10
MAX_YAW_RATE_DEG = 20.0


def _load_calibration():
    with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)
    required = ("img_w", "img_h", "fx", "fy", "cx", "cy", "dist_coeffs")
    missing = [key for key in required if key not in calib]
    if missing:
        raise ValueError(f"Missing calibration keys in {CALIBRATION_PATH}: {missing}")
    return calib


def _scaled_calibration(calib, img_w, img_h):
    sx = float(img_w) / float(calib["img_w"])
    sy = float(img_h) / float(calib["img_h"])
    scaled = dict(calib)
    scaled["img_w"] = int(img_w)
    scaled["img_h"] = int(img_h)
    scaled["fx"] = float(calib["fx"]) * sx
    scaled["fy"] = float(calib["fy"]) * sy
    scaled["cx"] = float(calib["cx"]) * sx
    scaled["cy"] = float(calib["cy"]) * sy
    return scaled


BASE_CALIBRATION = _load_calibration()
CAMERA_CALIBRATION = dict(BASE_CALIBRATION)
IMG_WIDTH = int(CAMERA_CALIBRATION["img_w"])
IMG_HEIGHT = int(CAMERA_CALIBRATION["img_h"])
CAMERA_FX = float(CAMERA_CALIBRATION["fx"])
CAMERA_FY = float(CAMERA_CALIBRATION["fy"])
CAMERA_CX = float(CAMERA_CALIBRATION["cx"])
CAMERA_CY = float(CAMERA_CALIBRATION["cy"])
DIST_COEFFS = np.array(CAMERA_CALIBRATION["dist_coeffs"], dtype=np.float64)


def _set_runtime_calibration(img_w, img_h):
    global CAMERA_CALIBRATION, IMG_WIDTH, IMG_HEIGHT, CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, DIST_COEFFS

    if img_w == int(BASE_CALIBRATION["img_w"]) and img_h == int(
        BASE_CALIBRATION["img_h"]
    ):
        CAMERA_CALIBRATION = dict(BASE_CALIBRATION)
        print(f"Radio image size OK: {img_w}x{img_h} matches calibration.json")
    else:
        CAMERA_CALIBRATION = _scaled_calibration(BASE_CALIBRATION, img_w, img_h)
        print(
            f"Radio image size {img_w}x{img_h} differs from calibration.json "
            f"{BASE_CALIBRATION['img_w']}x{BASE_CALIBRATION['img_h']}; scaled calibration in memory"
        )

    IMG_WIDTH = int(CAMERA_CALIBRATION["img_w"])
    IMG_HEIGHT = int(CAMERA_CALIBRATION["img_h"])
    CAMERA_FX = float(CAMERA_CALIBRATION["fx"])
    CAMERA_FY = float(CAMERA_CALIBRATION["fy"])
    CAMERA_CX = float(CAMERA_CALIBRATION["cx"])
    CAMERA_CY = float(CAMERA_CALIBRATION["cy"])
    DIST_COEFFS = np.array(CAMERA_CALIBRATION["dist_coeffs"], dtype=np.float64)


def _camera_matrix():
    return np.array(
        [[CAMERA_FX, 0.0, CAMERA_CX], [0.0, CAMERA_FY, CAMERA_CY], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _undistort_image(rgb_img):
    if DIST_COEFFS.size == 0 or np.allclose(DIST_COEFFS, 0.0):
        return rgb_img
    return cv2.undistort(rgb_img, _camera_matrix(), DIST_COEFFS)


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


def wrap_deg(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def rad_to_deg(angle_rad):
    return float(np.degrees(angle_rad))


def _order_corners(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return np.array(
        [
            pts[np.argmin(s)],
            pts[np.argmax(d)],
            pts[np.argmax(s)],
            pts[np.argmin(d)],
        ],
        dtype=np.float64,
    )


def _quad_corners(contour, approx):
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float64)
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
    ordered = _order_corners(pts)
    if len(np.unique(np.round(ordered, 1), axis=0)) != 4:
        return None
    return ordered


def _pixels_to_world(pixels, cam_pos, cam_rot, img_shape, fov=1.5, real_height=0.4):
    h, w = img_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    cam_pos = np.asarray(cam_pos, dtype=float)
    f = (w / 2.0) / np.tan(fov / 2.0)

    rays_world = []
    for u, v in pixels:
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


def _gate_candidate(contour, img_w, img_h):
    area = float(cv2.contourArea(contour))
    if area < (MIN_GREEN_AREA_FRAC * float(img_w * img_h)):
        return None

    peri = cv2.arcLength(contour, True)
    if peri <= 1e-6:
        return None

    approx = cv2.approxPolyDP(contour, GATE_APPROX_EPS * peri, True)
    n_vert = len(approx)
    if not (GATE_MIN_VERTICES <= n_vert <= GATE_MAX_VERTICES):
        return None

    x, y, bw, bh = cv2.boundingRect(contour)
    if bh <= 0:
        return None
    aspect = bw / float(bh)
    if not (GATE_ASPECT_MIN <= aspect <= GATE_ASPECT_MAX):
        return None

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 1e-6 else 0.0
    if solidity < GATE_MIN_SOLIDITY:
        return None

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
        "corners": _quad_corners(contour, approx),
    }


def _detect_green_gate(rgb_img):
    h, w = rgb_img.shape[:2]
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    threshold = max(int(GREEN_HSV_LO[2]), int(GREEN_MIN_V))
    mask = np.where(gray >= threshold, np.uint8(255), np.uint8(0))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        cand = _gate_candidate(contour, w, h)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return {"found": False, "mask": mask, "candidates": []}

    best = dict(max(candidates, key=lambda c: c["cx"]))
    best["mask"] = mask
    best["candidates"] = candidates
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
        sock.bind(("0.0.0.0", LOCAL_PORT))
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
                _, w, h, _, _, size = struct.unpack(
                    "<BHHBBI", payload[:IMG_HEADER_SIZE]
                )
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
        soi = buffer.find(b"\xff\xd8")
        eoi = buffer.rfind(b"\xff\xd9")
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
        self.frame_ready.emit(img)


class FPVWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crazyflie - simple gate follower")

        self.image_label = QtWidgets.QLabel("Waiting for AI-deck video...")
        self.status_label = QtWidgets.QLabel("Starting...")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self._pos = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self._est = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self._pos_lock = threading.Lock()
        self._vision_lock = threading.Lock()
        self._vision = {
            "found": False,
            "ex": 0.0,
            "ey": 0.0,
            "bbox": None,
            "area": 0.0,
            "corners": None,
        }

        self._log_ready = False
        self._state = "WAIT"
        self._state_t0 = time.monotonic()
        self._last_ctrl_time = time.monotonic()
        self._gates_passed = 0
        self._last_seen_t = 0.0
        self._approach_target = None
        self._gate_heading_deg = 0.0
        self._backup_target = None
        self._search_anchor_yaw = None
        self._search_yaw_dir = 1.0
        self._recover_until = 0.0
        self._initial_z = 0.0
        self._max_xy_speed = MAX_XY_SPEED
        self._max_yaw_rate_deg = MAX_YAW_RATE_DEG

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
        self._timer.setInterval(20)
        self._timer.start()

    def _update_image(self, img):
        if img.ndim == 2:
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            color = img
        color = _undistort_image(color)
        det = _detect_green_gate(color)

        if det.get("found", False):
            corners = det.get("corners", None)
            if corners is not None:
                corners = np.asarray(corners, dtype=float).reshape(-1, 2).tolist()
            with self._vision_lock:
                self._vision = {
                    "found": True,
                    "ex": float(det.get("ex", 0.0)),
                    "ey": float(det.get("ey", 0.0)),
                    "bbox": det.get("bbox", None),
                    "area": float(det.get("area", 0.0)),
                    "corners": corners,
                }
        else:
            with self._vision_lock:
                self._vision = {
                    "found": False,
                    "ex": 0.0,
                    "ey": 0.0,
                    "bbox": None,
                    "area": 0.0,
                    "corners": None,
                }

        disp = color.copy()
        sel_bbox = det.get("bbox")
        for cand in det.get("candidates", []):
            is_sel = cand["bbox"] == sel_bbox
            ccolor = (0, 255, 0) if is_sel else (255, 255, 0)
            cthick = 2 if is_sel else 1
            cx_b, cy_b, bw_b, bh_b = cand["bbox"]
            cv2.rectangle(
                disp, (cx_b, cy_b), (cx_b + bw_b, cy_b + bh_b), ccolor, cthick
            )
            if cand.get("approx") is not None:
                cv2.polylines(disp, [cand["approx"]], True, ccolor, cthick)

        if det.get("found", False) and sel_bbox is not None:
            cx, cy = int(round(det["cx"])), int(round(det["cy"]))
            cv2.circle(disp, (cx, cy), 4, (255, 0, 0), -1)

        h, w = disp.shape[:2]
        cv2.drawMarker(disp, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(
            disp,
            f'state={self._state} gates={self._gates_passed}/{MAX_GATES} found={int(det.get("found", False))}',
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        rgb = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
        qimg = QtGui.QImage(
            rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_BGR888
        ).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

    def _gate_to_world(self, corners):
        if corners is None:
            return None

        with self._pos_lock:
            yaw_r = np.deg2rad(self._est["yaw"])
            cam_pos = np.array(
                [self._est["x"], self._est["y"], self._est["z"]], dtype=float
            )

        cam_rot = np.array(
            [
                [np.cos(yaw_r), -np.sin(yaw_r), 0.0],
                [np.sin(yaw_r), np.cos(yaw_r), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        world_corners = _pixels_to_world(
            corners, cam_pos, cam_rot, (IMG_HEIGHT, IMG_WIDTH, 3)
        )
        if world_corners is None:
            return None

        gate_center = np.mean(world_corners, axis=0)
        gate_left = np.mean(world_corners[[0, 3]], axis=0)
        gate_right = np.mean(world_corners[[1, 2]], axis=0)
        gate_angle = np.arctan2(
            gate_right[1] - gate_left[1], gate_right[0] - gate_left[0]
        )
        return gate_center, gate_angle

    def _gate_target_after_pass(self, gate_center, gate_angle):
        forward = np.array(
            [np.cos(gate_angle + np.pi / 2.0), np.sin(gate_angle + np.pi / 2.0), 0.0],
            dtype=float,
        )
        return gate_center + PASS_OFFSET_M * forward

    def _ramp_position(self, target, dt):
        current = np.array(
            [self._pos["x"], self._pos["y"], self._pos["z"]], dtype=float
        )
        target_pos = np.array(target[:3], dtype=float)
        delta = target_pos - current

        max_xy_step = self._max_xy_speed * max(float(dt), 1e-3)
        xy_step = delta[:2]
        xy_norm = float(np.linalg.norm(xy_step))
        if xy_norm > max_xy_step:
            xy_step = xy_step * (max_xy_step / max(xy_norm, 1e-6))

        next_pos = current + np.array([xy_step[0], xy_step[1], 0.0], dtype=float)
        self._pos["x"] = float(next_pos[0])
        self._pos["y"] = float(next_pos[1])
        self._pos["z"] = float(self._initial_z)

        target_yaw = wrap_deg(float(target[3]))
        yaw_current = wrap_deg(self._pos["yaw"])
        yaw_diff = wrap_deg(target_yaw - yaw_current)
        max_yaw_step = self._max_yaw_rate_deg * max(float(dt), 1e-3)
        yaw_step = float(np.clip(yaw_diff, -max_yaw_step, max_yaw_step))
        self._pos["yaw"] = wrap_deg(yaw_current + yaw_step)

    def _distance_to_target(self, target):
        dx = float(target[0]) - self._est["x"]
        dy = float(target[1]) - self._est["y"]
        return float(np.hypot(dx, dy))

    def _send_setpoint(self):
        now = time.monotonic()
        dt = float(np.clip(now - self._last_ctrl_time, 0.02, 0.3))
        self._last_ctrl_time = now

        if not self._log_ready:
            return

        with self._pos_lock:
            est = dict(self._est)
        with self._vision_lock:
            vision = dict(self._vision)

        if self._state == "DONE":
            self.cf.commander.send_stop_setpoint()
            return

        if self._state == "STOP":
            self.cf.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
            return

        if self._state == "WAIT":
            heading_rad = np.deg2rad(float(est["yaw"]))
            self._pos["x"] = est["x"]
            self._pos["y"] = est["y"]
            self._initial_z = float(est["z"])
            self._pos["z"] = self._initial_z
            self._pos["yaw"] = float(est["yaw"])
            self._backup_target = np.array(
                [
                    est["x"] - START_BACKUP_DISTANCE_M * np.cos(heading_rad),
                    est["y"] - START_BACKUP_DISTANCE_M * np.sin(heading_rad),
                    self._initial_z,
                    est["yaw"],
                ],
                dtype=float,
            )
            self._state = "BACKUP"
            self._state_t0 = now

        if self._state == "BACKUP":
            if self._backup_target is None:
                self._state = "SEARCH"
            else:
                self._max_xy_speed = START_BACKUP_MAX_SPEED
                self._ramp_position(self._backup_target, dt)
                self._pos["yaw"] = float(self._backup_target[3])

                if self._distance_to_target(self._backup_target) <= 0.05:
                    self._backup_target = None
                    self._search_anchor_yaw = float(self._pos["yaw"])
                    self._search_yaw_dir = 1.0
                    self._state = "SEARCH"
                    self._state_t0 = now

        if self._state == "SEARCH":
            if self._search_anchor_yaw is None:
                self._search_anchor_yaw = float(self._pos["yaw"])

            relative_yaw = wrap_deg(self._pos["yaw"] - self._search_anchor_yaw)
            next_relative_yaw = (
                relative_yaw + self._search_yaw_dir * SEARCH_YAWRATE * dt
            )
            if next_relative_yaw >= SEARCH_SWEEP_LIMIT_DEG:
                next_relative_yaw = SEARCH_SWEEP_LIMIT_DEG
                self._search_yaw_dir = -1.0
            elif next_relative_yaw <= -SEARCH_SWEEP_LIMIT_DEG:
                next_relative_yaw = -SEARCH_SWEEP_LIMIT_DEG
                self._search_yaw_dir = 1.0

            self._pos["yaw"] = wrap_deg(self._search_anchor_yaw + next_relative_yaw)
            self._pos["x"] = est["x"]
            self._pos["y"] = est["y"]
            self._pos["z"] = self._initial_z

            if vision["found"] and vision["corners"] is not None:
                gate_info = self._gate_to_world(vision["corners"])
                if gate_info is not None:
                    gate_center, gate_angle = gate_info
                    self._gate_heading_deg = wrap_deg(
                        rad_to_deg(gate_angle + np.pi / 2.0)
                    )
                    target = self._gate_target_after_pass(gate_center, gate_angle)
                    self._approach_target = np.array(
                        [target[0], target[1], self._initial_z, self._gate_heading_deg],
                        dtype=float,
                    )
                    self._state = "APPROACH"
                    self._state_t0 = now
                    self._last_seen_t = now
                    print(
                        f"[CTRL] SEARCH -> APPROACH gate={self._gates_passed + 1} "
                        f"target=({self._approach_target[0]:.2f},{self._approach_target[1]:.2f},{self._approach_target[2]:.2f})",
                        flush=True,
                    )

        elif self._state == "APPROACH":
            if vision["found"] and vision["corners"] is not None:
                self._last_seen_t = now
            elif now - self._last_seen_t > APPROACH_TIMEOUT_S:
                print("[CTRL] APPROACH timeout -> SEARCH", flush=True)
                self._approach_target = None
                self._search_anchor_yaw = float(self._pos["yaw"])
                self._search_yaw_dir = 1.0
                self._state = "SEARCH"
                self._state_t0 = now
                self.cf.commander.send_position_setpoint(
                    self._pos["x"], self._pos["y"], self._pos["z"], self._pos["yaw"]
                )
                return

            if self._approach_target is not None:
                self._ramp_position(self._approach_target, dt)
                self._pos["yaw"] = wrap_deg(self._gate_heading_deg)

                if self._distance_to_target(self._approach_target) <= TARGET_TOL_M:
                    self._state = "RECOVER"
                    self._state_t0 = now
                    self._recover_until = now + RECOVER_DURATION_S
                    print("[CTRL] APPROACH reached -> RECOVER", flush=True)

        elif self._state == "RECOVER":
            if self._approach_target is not None:
                self._pos["x"] = float(self._approach_target[0])
                self._pos["y"] = float(self._approach_target[1])
                self._pos["z"] = float(self._initial_z)
            self._pos["yaw"] = wrap_deg(self._pos["yaw"] + RECOVER_YAWRATE * dt)

            if now >= self._recover_until:
                self._gates_passed += 1
                self._approach_target = None
                print(
                    f"[CTRL] RECOVER complete -> SEARCH (gates={self._gates_passed})",
                    flush=True,
                )
                if self._gates_passed >= MAX_GATES:
                    self._state = "DONE"
                    self.cf.commander.send_stop_setpoint()
                    self._timer.stop()
                    return
                self._search_anchor_yaw = float(self._pos["yaw"])
                self._search_yaw_dir = 1.0
                self._state = "SEARCH"
                self._state_t0 = now

        self._max_xy_speed = MAX_XY_SPEED

        self.cf.commander.send_position_setpoint(
            float(self._pos["x"]),
            float(self._pos["y"]),
            float(self._pos["z"]),
            float(self._pos["yaw"]),
        )

    def _set_status(self, text):
        QtCore.QMetaObject.invokeMethod(
            self.status_label,
            "setText",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text),
        )

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
                self._pos["x"] = self._est["x"]
                self._pos["y"] = self._est["y"]
                self._pos["z"] = self._est["z"]
                self._pos["yaw"] = self._est["yaw"]
                self._initial_z = float(self._est["z"])
                self._log_ready = True
                print(
                    f"First estimate: x={self._est['x']:.2f}, y={self._est['y']:.2f}, "
                    f"z={self._est['z']:.2f}, yaw={self._est['yaw']:.1f}"
                )

    def _connected(self, uri):
        self._set_status(f"Connected to {uri}")
        self._setup_log()

    def _disconnected(self, uri):
        print("Disconnected")
        sys.exit(1)

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._state = "STOP"
            self._set_status("STOP: zero-velocity safety brake")
        elif event.key() == QtCore.Qt.Key.Key_Space:
            self._state = "DONE"
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    def closeEvent(self, event):
        self._timer.stop()
        if hasattr(self, "_log_cfg"):
            self._log_cfg.stop()
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()
