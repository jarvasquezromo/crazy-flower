#!/usr/bin/env python3
"""Crazyflie gate scan (lap 1 only) with FPV + gate map visualization."""
import contextlib
import logging
import os
import socket
import struct
import sys
import threading
import time
import warnings

import numpy as np
import cv2
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt6 import QtCore, QtWidgets, QtGui

from gate_map import GateZoneMap, GateMapWidget

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore', message='.*TYPE_HOVER_LEGACY.*')
warnings.filterwarnings('ignore', message='.*supervisor subsystem requires CRTP.*')

URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E705')
AIDECK_IP = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b'FER'

COURSE_CENTER_X = 1.15
COURSE_CENTER_Y = 0.0

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
MIN_JPEG_BYTES = 5000
DEFAULT_IMG_W = 324
DEFAULT_IMG_H = 244

MIN_GATE_V = 240
MIN_GATE_AREA_FRAC = 0.01
GATE_MIN_VERTICES = 4
GATE_MAX_VERTICES = 8
GATE_ASPECT_MIN = 0.45
GATE_ASPECT_MAX = 2.2
GATE_MIN_SOLIDITY = 0.80
GATE_APPROX_EPS = 0.04
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


IMG_WIDTH = DEFAULT_IMG_W
IMG_HEIGHT = DEFAULT_IMG_H


def wrap_deg(angle):
    """Wrap an angle to (-180, 180] degrees."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def rad_to_deg(angle_rad):
    return float(np.degrees(angle_rad))


class Surveyer:
    def __init__(self):
        self.target = None
        self.gate_center = None
        self.heading = 0.0
        self.mapping_progress = 0
        self.gate_to_go = 0
        self.gates = []
        self.detected_gate_ids = []
        self.detected_gate_angles = []
        self.pass_through_distance = 0.5
        self.stabilization_wait_duration = 6
        self.stabilization_start_time = None
        self.stabilization_context = None
        self.mapping_radii = [0.75, 1.1, 1.4]
        self.mapping_radius_idx = 0
        self.height_offset = 0.0
        self.angle_offset = 0.0
        self._last_mapping_key = None
        self._hold_start_time = None
        self._hold_context = None

    def _reset_stabilization(self):
        self.stabilization_start_time = None
        self.stabilization_context = None

    def _hold_elapsed(self, sensor_data, context_key, duration_s):
        if self._hold_context != context_key:
            self._hold_context = context_key
            self._hold_start_time = float(sensor_data['t'])
            return False
        if self._hold_start_time is None:
            self._hold_start_time = float(sensor_data['t'])
            return False
        return (float(sensor_data['t']) - self._hold_start_time) >= duration_s

    def _stable_at_target(self, sensor_data, control_command, context_key):
        if not self.reached_target(sensor_data, control_command, threshold=0.18):
            self._reset_stabilization()
            return False

        if self.stabilization_context != context_key:
            self.stabilization_context = context_key
            self.stabilization_start_time = float(sensor_data['t'])
            return False

        if self.stabilization_start_time is None:
            self.stabilization_start_time = float(sensor_data['t'])
            return False

        return (float(sensor_data['t']) - self.stabilization_start_time) >= self.stabilization_wait_duration

    def record_gate_observation(self, gate_id, gate_position, gate_angle):
        if gate_id in self.detected_gate_ids:
            return

        self.detected_gate_ids.append(gate_id)
        self.detected_gate_angles.append(float(gate_angle))

    def _order_corners(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        s = pts[:, 0] + pts[:, 1]
        d = pts[:, 0] - pts[:, 1]
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmax(d)],
            pts[np.argmax(s)],
            pts[np.argmin(d)],
        ], dtype=np.float64)

    def _quad_corners(self, contour, approx):
        if len(approx) == 4:
            pts = approx.reshape(-1, 2).astype(np.float64)
        else:
            pts = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
        ordered = self._order_corners(pts)
        if len(np.unique(np.round(ordered, 1), axis=0)) != 4:
            return None
        return ordered

    def _gate_candidate(self, contour, img_w, img_h):
        area = float(cv2.contourArea(contour))
        if area < (MIN_GATE_AREA_FRAC * float(img_w * img_h)):
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

        corners = self._quad_corners(contour, approx)
        if corners is None:
            return None

        return {"cx": cx, "cy": cy, "corners": corners}

    def _detect_gate_outline(self, camera_data):
        h, w = camera_data.shape[:2]
        gray = cv2.cvtColor(camera_data, cv2.COLOR_BGR2GRAY)
        mask = np.where(gray >= int(MIN_GATE_V), np.uint8(255), np.uint8(0))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            cand = self._gate_candidate(contour, w, h)
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        return max(candidates, key=lambda c: c["cx"])

    def pixels_to_world(self, pixels, cam_pos, cam_rot, img_shape, fov=1.5, real_height=0.4):
        h, w, _ = img_shape
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

    def quaternion_to_rotation_matrix(self, q_x, q_y, q_z, q_w):
        q = np.array([q_x, q_y, q_z, q_w], dtype=float)
        norm_q = np.linalg.norm(q)
        if norm_q < 1e-12:
            return np.eye(3)
        q_x, q_y, q_z, q_w = q / norm_q

        return np.array([
            [1 - 2 * q_y**2 - 2 * q_z**2, 2 * q_x * q_y - 2 * q_z * q_w, 2 * q_x * q_z + 2 * q_y * q_w],
            [2 * q_x * q_y + 2 * q_z * q_w, 1 - 2 * q_x**2 - 2 * q_z**2, 2 * q_y * q_z - 2 * q_x * q_w],
            [2 * q_x * q_z - 2 * q_y * q_w, 2 * q_y * q_z + 2 * q_x * q_w, 1 - 2 * q_x**2 - 2 * q_y**2],
        ], dtype=float)

    def detect_gate(self, camera_data, sensor_data, check=False):
        candidate = self._detect_gate_outline(camera_data)
        if candidate is None:
            return None, None

        corners = candidate["corners"]
        if check:
            print("Checking gate detection: corners =", corners)

        position = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']]
        rotation_matrix = self.quaternion_to_rotation_matrix(
            sensor_data['q_x'], sensor_data['q_y'], sensor_data['q_z'], sensor_data['q_w']
        )
        world_corners = self.pixels_to_world(corners, position, rotation_matrix, camera_data.shape)
        if world_corners is None:
            return None, None

        gate_center = np.mean(world_corners, axis=0)
        gate_left = np.mean(world_corners[[0, 3]], axis=0)
        gate_right = np.mean(world_corners[[1, 2]], axis=0)
        gate_angle = np.arctan2(gate_right[1] - gate_left[1], gate_right[0] - gate_left[0])
        print(gate_center, gate_angle)
        return gate_center, gate_angle

    def fly_to_mapping_position(self, position_id, radius=None):
        if radius is None:
            radius = self.mapping_radii[self.mapping_radius_idx]

        angle = position_id * (np.pi / 3) + np.pi / 12 - np.pi
        target_y = COURSE_CENTER_Y + np.sin(angle) * radius
        target_x = COURSE_CENTER_X + np.cos(angle) * radius
        target_z = 1.1
        # Internal geometry is in radians, but Crazyflie position-setpoint yaw is degrees.
        target_yaw_rad = -angle + np.pi / 2 + np.pi / 6 + np.pi / 12

        target_z += self.height_offset
        target_yaw_rad += self.angle_offset

        target_yaw_deg = wrap_deg(rad_to_deg(target_yaw_rad))
        return [target_x, target_y, target_z, target_yaw_deg]

    def _advance_mapping_radius(self):
        self.mapping_radius_idx = (self.mapping_radius_idx + 1) % len(self.mapping_radii)
        # Keep search deterministic and stable during hardware testing.
        self.height_offset = 0.0
        self.angle_offset = 0.0

    @staticmethod
    def segment_from_xy(x, y, center_x=COURSE_CENTER_X, center_y=COURSE_CENTER_Y):
        dx = float(x) - center_x
        dy = float(y) - center_y

        angle_deg = np.degrees(np.arctan2(dy, dx))
        aligned_deg = (angle_deg + 15.0) % 360.0
        raw_segment = int(np.floor(aligned_deg / 30.0))

        if raw_segment % 2 == 1:
            return -1

        return ((raw_segment + 4) % 12) // 2

    @staticmethod
    def reached_target(sensor_data, control_command, threshold=0.25):
        dx = control_command[0] - sensor_data['x_global']
        dy = control_command[1] - sensor_data['y_global']
        dz = control_command[2] - sensor_data['z_global']
        return np.sqrt(dx**2 + dy**2 + dz**2) < threshold

    def fly_to_target(self):
        target_x, target_y, target_z = self.target
        target_yaw_deg = wrap_deg(rad_to_deg(self.heading + np.pi / 2))
        return [target_x, target_y, target_z, target_yaw_deg]

    def fly_before_gate(self):
        if self.gate_center is None:
            target_x, target_y, target_z = self.target
        else:
            target_x, target_y, target_z = self.gate_center
        target_x -= 0.6 * np.cos(self.heading + np.pi / 2)
        target_y -= 0.6 * np.sin(self.heading + np.pi / 2)
        target_yaw_deg = wrap_deg(rad_to_deg(self.heading + np.pi / 2))
        return [target_x, target_y, target_z, target_yaw_deg]

    def acquire_gate(self, sensor_data, camera_data, check=False):
        gate_position, gate_angle = self.detect_gate(camera_data, sensor_data, check)
        if gate_position is not None:
            self.gate_center = np.asarray(gate_position, dtype=float)
            self.heading = gate_angle
            forward = np.array([
                np.cos(self.heading + np.pi / 2),
                np.sin(self.heading + np.pi / 2),
                0.0,
            ], dtype=float)
            self.target = self.gate_center + self.pass_through_distance * forward
            return True
        return False

    def map_gate(self, sensor_data, camera_data):
        gate_id = self.gate_to_go
        control_command = [
            sensor_data['x_global'],
            sensor_data['y_global'],
            max(sensor_data['z_global'], 1.0),
            sensor_data['yaw'],
        ]

        if self.mapping_progress == 0:
            radius = self.mapping_radii[self.mapping_radius_idx]
            control_command = self.fly_to_mapping_position(gate_id, radius=radius)
            mapping_key = (gate_id, self.mapping_radius_idx)
            if mapping_key != self._last_mapping_key:
                self._reset_stabilization()
                self._last_mapping_key = mapping_key
            if self._stable_at_target(sensor_data, control_command, (gate_id, self.mapping_progress, self.mapping_radius_idx)) and self.acquire_gate(sensor_data, camera_data):
                observed_segment = self.segment_from_xy(self.gate_center[0], self.gate_center[1])

                if observed_segment == gate_id:
                    self.mapping_progress += 1
                    self._hold_context = None
                else:
                    self.gate_center = None
                    self._advance_mapping_radius()
                    self._hold_context = None

                self._reset_stabilization()
            elif self._hold_elapsed(sensor_data, (gate_id, self.mapping_progress, self.mapping_radius_idx), 10.0):
                self._advance_mapping_radius()
                self._reset_stabilization()
                self._hold_context = None

        elif self.mapping_progress == 1:
            control_command = self.fly_before_gate()
            if self._stable_at_target(sensor_data, control_command, (gate_id, self.mapping_progress)) and self.acquire_gate(sensor_data, camera_data, check=True):
                center = np.asarray(self.gate_center, dtype=float)
                self.gates += [[center[0], center[1], center[2]]]
                observed_segment = self.segment_from_xy(center[0], center[1])
                self.record_gate_observation(observed_segment, center, self.heading)
                self.mapping_progress += 1
                self._reset_stabilization()
                self._hold_context = None
            elif self._hold_elapsed(sensor_data, (gate_id, self.mapping_progress), 10.0):
                self.mapping_progress = 0
                self.gate_center = None
                self._advance_mapping_radius()
                self._reset_stabilization()
                self._hold_context = None

        elif self.mapping_progress == 2:
            control_command = self.fly_to_target()
            if self.reached_target(sensor_data, control_command):
                self.mapping_progress = 0
                self.gate_to_go += 1
                self.gate_center = None
                self.mapping_radius_idx = 0
                self._reset_stabilization()

        return control_command


class MyAssignment:
    def __init__(self):
        self.home = None
        self.surveyer = Surveyer()
        self.home_wait_altitude = 0.9

    def compute_command(self, sensor_data, camera_data, dt):
        if sensor_data['z_global'] < 0.49:
            return [sensor_data['x_global'], sensor_data['y_global'], 1, sensor_data['yaw']]

        if self.home is None:
            self.home = [sensor_data['x_global'], sensor_data['y_global'], self.home_wait_altitude]

        if self.surveyer.gate_to_go < 5:
            return self.surveyer.map_gate(sensor_data, camera_data)

        return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]


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
                        print(f"Radio image size: {self._expected_w}x{self._expected_h}")
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
        self._pos = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        # State estimator readout (updated by log callback at 50 Hz).
        self._est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self._pos_lock = threading.Lock()

        self._controller = MyAssignment()
        self._last_frame = None
        self._last_ctrl_time = time.monotonic()
        self._log_ready = False
        self._cmd_pos = None
        # Speeds are true rates per second, not per 20 ms control tick.
        # The previous _xy_step=0.02 per tick was about 1.0 m/s at 50 Hz.
        self._max_xy_speed = 0.045       # m/s, very slow horizontal motion
        self._max_z_speed = 0.025        # m/s, very slow vertical motion
        self._max_yaw_rate_deg = 18.0    # deg/s, slow yaw motion
        self._ramp_target = None
        self._ramp_pos_eps = 1e-3
        self._ramp_yaw_eps = 0.5  # degrees
        self._debug_last = 0.0
        self._debug_wait_last = 0.0
        self._debug_frame_last = 0.0
        self._debug_cmd_last = 0.0

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
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img
        self._last_frame = bgr

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QtGui.QImage(rgb.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888).copy()
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    def _send_setpoint(self):
        now = time.monotonic()
        dt = float(np.clip(now - self._last_ctrl_time, 0.02, 0.3))
        self._last_ctrl_time = now

        if not self._log_ready:
            if now - self._debug_wait_last >= 1.0:
                self._debug_wait_last = now
                print("Waiting for stateEstimate log...")
            return

        camera_data = self._last_frame
        if camera_data is None:
            if now - self._debug_frame_last >= 1.0:
                self._debug_frame_last = now
                print("Waiting for camera frames from AI-deck...")
            return

        with self._pos_lock:
            est = dict(self._est)

        yaw_rad = np.deg2rad(est['yaw'])
        sensor_data = {
            't': now,
            'x_global': float(est['x']),
            'y_global': float(est['y']),
            'z_global': float(est['z']),
            'yaw': float(est['yaw']),
            'q_x': 0.0,
            'q_y': 0.0,
            'q_z': float(np.sin(yaw_rad / 2.0)),
            'q_w': float(np.cos(yaw_rad / 2.0)),
        }

        cmd = self._controller.compute_command(sensor_data, camera_data, dt)
        if not np.all(np.isfinite(cmd)):
            if now - self._debug_cmd_last >= 1.0:
                self._debug_cmd_last = now
                print(f"Invalid cmd (non-finite): {cmd}")
            return
        self._pos['x'], self._pos['y'], self._pos['z'], self._pos['yaw'] = self._ramp_setpoint(cmd, dt)

        if now - self._debug_last >= 0.5:
            self._debug_last = now
            mapping_progress = int(self._controller.surveyer.mapping_progress)
            gate_to_go = int(self._controller.surveyer.gate_to_go)
            print(
                "gate={} prog={} | pos=({:.2f},{:.2f},{:.2f}) yaw={:.1f} | cmd=({:.2f},{:.2f},{:.2f},{:.1f}) | sp=({:.2f},{:.2f},{:.2f},{:.1f})".format(
                    gate_to_go,
                    mapping_progress,
                    est['x'],
                    est['y'],
                    est['z'],
                    est['yaw'],
                    float(cmd[0]),
                    float(cmd[1]),
                    float(cmd[2]),
                    float(cmd[3]),
                    self._pos['x'],
                    self._pos['y'],
                    self._pos['z'],
                    self._pos['yaw'],
                )
            )

        # Update the top-down map using the latest estimate.
        gate_index = min(self._controller.surveyer.gate_to_go + 1, 5)
        est_gate = None
        est_in_zone = None
        if self._controller.surveyer.gate_center is not None:
            est_gate = (
                float(self._controller.surveyer.gate_center[0]),
                float(self._controller.surveyer.gate_center[1]),
                float(self._controller.surveyer.gate_center[2]),
            )
            snapped = self._zone_map.validate_and_snap(est_gate, gate_index)
            est_in_zone = snapped is not None
            if snapped is not None:
                est_gate = snapped

        self.gate_map.update_state(
            drone=(est['x'], est['y'], est['yaw']),
            est_gate=est_gate,
            target_gate=gate_index,
            est_in_zone=est_in_zone,
        )

        self.cf.commander.send_position_setpoint(
            float(self._pos['x']),
            float(self._pos['y']),
            float(self._pos['z']),
            float(self._pos['yaw']),
        )

    def _ramp_setpoint(self, cmd, dt):
        """Rate-limit position and yaw setpoints.

        Convention in this function:
        - x, y, z are metres in the world frame.
        - yaw is ALWAYS degrees, because stabilizer.yaw and
          send_position_setpoint(..., yaw) use degrees.
        """
        target = [float(cmd[0]), float(cmd[1]), float(cmd[2]), wrap_deg(cmd[3])]

        if self._cmd_pos is None:
            self._cmd_pos = list(target)
            self._ramp_target = list(target)
            return list(self._cmd_pos)

        if self._ramp_target is None:
            self._ramp_target = list(target)
        else:
            delta_pos = np.linalg.norm(np.array(target[:3]) - np.array(self._ramp_target[:3]))
            yaw_delta = wrap_deg(target[3] - self._ramp_target[3])
            if delta_pos > self._ramp_pos_eps or abs(yaw_delta) > self._ramp_yaw_eps:
                self._ramp_target = list(target)

        current = np.array(self._cmd_pos[:3], dtype=float)
        target_pos = np.array(self._ramp_target[:3], dtype=float)
        delta = target_pos - current

        max_xy_step = self._max_xy_speed * max(float(dt), 1e-3)
        max_z_step = self._max_z_speed * max(float(dt), 1e-3)

        xy_step = delta[:2]
        xy_norm = float(np.linalg.norm(xy_step))
        if xy_norm > max_xy_step:
            xy_step = xy_step * (max_xy_step / max(xy_norm, 1e-6))
        z_step = float(np.clip(delta[2], -max_z_step, max_z_step))

        next_pos = current + np.array([xy_step[0], xy_step[1], z_step], dtype=float)

        yaw_target = wrap_deg(self._ramp_target[3])
        yaw_current = wrap_deg(self._cmd_pos[3])
        yaw_diff = wrap_deg(yaw_target - yaw_current)
        max_yaw_step = self._max_yaw_rate_deg * max(float(dt), 1e-3)
        yaw_step = float(np.clip(yaw_diff, -max_yaw_step, max_yaw_step))
        next_yaw = wrap_deg(yaw_current + yaw_step)

        self._cmd_pos = [float(next_pos[0]), float(next_pos[1]), float(next_pos[2]), float(next_yaw)]
        return list(self._cmd_pos)

    def _set_status(self, text):
        QtCore.QMetaObject.invokeMethod(
            self.status_label, 'setText',
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, text))

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
        print('Logging: stateEstimate.x/y/z, stabilizer.yaw')

    def _on_log(self, _ts, data, _lc):
        with self._pos_lock:
            self._est['x'] = data['stateEstimate.x']
            self._est['y'] = data['stateEstimate.y']
            self._est['z'] = data['stateEstimate.z']
            self._est['yaw'] = data['stabilizer.yaw']
            if not self._log_ready:
                self._pos['x'] = self._est['x']
                self._pos['y'] = self._est['y']
                self._pos['z'] = self._est['z']
                self._pos['yaw'] = self._est['yaw']
                self._log_ready = True
                print(
                    f"First Lighthouse position: x={self._est['x']:.2f} "
                    f"y={self._est['y']:.2f} z={self._est['z']:.2f} "
                    f"yaw={self._est['yaw']:.1f} — leaving WAIT"
                )

    def _connected(self, uri):
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
