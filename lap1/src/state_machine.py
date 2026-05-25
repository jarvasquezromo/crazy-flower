"""Drone state machine: WAIT → TAKEOFF → SEARCH → CHASE → PUSH → DONE."""
import threading
import time

import cv2
import numpy as np

from .constants import (
    TAKEOFF_START_HEIGHT, TAKEOFF_RATE, SEARCH_HEIGHT,
    SEARCH_YAWRATE, FORWARD_SPEED, PUSH_DURATION_S, MAX_GATES,
    MIN_HEIGHT, MAX_HEIGHT, GATE_PHYS_W, TRAJ_N_STEPS, TRAJ_OVERSHOOT,
    WAYPOINT_TOL, PASS_AREA_FRAC, CHASE_TIMEOUT, GATE_EMA_ALPHA,
    GATE_LOCK_MIN_OBS, GATE_LOCK_MAX_OBS, GATE_LOCK_MAX_AGE_S,
    GATE_LOCK_MAX_SPREAD_XY, CHASE_UPDATE_MAX_JUMP_XY,
    PNP_RANGE_MIN, PNP_RANGE_MAX, PNP_LATERAL_MAX,
    USE_MONO_HEIGHT_PROJECTION,
)
from .gate_projection import GateTriangulator


class GateStateMachine:
    """Pure flight-logic state machine, decoupled from GUI/Crazyflie."""

    def __init__(self, calib=None):
        self.pos = {'x': 0.0, 'y': 0.0, 'z': TAKEOFF_START_HEIGHT, 'yaw': 0.0}
        self.est = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0}
        self.pos_lock = threading.Lock()

        self.state = "WAIT"
        self.log_ready = False
        self.gates_passed = 0

        self._gate_world = None
        self._reset_gate_tracking()
        self._traj = []
        self._traj_idx = 0
        self._push_target = (0.0, 0.0, TAKEOFF_START_HEIGHT)
        self._state_t0 = time.monotonic()
        self._last_ctrl_time = time.monotonic()

        # Triangulator (used when USE_MONO_HEIGHT_PROJECTION is True)
        self._triangulator = GateTriangulator(calib) if (USE_MONO_HEIGHT_PROJECTION and calib) else None

    def seed_position(self, x, y, z, yaw):
        """Seed commanded pos from first Lighthouse estimate."""
        self.pos = {'x': x, 'y': y, 'z': z, 'yaw': yaw}
        self.log_ready = True

    def _reset_gate_tracking(self):
        """Clear gate observations and triangulator buffer."""
        if hasattr(self, "_triangulator") and self._triangulator is not None:
            self._triangulator.reset()

    def update(self, vision, calib, cam_mtx):
        """Advance state machine one tick. Call at ~50 Hz.

        Args:
            vision: dict with found, ex, ey, bbox, area, corners
            calib: dict with fx, fy, img_w, img_h
            cam_mtx: 3x3 numpy camera matrix
        Returns:
            True if flight is done (DONE state reached).
        """
        now = time.monotonic()
        dt = float(np.clip(now - self._last_ctrl_time, 0.02, 0.3))
        self._last_ctrl_time = now

        if self.state in ("STOP", "DONE"):
            return self.state == "DONE"
        if not self.log_ready:
            return False

        found = vision.get("found", False)
        ex = vision.get("ex", 0.0)
        ey = vision.get("ey", 0.0)
        bbox = vision.get("bbox")
        area = vision.get("area", 0.0)
        corners = vision.get("corners")

        with self.pos_lock:
            est = dict(self.est)

        img_area = float(calib['img_w'] * calib['img_h'])

        if self.state == "WAIT":
            self.state = "TAKEOFF"
            self._state_t0 = now

        if self.state == "TAKEOFF":
            self.pos['z'] = float(min(self.pos['z'] + TAKEOFF_RATE * dt, SEARCH_HEIGHT))
            if self.pos['z'] >= SEARCH_HEIGHT - 1e-3:
                self.state = "SEARCH"
                self._state_t0 = now

        elif self.state == "SEARCH":
            self.pos['yaw'] += SEARCH_YAWRATE * dt
            if found and bbox is not None:
                gw = self._gate_to_world(ex, ey, bbox, corners, calib, cam_mtx)
                locked = self._try_lock_gate(gw, now)
                if locked is not None:
                    self._gate_world = locked
                    self._traj = self._build_traj(locked)
                    self._traj_idx = 0
                    self.state = "CHASE"
                    self._state_t0 = now
                    self._reset_gate_tracking()
            else:
                self._reset_gate_tracking()

        elif self.state == "CHASE":
            if (now - self._state_t0) > CHASE_TIMEOUT:
                self._gate_world = None
                self._traj = []
                self._reset_gate_tracking()
                self.state = "SEARCH"
                self._state_t0 = now
            else:
                if found and bbox is not None:
                    new_gw = self._gate_to_world(ex, ey, bbox, corners, calib, cam_mtx)
                    if self._valid_chase_update(new_gw):
                        ox, oy, oz = self._gate_world
                        nx, ny, nz = new_gw
                        self._gate_world = (
                            ox + GATE_EMA_ALPHA * (nx - ox),
                            oy + GATE_EMA_ALPHA * (ny - oy),
                            oz + GATE_EMA_ALPHA * (nz - oz),
                        )
                        if self._traj_idx == 0:
                            self._traj = self._build_traj(self._gate_world)

                if area / img_area > PASS_AREA_FRAC:
                    yaw_r = np.deg2rad(est['yaw'])
                    push_dist = FORWARD_SPEED * PUSH_DURATION_S + TRAJ_OVERSHOOT
                    self._push_target = (
                        est['x'] + push_dist * np.cos(yaw_r),
                        est['y'] + push_dist * np.sin(yaw_r),
                        self.pos['z'],
                    )
                    self.state = "PUSH"
                    self._state_t0 = now
                elif self._traj:
                    wp = self._traj[self._traj_idx]
                    self.pos['x'], self.pos['y'], self.pos['z'] = wp
                    gx, gy, _ = self._gate_world
                    self.pos['yaw'] = float(np.degrees(
                        np.arctan2(gy - est['y'], gx - est['x'])))
                    dist_wp = np.hypot(est['x'] - wp[0], est['y'] - wp[1])
                    if dist_wp < WAYPOINT_TOL and self._traj_idx < len(self._traj) - 1:
                        self._traj_idx += 1

        elif self.state == "PUSH":
            self.pos['x'], self.pos['y'], self.pos['z'] = self._push_target
            if (now - self._state_t0) >= PUSH_DURATION_S:
                self.gates_passed += 1
                self._gate_world = None
                self._traj = []
                self._reset_gate_tracking()
                if self.gates_passed >= MAX_GATES:
                    self.state = "DONE"
                    return True
                self.state = "SEARCH"
                self._state_t0 = now

        self.pos['z'] = float(np.clip(self.pos['z'], MIN_HEIGHT, MAX_HEIGHT))
        return False

    # ─── Gate projection ─────────────────────────────────────────────────

    def _gate_camera_pose(self, corners, cam_mtx):
        if corners is None:
            return None
        img_pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        if img_pts.shape[0] != 4:
            return None
        h = 0.5 * GATE_PHYS_W
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
        try:
            ok, _rvec, tvec = cv2.solvePnP(obj, img_pts, cam_mtx, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except cv2.error:
            return None
        if not ok:
            return None
        t = tvec.reshape(3)
        if not np.all(np.isfinite(t)):
            return None
        if not (PNP_RANGE_MIN <= t[2] <= PNP_RANGE_MAX):
            return None
        if abs(t[0]) > PNP_LATERAL_MAX or abs(t[1]) > PNP_LATERAL_MAX:
            return None
        return float(t[0]), float(t[1]), float(t[2])

    def _gate_to_world(self, ex, ey, bbox, corners, calib, cam_mtx):
        # Alternative: triangulation method (from GatesDetectorTriangulation)
        if USE_MONO_HEIGHT_PROJECTION and self._triangulator is not None:
            with self.pos_lock:
                est = dict(self.est)
            self._triangulator.add_observation(corners, est)
            result = self._triangulator.triangulate(est)
            if result is not None:
                return result
            # Fall through to default method if triangulation fails

        cam = self._gate_camera_pose(corners, cam_mtx)
        if cam is not None:
            xc, yc, zc = cam
            dx_b, dy_b, dz_b = zc, -xc, -yc
        else:
            bw = bbox[2]
            fx = float(calib['fx'])
            fy = float(calib['fy'])
            dist = float(np.clip((GATE_PHYS_W * fx) / max(bw, 1), PNP_RANGE_MIN, PNP_RANGE_MAX))
            x_err_px = ex * max(0.5 * calib['img_w'], 1.0)
            y_err_px = ey * max(0.5 * calib['img_h'], 1.0)
            dx_b = dist
            dy_b = -x_err_px * dist / fx
            dz_b = -y_err_px * dist / fy

        with self.pos_lock:
            yaw_r = np.deg2rad(self.est['yaw'])
            ox, oy, oz = self.est['x'], self.est['y'], self.est['z']
        gx = ox + dx_b * np.cos(yaw_r) - dy_b * np.sin(yaw_r)
        gy = oy + dx_b * np.sin(yaw_r) + dy_b * np.cos(yaw_r)
        gz = float(np.clip(oz + dz_b, MIN_HEIGHT, MAX_HEIGHT))
        return gx, gy, gz

    def _try_lock_gate(self, gw, now):
        if gw is None:
            return None
        g = np.asarray(gw, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(g)):
            return None
        self._gate_obs = [(t, p) for t, p in self._gate_obs if (now - t) <= GATE_LOCK_MAX_AGE_S]
        self._gate_obs.append((now, tuple(g.tolist())))
        self._gate_obs = self._gate_obs[-GATE_LOCK_MAX_OBS:]
        if len(self._gate_obs) < GATE_LOCK_MIN_OBS:
            return None
        pts = np.array([p for _, p in self._gate_obs], dtype=np.float64)
        med = np.median(pts, axis=0)
        spread = float(np.max(np.linalg.norm(pts[:, :2] - med[:2], axis=1)))
        if spread <= GATE_LOCK_MAX_SPREAD_XY:
            return tuple(med.tolist())
        return None

    def _valid_chase_update(self, new_gw):
        if self._gate_world is None or new_gw is None:
            return False
        jump = float(np.linalg.norm(
            np.array(new_gw[:2]) - np.array(self._gate_world[:2])))
        return jump <= CHASE_UPDATE_MAX_JUMP_XY

    def _build_traj(self, gate):
        with self.pos_lock:
            x0, y0, z0 = self.est['x'], self.est['y'], self.est['z']
        gx, gy, gz = gate
        dx, dy = gx - x0, gy - y0
        mag = np.hypot(dx, dy) + 1e-6
        end_x = gx + TRAJ_OVERSHOOT * dx / mag
        end_y = gy + TRAJ_OVERSHOOT * dy / mag
        ts = np.linspace(0.0, 1.0, TRAJ_N_STEPS + 1)[1:]
        return [(x0 + (end_x - x0) * t, y0 + (end_y - y0) * t, z0 + (gz - z0) * t) for t in ts]
