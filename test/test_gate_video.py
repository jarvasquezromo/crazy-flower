#!/usr/bin/env python3
"""Offline video tester for the lap1_test.py gate detector + state machine.

Loads a recorded MP4 and replays the *exact same* detection and autonomy logic
that lap1_test.py runs on the real Crazyflie, so the pipeline can be tuned and
validated frame-by-frame without flying.

Detection (shared with lap1_test.py)
  - Threshold bright pixels into a mask, morphologically close it.
  - Validate each contour as a gate-shaped polygon (approxPolyDP vertex count,
    aspect ratio, solidity); the centre is the polygon's area centroid and the
    four corners are ordered TL/TR/BR/BL for height ranging.
  - Keep all candidates, select the rightmost; project it to a world point via
    a pinhole model (gate_to_world) whose depth comes from the gate's pixel
    *height* (corners, bbox-height fallback) using the (simulated) drone pose.

State machine (mirrors lap1_test.py, simulated offline)
  WAIT -> TAKEOFF -> SEARCH -> CHASE -> PUSH -> (repeat) -> DONE
  Detection comes from the real video frame each step; the drone pose is
  *simulated* by integrating the commanded position setpoints (there is no real
  Kalman estimate offline), so CHASE trajectory-following and the PUSH-through
  reset can be observed. See GateController for the per-state behaviour.

GUI
  Left: BW overlay with candidate polygons (rightmost = green). Right: mask.
  Sliders tune detection thresholds and a few control params; Prev/Next/Play
  scrub the video and Reset re-initialises the state machine.

Usage:
  python3 crazy-flower/test/test_gate_video.py --video Video/First_try.mp4
"""

import argparse
import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


TARGET_W = 324
TARGET_H = 244
PROCESS_FPS = 2.0

GATE_PHYS_H  = 0.4             # metres, physical gate height — the stable ranging
                               # dimension (yaw foreshortens width, not height)
CAMERA_FOV_H = np.deg2rad(87.0)  # AI-deck color camera H FoV (datasheet); overridden by calib file

# --- Control / state-machine tuning (mirrors lap1_test.py) ---
SEARCH_YAWRATE       = -20.0     # deg/s, yaw-in-place scan rate during SEARCH
SEARCH_HEIGHT        = 0.8       # metres, target height for search/chase
TAKEOFF_START_HEIGHT = 0.1       # metres, initial setpoint at takeoff
TAKEOFF_RATE         = 0.4       # m/s climb rate during the takeoff ramp
FORWARD_SPEED        = 0.35      # m/s body-X push speed
PUSH_DURATION_S      = 1.0       # seconds to commit straight through a gate
MAX_GATES            = 4         # stop (DONE) after this many gates
MIN_HEIGHT           = 0.2       # metres, safety clamp
MAX_HEIGHT           = 2.0       # metres, safety clamp
CHASE_TIMEOUT        = 8.0       # seconds before giving up and returning to SEARCH
PASS_AREA_FRAC       = 0.25      # gate bbox / image area threshold -> commit PUSH
GATE_EMA_ALPHA       = 0.35      # EMA weight for refining the locked gate position
TRAJ_N_STEPS         = 5         # waypoints in the interpolated trajectory
TRAJ_OVERSHOOT       = 0.30      # metres past gate centre (fly through cleanly)
WAYPOINT_TOL         = 0.15      # metres, advance to next waypoint within this radius

# Offline-only: how fast the simulated pose tracks the commanded setpoint.
SIM_MAX_SPEED        = 0.8       # m/s
SIM_MAX_YAWRATE      = 90.0      # deg/s


def load_calibration(path):
    """Return calibration dict. Falls back to FOV approximation when no file is given."""
    if path and os.path.isfile(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded calibration from {path}: "
              f"fx={data['fx']:.1f} fy={data['fy']:.1f} "
              f"FoV_H={data.get('fov_h_deg', float('nan')):.1f}°")
        return data
    fx = TARGET_W / (2.0 * np.tan(CAMERA_FOV_H / 2.0))
    print(f"No calibration file — using FOV approximation: fx=fy={fx:.1f}")
    return {'fx': fx, 'fy': fx,
            'cx': TARGET_W / 2.0, 'cy': TARGET_H / 2.0,
            'dist_coeffs': [0, 0, 0, 0, 0],
            'source': 'fov_approx'}


def gate_to_world(ex, ey, bbox, drone_z, drone_yaw_deg, calib,
                  drone_x=0.0, drone_y=0.0, corners=None):
    """Project detected gate centre (image coords) to world frame.

    Assumes camera looks along body +X (forward-facing AI-deck).
    ex, ey are normalised to [-1, 1] from the image centre. Depth comes from
    the gate's pixel *height* (see gate_distance), mirroring lap1_test.py.
    Returns (gx, gy, gz) in metres, relative to Lighthouse origin.
    """
    fx = calib['fx']
    fy = calib.get('fy', fx)
    # Principal-point offset correction (normalised, 0 if calibration is centred)
    cx_off = (calib.get('cx', TARGET_W / 2.0) - TARGET_W / 2.0) / max(TARGET_W / 2.0, 1.0)
    cy_off = (calib.get('cy', TARGET_H / 2.0) - TARGET_H / 2.0) / max(TARGET_H / 2.0, 1.0)
    # Distance from gate pixel height (corners), bbox-height fallback.
    dist   = gate_distance(bbox, corners, calib)
    # Body-frame offsets (pinhole: angle = pixel_offset / focal_length)
    dx_b   =  dist
    dy_b   = -(ex - cx_off) * (TARGET_W / 2.0) / fx * dist
    dz_b   = -(ey - cy_off) * (TARGET_H / 2.0) / fy * dist
    # Rotate body frame → world frame by drone yaw
    yaw_r  = np.deg2rad(drone_yaw_deg)
    gx = drone_x + dx_b * np.cos(yaw_r) - dy_b * np.sin(yaw_r)
    gy = drone_y + dx_b * np.sin(yaw_r) + dy_b * np.cos(yaw_r)
    gz = drone_z + dz_b
    return gx, gy, gz


def _order_corners(pts):
    """Order 4 image points as [TL, TR, BR, BL] (image y increases downward)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmax(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmin(d)],   # BL
    ], dtype=np.float64)


def _quad_corners(cnt, approx):
    """Best 4-corner estimate of the gate quad, ordered TL/TR/BR/BL.

    Uses the polygon approximation directly when it is a clean quad (keeps the
    true perspective of the four corners); otherwise falls back to the rotated
    min-area rectangle around the contour. Returns None if the 4 corners are not
    distinct. Mirrors lap1_test.py._quad_corners.
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

    Uses the side edges (not the bbox) so it stays correct under tilt, and uses
    height rather than width because a yaw off head-on foreshortens the gate's
    width but leaves its height intact. Returns None if unusable. Mirrors
    lap1_test.py._gate_pixel_height.
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


def gate_distance(bbox, corners, calib):
    """Camera->gate distance (m) from the gate's pixel height.

    Pinhole: dist = fy * GATE_PHYS_H / h_px, using the corner-derived side-edge
    height when available and the bbox height as a fallback. Height is the
    stable ranging dimension (mirrors lap1_test.py._gate_to_world).
    """
    fy = calib.get('fy', calib['fx'])
    h_px = _gate_pixel_height(corners)
    if h_px is None:
        h_px = max(bbox[3], 1)   # bbox-height fallback
    return max(fy * GATE_PHYS_H / h_px, 0.3)


def gate_edge_world_sizes(corners, bbox, calib):
    """World-frame lengths (m) of the 4 detected gate edges.

    Depth comes from gate_distance (gate pixel height). A pixel displacement
    (du, dv) on a fronto-parallel plane at that depth spans (du*dist/fx,
    dv*dist/fy) metres, so each polygon edge's world length is the norm of that.
    Returns a dict with the four edges (top/right/bottom/left), their mean
    width/height and depth, or None when the corners are unavailable.
    """
    if corners is None:
        return None
    fx = calib['fx']
    fy = calib.get('fy', fx)
    dist = gate_distance(bbox, corners, calib)
    tl, tr, br, bl = np.asarray(corners, dtype=np.float64).reshape(-1, 2)

    def world_len(a, b):
        du = (b[0] - a[0]) * dist / fx
        dv = (b[1] - a[1]) * dist / fy
        return float(np.hypot(du, dv))

    top, right, bottom, left = (world_len(tl, tr), world_len(tr, br),
                                world_len(br, bl), world_len(bl, tl))
    return {
        "top": top, "right": right, "bottom": bottom, "left": left,
        "width": 0.5 * (top + bottom),    # horizontal edges
        "height": 0.5 * (left + right),   # vertical edges
        "dist": dist,
    }


class VideoFrameStore:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.isOpened() else 0.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap.isOpened() else 0
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap.isOpened() else 0

    def get(self, frame_idx: int):
        if self.frame_count <= 0:
            return None
        idx = int(frame_idx)
        if idx < 0 or idx >= self.frame_count:
            return None
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return None

        # Always work at a smaller fixed resolution for consistent tuning.
        frame = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        if self.cap.isOpened():
            self.cap.release()


# --- Gate-shape acceptance thresholds (a gate is a roughly-square quad frame) ---
GATE_MIN_VERTICES = 4      # quad after polygon approximation
GATE_MAX_VERTICES = 8      # allow a few extra vertices from noise / rounded corners
GATE_ASPECT_MIN   = 0.45   # bbox w/h: tolerate perspective foreshortening
GATE_ASPECT_MAX   = 2.2
GATE_MIN_SOLIDITY = 0.80   # area / convex-hull area: frame outline is near-convex
GATE_APPROX_EPS   = 0.04   # approxPolyDP epsilon, fraction of perimeter


def _build_mask(rgb_img, hsv_lo, hsv_hi, min_v, kernel_size, use_bw):
    """Threshold + morphological cleanup. Returns a uint8 0/255 mask."""
    if use_bw:
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        v_lo = int(hsv_lo[2]) if hsv_lo is not None else 0
        thr = max(v_lo, int(min_v) if (min_v is not None and min_v > 0) else 0)
        mask = np.where(gray >= thr, np.uint8(255), np.uint8(0))
    else:
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
        if min_v is not None and min_v > 0:
            v = hsv[:, :, 2]
            v_mask = np.where(v >= int(min_v), np.uint8(255), np.uint8(0))
            mask = cv2.bitwise_and(mask, v_mask)

    k = max(1, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return mask


def _gate_candidate(cnt, img_w, img_h, min_area_frac):
    """Validate a contour as a gate-shaped polygon.

    Returns a candidate dict if the contour passes the shape tests, else None.
    """
    area = float(cv2.contourArea(cnt))
    if area < float(min_area_frac) * float(img_w * img_h):
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

    # Centre from the polygon itself (its area centroid), never the axis-aligned bbox.
    m = cv2.moments(approx)
    if abs(m.get("m00", 0.0)) < 1e-6:
        pts = approx.reshape(-1, 2).astype(np.float64)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
    else:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

    ex = (cx - 0.5 * img_w) / max(0.5 * img_w, 1.0)
    ey = (cy - 0.5 * img_h) / max(0.5 * img_h, 1.0)

    return {
        "found": True,
        "cx": cx,
        "cy": cy,
        "bbox": (x, y, bw, bh),
        "area": area,
        "ex": ex,
        "ey": ey,
        "n_vert": int(n_vert),
        "aspect": float(aspect),
        "solidity": float(solidity),
        "approx": approx,
        "corners": _quad_corners(cnt, approx),  # ordered TL/TR/BR/BL for height ranging
    }


def detect_green_gate(rgb_img, hsv_lo, hsv_hi, min_v, min_area_frac, kernel_size=5, *, use_bw=False):
    """Detect gate(s) and return (detection_dict, mask).

    The detection dict describes the *selected* gate (the rightmost one when
    several are visible) and carries the full candidate list under "candidates"
    so callers can visualise rejected/extra detections.

    Robustness vs. the old "largest blob" approach:
      * each contour is approximated to a polygon and must look like a roughly
        square quad frame (vertex count / aspect / solidity gates);
      * multiple gates are kept, and the rightmost is chosen for the controller.
    """
    h, w = rgb_img.shape[:2]
    mask = _build_mask(rgb_img, hsv_lo, hsv_hi, min_v, kernel_size, use_bw)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        cand = _gate_candidate(cnt, w, h, min_area_frac)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return {"found": False, "candidates": []}, mask

    # Selection policy: take the most-right gate (largest centroid x).
    best = max(candidates, key=lambda c: c["cx"])
    best = dict(best)  # don't mutate the list entry
    best["candidates"] = candidates
    best["n_candidates"] = len(candidates)
    return best, mask


class GateController:
    """Offline simulation of the lap1_test.py world-frame state machine.

    Detection is fed in from the real video frame each step; the drone pose is
    simulated by integrating the commanded position setpoints (no Kalman offline).

    States: WAIT -> TAKEOFF -> SEARCH -> CHASE -> PUSH -> (repeat) -> DONE
      WAIT     immediately arms the sequence.
      TAKEOFF  ramps the height setpoint up to search_height.
      SEARCH   yaws in place until a gate is found, then locks it and builds a
               fly-through trajectory.
      CHASE    follows the trajectory, refines the locked gate (EMA), points yaw
               at it, and commits to PUSH once the gate fills the frame; times
               out back to SEARCH if the gate is lost.
      PUSH     drives straight to an overshoot point for push_duration_s, then
               counts the gate and returns to SEARCH (or DONE at max_gates).
    """

    def __init__(self, calib):
        self.calib = calib
        self.reset()

    def reset(self):
        self.state = "WAIT"
        self.est = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.pos = {"x": 0.0, "y": 0.0, "z": TAKEOFF_START_HEIGHT, "yaw": 0.0}
        self.gate_world = None
        self.traj = []
        self.traj_idx = 0
        self.gates_passed = 0
        self.push_target = (0.0, 0.0, TAKEOFF_START_HEIGHT)
        self.t = 0.0
        self.state_t0 = 0.0

    def _gate_to_world(self, ex, ey, bbox, corners=None):
        return gate_to_world(
            ex, ey, bbox,
            self.est["z"], self.est["yaw"], self.calib,
            drone_x=self.est["x"], drone_y=self.est["y"],
            corners=corners,
        )

    def _build_traj(self, gate):
        x0, y0, z0 = self.est["x"], self.est["y"], self.est["z"]
        gx, gy, gz = gate
        dx, dy = gx - x0, gy - y0
        mag = np.hypot(dx, dy) + 1e-6
        end_x = gx + TRAJ_OVERSHOOT * dx / mag
        end_y = gy + TRAJ_OVERSHOOT * dy / mag
        ts = np.linspace(0.0, 1.0, TRAJ_N_STEPS + 1)[1:]
        return [(x0 + (end_x - x0) * t,
                 y0 + (end_y - y0) * t,
                 z0 + (gz - z0) * t) for t in ts]

    def _integrate(self, dt):
        """Simulated pose tracks the commanded setpoint (first-order, rate-limited)."""
        if dt <= 0:
            return
        for k in ("x", "y", "z"):
            err = self.pos[k] - self.est[k]
            self.est[k] += float(np.clip(err, -SIM_MAX_SPEED * dt, SIM_MAX_SPEED * dt))
        dyaw = (self.pos["yaw"] - self.est["yaw"] + 180.0) % 360.0 - 180.0
        self.est["yaw"] += float(np.clip(dyaw, -SIM_MAX_YAWRATE * dt, SIM_MAX_YAWRATE * dt))

    def step(self, dt, det, img_area, *,
             search_yawrate=SEARCH_YAWRATE,
             search_height=SEARCH_HEIGHT,
             pass_area_frac=PASS_AREA_FRAC,
             push_duration_s=PUSH_DURATION_S):
        self.t += float(dt)
        now = self.t

        found = bool(det.get("found", False))
        ex = float(det.get("ex", 0.0))
        ey = float(det.get("ey", 0.0))
        bbox = det.get("bbox", None)
        corners = det.get("corners", None)
        area = float(det.get("area", 0.0))

        if self.state == "WAIT":
            self.state = "TAKEOFF"
            self.state_t0 = now

        if self.state == "TAKEOFF":
            self.pos["z"] = float(min(self.pos["z"] + TAKEOFF_RATE * dt, search_height))
            if self.pos["z"] >= search_height - 1e-3:
                self.state = "SEARCH"
                self.state_t0 = now

        elif self.state == "SEARCH":
            self.pos["yaw"] += search_yawrate * dt
            if found and bbox is not None:
                gw = self._gate_to_world(ex, ey, bbox, corners)
                self.gate_world = gw
                self.traj = self._build_traj(gw)
                self.traj_idx = 0
                self.state = "CHASE"
                self.state_t0 = now

        elif self.state == "CHASE":
            if (now - self.state_t0) > CHASE_TIMEOUT:
                self.gate_world = None
                self.traj = []
                self.traj_idx = 0
                self.state = "SEARCH"
                self.state_t0 = now
            else:
                if found and bbox is not None:
                    nx, ny, nz = self._gate_to_world(ex, ey, bbox, corners)
                    ox, oy, oz = self.gate_world
                    self.gate_world = (
                        ox + GATE_EMA_ALPHA * (nx - ox),
                        oy + GATE_EMA_ALPHA * (ny - oy),
                        oz + GATE_EMA_ALPHA * (nz - oz),
                    )
                    if self.traj_idx == 0:
                        self.traj = self._build_traj(self.gate_world)

                if img_area > 0 and area / img_area > pass_area_frac:
                    yaw_r = np.deg2rad(self.est["yaw"])
                    push_dist = FORWARD_SPEED * push_duration_s + TRAJ_OVERSHOOT
                    self.push_target = (
                        self.est["x"] + push_dist * np.cos(yaw_r),
                        self.est["y"] + push_dist * np.sin(yaw_r),
                        self.pos["z"],
                    )
                    self.state = "PUSH"
                    self.state_t0 = now
                elif self.traj:
                    wp = self.traj[self.traj_idx]
                    self.pos["x"], self.pos["y"], self.pos["z"] = wp
                    gx, gy, _ = self.gate_world
                    self.pos["yaw"] = float(np.degrees(
                        np.arctan2(gy - self.est["y"], gx - self.est["x"])))
                    dist_wp = np.hypot(self.est["x"] - wp[0], self.est["y"] - wp[1])
                    if dist_wp < WAYPOINT_TOL and self.traj_idx < len(self.traj) - 1:
                        self.traj_idx += 1

        elif self.state == "PUSH":
            self.pos["x"], self.pos["y"], self.pos["z"] = self.push_target
            if (now - self.state_t0) >= push_duration_s:
                self.gates_passed += 1
                self.gate_world = None
                self.traj = []
                self.traj_idx = 0
                if self.gates_passed >= MAX_GATES:
                    self.state = "DONE"
                else:
                    self.state = "SEARCH"
                    self.state_t0 = now

        self.pos["z"] = float(np.clip(self.pos["z"], MIN_HEIGHT, MAX_HEIGHT))
        self._integrate(dt)
        return self.state


def main():
    parser = argparse.ArgumentParser(description="Green gate detector video harness")
    parser.add_argument("--video",      default=os.path.join("Video", "fpv_20260519_093807.mp4"))
    parser.add_argument("--calib",      default=None,  help="Path to calibration.json")
    args = parser.parse_args()
    calib = load_calibration(args.calib)

    store = VideoFrameStore(args.video)
    if store.frame_count <= 0:
        print(f"Could not open video: {args.video}")
        return

    src_fps = store.fps if store.fps > 0 else 30.0
    fps = float(PROCESS_FPS)
    step_frames = int(max(1, round(src_fps / max(fps, 1e-6))))

    # State
    idx = 0
    playing = False
    fsm = GateController(calib)

    # Initial parameters (match lap1 defaults)
    params = {
        "h_lo": 35,
        "h_hi": 85,
        "s_lo": 50,
        "v_lo": 50,
        "min_v": 240,
        "min_area_frac": 0.01,
        "kernel": 5,
        "search_yawrate": SEARCH_YAWRATE,
        "search_height": SEARCH_HEIGHT,
        "pass_area_frac": PASS_AREA_FRAC,
    }

    frame0 = store.get(0)
    if frame0 is None:
        print("Could not read first frame")
        return

    # Build figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.38)

    ax0, ax1 = axs

    def compute_panels(frame_bgr):
        if frame_bgr is None:
            blank = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
            return blank, np.zeros((TARGET_H, TARGET_W), dtype=np.uint8), {"found": False}

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hsv_lo = np.array([params["h_lo"], params["s_lo"], params["v_lo"]], dtype=np.uint8)
        hsv_hi = np.array([params["h_hi"], 255, 255], dtype=np.uint8)

        det, mask = detect_green_gate(
            rgb,
            hsv_lo,
            hsv_hi,
            params["min_v"],
            params["min_area_frac"],
            kernel_size=params["kernel"],
            use_bw=True,
        )

        # Display + detection are both based on BW (intensity) input.
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        h, w = overlay.shape[:2]
        cv2.drawMarker(overlay, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)

        # Draw the detected polygon for every accepted candidate (no bounding
        # box): thin/yellow, with the selected (rightmost) gate thick/green.
        sel_bbox = det.get("bbox")
        for cand in det.get("candidates", []):
            if cand.get("approx") is None:
                continue
            is_sel = (cand["bbox"] == sel_bbox)
            color = (0, 255, 0) if is_sel else (255, 255, 0)
            thick = 2 if is_sel else 1
            cv2.polylines(overlay, [cand["approx"]], True, color, thick)

        if det.get("found", False) and sel_bbox is not None:
            cv2.circle(overlay, (int(round(det["cx"])), int(round(det["cy"]))), 4, (255, 0, 0), -1)

            # World-frame size of the detected gate polygon's edges.
            corners = det.get("corners")
            edges = gate_edge_world_sizes(corners, sel_bbox, calib)
            if edges is not None:
                det["edge_world"] = edges
                pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                labels = ("top", "right", "bottom", "left")
                for i, name in enumerate(labels):
                    a = pts[i]
                    b = pts[(i + 1) % 4]
                    mx, my = int(round(0.5 * (a[0] + b[0]))), int(round(0.5 * (a[1] + b[1])))
                    cv2.putText(overlay, f"{edges[name]:.2f}m", (mx - 18, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        return overlay, mask, det

    overlay0, mask0, det0 = compute_panels(frame0)
    im0 = ax0.imshow(overlay0)
    im1 = ax1.imshow(mask0, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Overlay (BW input)")
    ax1.set_title("Mask")
    ax0.axis("off")
    ax1.axis("off")

    title = fig.suptitle("", fontsize=12)

    last_print_idx = [-1]   # avoid re-printing the same frame on slider redraws

    def refresh():
        nonlocal idx, playing
        frame = store.get(idx)
        overlay, mask, det = compute_panels(frame)

        found = bool(det.get("found", False))
        ex = float(det.get("ex", 0.0))
        ey = float(det.get("ey", 0.0))

        # Print the world-frame size of the detected gate polygon's edges.
        edges = det.get("edge_world")
        if edges is not None and (playing or idx != last_print_idx[0]):
            print(
                f"[frame {idx + 1:04d}] gate edges (world): "
                f"top={edges['top']:.2f} right={edges['right']:.2f} "
                f"bottom={edges['bottom']:.2f} left={edges['left']:.2f} m  "
                f"(W={edges['width']:.2f} H={edges['height']:.2f} m @ dist={edges['dist']:.2f} m)"
            )
            last_print_idx[0] = idx

        # Advance the state machine only while playing (dt=0 when paused/scrubbing).
        dt = (1.0 / fps) if playing else 0.0
        prev_state = fsm.state
        state = fsm.step(
            dt,
            det,
            float(TARGET_W * TARGET_H),
            search_yawrate=params["search_yawrate"],
            search_height=params["search_height"],
            pass_area_frac=params["pass_area_frac"],
        )
        if playing and state != prev_state:
            print(f"[frame {idx + 1:04d}] {prev_state} -> {state}  gates={fsm.gates_passed}")
        if state == "DONE" and playing:
            playing = False
            b_play.label.set_text("Play")

        # Locked world-frame gate estimate (maintained by the controller).
        if fsm.gate_world is not None:
            gx, gy, gz = fsm.gate_world
            world_str = f"({gx:+.2f}, {gy:+.2f}, {gz:+.2f}) m"
        else:
            world_str = "n/a"

        im0.set_data(overlay)
        im1.set_data(mask)

        n_cand = int(det.get("n_candidates", 0))
        shape_str = ""
        if found:
            shape_str = f" verts={det.get('n_vert', 0)} ar={det.get('aspect', 0):.2f} sol={det.get('solidity', 0):.2f}"
        est = fsm.est
        pos = fsm.pos
        title.set_text(
            f"Frame {idx + 1}/{store.frame_count}  proc_fps={fps:.1f} src_fps={src_fps:.1f} step={step_frames}  "
            f"found={int(found)} cand={n_cand}{shape_str}  ex={ex:+.2f} ey={ey:+.2f}\n"
            f"state={fsm.state}  gates={fsm.gates_passed}/{MAX_GATES}  traj={fsm.traj_idx}/{len(fsm.traj)}  "
            f"gate_world={world_str}\n"
            f"est: x={est['x']:+.2f} y={est['y']:+.2f} z={est['z']:.2f} yaw={est['yaw']:+.0f}°   "
            f"cmd: x={pos['x']:+.2f} y={pos['y']:+.2f} z={pos['z']:.2f} yaw={pos['yaw']:+.0f}°"
        )
        fig.canvas.draw_idle()

    # --- Sliders ---
    ax_hlo = plt.axes([0.12, 0.22, 0.35, 0.03])
    s_hlo = Slider(ax_hlo, "H low", 0, 179, valinit=params["h_lo"], valfmt="%d")

    ax_hhi = plt.axes([0.55, 0.22, 0.35, 0.03])
    s_hhi = Slider(ax_hhi, "H high", 0, 179, valinit=params["h_hi"], valfmt="%d")

    ax_slo = plt.axes([0.12, 0.17, 0.35, 0.03])
    s_slo = Slider(ax_slo, "S low", 0, 255, valinit=params["s_lo"], valfmt="%d")

    ax_vlo = plt.axes([0.55, 0.17, 0.35, 0.03])
    s_vlo = Slider(ax_vlo, "V low", 0, 255, valinit=params["v_lo"], valfmt="%d")

    ax_minv = plt.axes([0.12, 0.12, 0.35, 0.03])
    s_minv = Slider(ax_minv, "min V", 0, 255, valinit=params["min_v"], valfmt="%d")

    ax_area = plt.axes([0.55, 0.12, 0.35, 0.03])
    s_area = Slider(ax_area, "min area%", 0.0, 0.10, valinit=params["min_area_frac"], valfmt="%.3f")

    ax_kern = plt.axes([0.12, 0.07, 0.35, 0.03])
    s_kern = Slider(ax_kern, "kernel", 1, 21, valinit=params["kernel"], valfmt="%d", valstep=2)

    ax_pass = plt.axes([0.55, 0.07, 0.35, 0.03])
    s_pass = Slider(ax_pass, "pass area%", 0.05, 0.60, valinit=params["pass_area_frac"], valfmt="%.2f")

    ax_sh   = plt.axes([0.12, 0.27, 0.35, 0.03])
    s_sh    = Slider(ax_sh,   "Search H (m)",  0.2,  2.5, valinit=params["search_height"], valfmt="%.2f")
    ax_syaw = plt.axes([0.55, 0.27, 0.35, 0.03])
    s_syaw  = Slider(ax_syaw, "Search yaw/s",  -60,   60, valinit=params["search_yawrate"], valfmt="%.0f")

    def on_slider(_):
        params["h_lo"] = int(s_hlo.val)
        params["h_hi"] = int(s_hhi.val)
        params["s_lo"] = int(s_slo.val)
        params["v_lo"] = int(s_vlo.val)
        params["min_v"] = int(s_minv.val)
        params["min_area_frac"] = float(s_area.val)
        params["kernel"] = int(s_kern.val)
        params["pass_area_frac"] = float(s_pass.val)
        params["search_height"]  = float(s_sh.val)
        params["search_yawrate"] = float(s_syaw.val)
        refresh()

    for s in (s_hlo, s_hhi, s_slo, s_vlo, s_minv, s_area, s_kern, s_pass, s_sh, s_syaw):
        s.on_changed(on_slider)

    # --- Buttons ---
    ax_prev = plt.axes([0.30, 0.01, 0.10, 0.045])
    b_prev = Button(ax_prev, "Prev")
    ax_next = plt.axes([0.41, 0.01, 0.10, 0.045])
    b_next = Button(ax_next, "Next")
    ax_play = plt.axes([0.52, 0.01, 0.10, 0.045])
    b_play = Button(ax_play, "Play")
    ax_reset = plt.axes([0.63, 0.01, 0.12, 0.045])
    b_reset = Button(ax_reset, "Reset state")

    def go(step):
        nonlocal idx
        idx = int(np.clip(idx + step, 0, store.frame_count - 1))
        refresh()

    def on_prev(_):
        go(-1)

    def on_next(_):
        go(+1)

    def on_play(_):
        nonlocal playing
        playing = not playing
        b_play.label.set_text("Pause" if playing else "Play")
        refresh()

    def on_reset(_):
        fsm.reset()
        refresh()

    b_prev.on_clicked(on_prev)
    b_next.on_clicked(on_next)
    b_play.on_clicked(on_play)
    b_reset.on_clicked(on_reset)

    # Timer for playback
    last = time.perf_counter()

    def tick(_evt):
        nonlocal idx, last
        if not playing:
            return
        now = time.perf_counter()
        if now - last < (1.0 / fps):
            return
        last = now
        idx += step_frames
        if idx >= store.frame_count:
            idx = store.frame_count - 1
        refresh()

    fig.canvas.mpl_connect("draw_event", tick)

    # Use a Matplotlib timer for smooth playback
    timer = fig.canvas.new_timer(interval=max(1, int(1000 / fps)))
    timer.add_callback(lambda: (tick(None), None))
    timer.start()

    refresh()
    print(
        f"Loaded {store.frame_count} frames, src_fps={src_fps:.1f}, proc_fps={fps:.1f}, "
        f"step={step_frames} frames/tick, src_size={store.width}x{store.height}, proc_size={TARGET_W}x{TARGET_H}"
    )
    plt.show()

    store.close()


if __name__ == "__main__":
    main()
