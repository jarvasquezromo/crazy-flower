#!/usr/bin/env python3
"""Offline video tester for the green-gate detector + simple state machine.

This is meant to be the "same kind" of workflow as test_video.py but for the
lap1 green gate logic:
- Load an MP4
- Scrub frames
- Tune HSV / min-V thresholds
- Visualize mask + overlay
- Simulate SEARCH/CENTER/PUSH commands (yaw / lateral / height)

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

GATE_PHYS_W  = 0.8               # metres, physical gate width
CAMERA_FOV_H = np.deg2rad(87.0)  # AI-deck color camera H FoV (datasheet); overridden by calib file


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
                  drone_x=0.0, drone_y=0.0):
    """Project detected gate centre (image coords) to world frame.

    Assumes camera looks along body +X (forward-facing AI-deck).
    ex, ey are normalised to [-1, 1] from the image centre.
    Returns (gx, gy, gz) in metres, relative to Lighthouse origin.
    """
    bw = bbox[2]
    fx = calib['fx']
    fy = calib.get('fy', fx)
    # Principal-point offset correction (normalised, 0 if calibration is centred)
    cx_off = (calib.get('cx', TARGET_W / 2.0) - TARGET_W / 2.0) / max(TARGET_W / 2.0, 1.0)
    cy_off = (calib.get('cy', TARGET_H / 2.0) - TARGET_H / 2.0) / max(TARGET_H / 2.0, 1.0)
    # Distance from gate bbox width: dist = (phys_width * focal_px) / bbox_px
    dist   = max(GATE_PHYS_W * fx / max(bw, 1), 0.3)
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


def detect_green_gate(rgb_img, hsv_lo, hsv_hi, min_v, min_area_frac, kernel_size=9, *, use_bw=False):
    """Return detection dict and mask.

    If use_bw=True, detection runs on a black & white (intensity) version of the image.
    In that mode, the HSV bounds are not meaningful; we reuse V thresholds as an
    intensity threshold.
    """
    h, w = rgb_img.shape[:2]

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
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False}, mask

    img_area = float(h * w)
    best = max(contours, key=cv2.contourArea)
    best_area = float(cv2.contourArea(best))

    if best_area < float(min_area_frac) * img_area:
        return {"found": False}, mask

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
        "bbox": (x, y, bw, bh),
        "area": best_area,
        "ex": ex,
        "ey": ey,
    }, mask


class GateStateMachine:
    def __init__(self):
        self.state = "SEARCH"
        self.t_state = 0.0
        self.gates = 0
        self.height = 0.8

    def reset(self, height=0.8):
        self.state = "SEARCH"
        self.t_state = 0.0
        self.gates = 0
        self.height = float(height)

    def step(
        self,
        dt,
        found,
        ex,
        ey,
        *,
        search_yawrate=-20.0,
        k_yaw=80.0,
        max_yawrate=70.0,
        forward_speed=0.35,
        push_duration_s=1.0,
        center_tol_x=0.10,
        center_tol_y=0.12,
        k_height=1.2,
        max_dh_per_s=0.6,
        min_height=0.2,
        max_height=2.0,
    ):
        self.t_state += float(dt)

        x_cmd = 0.0
        y_cmd = 0.0
        yaw_cmd = 0.0

        if self.state == "SEARCH":
            yaw_cmd = float(search_yawrate)
            if found:
                self.state = "CENTER"
                self.t_state = 0.0

        elif self.state == "CENTER":
            if not found:
                self.state = "SEARCH"
                self.t_state = 0.0
            else:
                yaw_cmd = float(np.clip(-k_yaw * ex, -max_yawrate, max_yawrate))
                y_cmd = float(np.clip(-0.15 * ex, -0.2, 0.2))

                dh = float(np.clip(-k_height * ey, -max_dh_per_s, max_dh_per_s))
                self.height = float(np.clip(self.height + dh * float(dt), min_height, max_height))

                centered = (abs(ex) <= center_tol_x) and (abs(ey) <= center_tol_y)
                if centered:
                    self.state = "PUSH"
                    self.t_state = 0.0

        elif self.state == "PUSH":
            x_cmd = float(forward_speed)
            if self.t_state >= float(push_duration_s):
                self.gates += 1
                self.state = "SEARCH"
                self.t_state = 0.0

        return x_cmd, y_cmd, yaw_cmd, self.height


def main():
    parser = argparse.ArgumentParser(description="Green gate detector video harness")
    parser.add_argument("--video",      default=os.path.join("Video", "fpv_20260519_093807.mp4"))
    parser.add_argument("--calib",      default=None,  help="Path to calibration.json")
    parser.add_argument("--drone-z",    type=float, default=0.8, help="Simulated drone Z in m")
    parser.add_argument("--drone-yaw",  type=float, default=0.0, help="Simulated drone yaw in deg")
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
    fsm = GateStateMachine()

    # Initial parameters (match lap1 defaults)
    params = {
        "h_lo": 35,
        "h_hi": 85,
        "s_lo": 50,
        "v_lo": 50,
        "min_v": 240,
        "min_area_frac": 0.01,
        "kernel": 9,
        "search_yawrate": -20.0,
        "k_yaw": 80.0,
        "k_height": 1.2,
        "push_duration": 1.0,
        "drone_z":   args.drone_z,
        "drone_yaw": args.drone_yaw,
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

        if det.get("found", False) and det.get("bbox") is not None:
            x, y, bw, bh = det["bbox"]
            cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.circle(overlay, (int(round(det["cx"])), int(round(det["cy"]))), 4, (255, 0, 0), -1)

        return overlay, mask, det

    overlay0, mask0, det0 = compute_panels(frame0)
    im0 = ax0.imshow(overlay0)
    im1 = ax1.imshow(mask0, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("Overlay (BW input)")
    ax1.set_title("Mask")
    ax0.axis("off")
    ax1.axis("off")

    title = fig.suptitle("", fontsize=12)

    def refresh():
        nonlocal idx
        frame = store.get(idx)
        overlay, mask, det = compute_panels(frame)

        found = bool(det.get("found", False))
        ex = float(det.get("ex", 0.0))
        ey = float(det.get("ey", 0.0))

        # Simulate controller step even when paused (dt=0) so state doesn't advance.
        dt = (1.0 / fps) if playing else 0.0
        x_cmd, y_cmd, yaw_cmd, h_cmd = fsm.step(
            dt,
            found,
            ex,
            ey,
            search_yawrate=params["search_yawrate"],
            k_yaw=params["k_yaw"],
            k_height=params["k_height"],
            push_duration_s=params["push_duration"],
        )

        # World-frame gate estimate
        world_str = "n/a"
        if found and det.get("bbox") is not None:
            gx, gy, gz = gate_to_world(
                ex, ey, det["bbox"],
                params["drone_z"], params["drone_yaw"], calib,
            )
            world_str = f"({gx:+.2f}, {gy:+.2f}, {gz:+.2f}) m"
            if playing:
                print(f"[frame {idx + 1:04d}] gate_world = {world_str}")

        im0.set_data(overlay)
        im1.set_data(mask)

        title.set_text(
            f"Frame {idx + 1}/{store.frame_count}  proc_fps={fps:.1f} src_fps={src_fps:.1f} step={step_frames}  found={int(found)}  "
            f"ex={ex:+.2f} ey={ey:+.2f}  state={fsm.state} gates={fsm.gates}  "
            f"cmd: x={x_cmd:+.2f} y={y_cmd:+.2f} yaw={yaw_cmd:+.1f} h={h_cmd:.2f}\n"
            f"gate_world={world_str}   drone: z={params['drone_z']:.2f} m  yaw={params['drone_yaw']:.0f}°"
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

    ax_kyaw = plt.axes([0.55, 0.07, 0.35, 0.03])
    s_kyaw = Slider(ax_kyaw, "K yaw", 0.0, 150.0, valinit=params["k_yaw"], valfmt="%.1f")

    ax_dz   = plt.axes([0.12, 0.27, 0.35, 0.03])
    s_dz    = Slider(ax_dz,   "Drone Z (m)",   0.0,  2.5, valinit=params["drone_z"],   valfmt="%.2f")
    ax_dyaw = plt.axes([0.55, 0.27, 0.35, 0.03])
    s_dyaw  = Slider(ax_dyaw, "Drone Yaw (°)", -180, 180, valinit=params["drone_yaw"], valfmt="%.0f")

    def on_slider(_):
        params["h_lo"] = int(s_hlo.val)
        params["h_hi"] = int(s_hhi.val)
        params["s_lo"] = int(s_slo.val)
        params["v_lo"] = int(s_vlo.val)
        params["min_v"] = int(s_minv.val)
        params["min_area_frac"] = float(s_area.val)
        params["kernel"] = int(s_kern.val)
        params["k_yaw"] = float(s_kyaw.val)
        params["drone_z"]   = float(s_dz.val)
        params["drone_yaw"] = float(s_dyaw.val)
        refresh()

    for s in (s_hlo, s_hhi, s_slo, s_vlo, s_minv, s_area, s_kern, s_kyaw, s_dz, s_dyaw):
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
        fsm.reset(height=0.8)
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
