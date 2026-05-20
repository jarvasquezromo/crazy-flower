#!/usr/bin/env python3
"""AI-deck camera calibration using a checkerboard pattern.

Usage — live stream from the AI-deck:
    python3 calibration.py

Usage — from a pre-recorded video:
    python3 calibration.py --video path/to/video.mp4

Controls:
    c / SPACE   capture current frame (only when checkerboard is detected)
    r           run calibration from captured frames
    s           save to calibration.json (runs calibration first if needed)
    d           delete last captured frame
    q / ESC     quit

Output — calibration.json:
    fx, fy, cx, cy        camera matrix entries (pixels)
    fov_h_deg, fov_v_deg  derived field of view angles
    dist_coeffs           5 distortion coefficients [k1,k2,p1,p2,k3]
    rms_px                reprojection RMS in pixels (lower is better, aim < 1.0)
    img_w, img_h          image size used during calibration
"""

import argparse
import contextlib
import json
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

# ── AI-deck UDP constants (same as lap1_test.py) ─────────────────────────────
AIDECK_IP    = '192.168.4.1'
AIDECK_PORT  = 5000
LOCAL_PORT   = 5001
START_MAGIC  = b'FER'
CPX_HEADER   = 4
IMG_MAGIC    = 0xBC
IMG_HDR_SIZE = 11
IMG_W, IMG_H = 324, 324
MIN_JPEG     = 5000

# ── Checkerboard defaults ─────────────────────────────────────────────────────
DEFAULT_COLS   = 10     # inner corners, horizontal
DEFAULT_ROWS   = 7      # inner corners, vertical
DEFAULT_SQ_MM  = 21.0   # physical square size in millimetres
MIN_FRAMES     = 10     # minimum captured frames before calibration is allowed

DETECT_SCALE   = 2      # upscale factor for findChessboardCorners (2 = 2× better detection)


# ── Suppress FFMPEG/libjpeg stderr noise ─────────────────────────────────────
@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null  = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


# ── Frame sources ─────────────────────────────────────────────────────────────
class LiveAiDeckSource:
    """Receives JPEG frames from the AI-deck over UDP (background thread)."""

    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        print(f"Connecting to AI-deck at {AIDECK_IP}:{AIDECK_PORT} ...")

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))
        buf       = bytearray()
        exp_size  = 0
        receiving = False
        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < CPX_HEADER:
                continue
            payload = data[CPX_HEADER:]
            if len(payload) >= IMG_HDR_SIZE and payload[0] == IMG_MAGIC:
                _, w, h, _, _, size = struct.unpack('<BHHBBI', payload[:IMG_HDR_SIZE])
                if w == IMG_W and h == IMG_H and 0 < size < 65536:
                    exp_size  = size
                    buf       = bytearray()
                    receiving = True
                    continue
            if not receiving:
                continue
            buf.extend(payload)
            if len(buf) >= exp_size:
                self._decode(bytes(buf))
                receiving = False

    def _decode(self, buf):
        soi = buf.find(b'\xff\xd8')
        eoi = buf.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi or (eoi + 2 - soi) < MIN_JPEG:
            return
        jpeg = np.frombuffer(buf, np.uint8, count=eoi + 2 - soi, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape == (IMG_H, IMG_W):
            with self._lock:
                self._frame = img.copy()

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def release(self):
        pass


class VideoSource:
    """Loops through a video file, returning grayscale frames."""

    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else None

    def release(self):
        self._cap.release()


# ── Calibration helpers ───────────────────────────────────────────────────────
def _run_calibration(obj_pts, img_pts, img_size):
    rms, mtx, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    return rms, mtx, dist


def _save(path, mtx, dist, img_w, img_h, rms):
    fx, fy = float(mtx[0, 0]), float(mtx[1, 1])
    cx, cy = float(mtx[0, 2]), float(mtx[1, 2])
    fov_h  = float(2.0 * np.degrees(np.arctan2(img_w / 2.0, fx)))
    fov_v  = float(2.0 * np.degrees(np.arctan2(img_h / 2.0, fy)))
    data   = {
        'img_w': img_w, 'img_h': img_h,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'fov_h_deg': fov_h, 'fov_v_deg': fov_v,
        'dist_coeffs': dist.flatten().tolist(),
        'rms_px': rms,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved → {path}")
    print(f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}")
    print(f"  FoV  H={fov_h:.1f}°  V={fov_v:.1f}°")
    print(f"  RMS reprojection error: {rms:.3f} px  (aim < 1.0 px)")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='AI-deck camera calibration')
    parser.add_argument('--video',  default=None,
                        help='Video file to use instead of live AI-deck stream')
    parser.add_argument('--output', default='calibration.json',
                        help='Output JSON path (default: calibration.json)')
    parser.add_argument('--fps',    type=float, default=None,
                        help='Playback FPS for video source (default: 5). Ignored for live.')
    parser.add_argument('--cols',   type=int,   default=DEFAULT_COLS,
                        help='Checkerboard inner corners (horizontal)')
    parser.add_argument('--rows',   type=int,   default=DEFAULT_ROWS,
                        help='Checkerboard inner corners (vertical)')
    parser.add_argument('--square', type=float, default=DEFAULT_SQ_MM,
                        help='Square size in mm (default: 25)')
    args = parser.parse_args()

    board = (args.cols, args.rows)
    objp  = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = (np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
                   * (args.square / 1000.0))

    source = VideoSource(args.video) if args.video else LiveAiDeckSource()

    obj_pts    = []
    img_pts    = []
    img_size   = None
    last_calib = None   # {'rms_px', 'mtx', 'dist', 'img_w', 'img_h'} after 'r'

    criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    delay_ms  = int(1000.0 / (args.fps if args.fps else 5.0)) if args.video else 30
    det_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 | cv2.CALIB_CB_NORMALIZE_IMAGE
                 | cv2.CALIB_CB_FAST_CHECK)

    print(f"\nCheckerboard: {args.cols}×{args.rows} inner corners, {args.square} mm squares")
    if args.video:
        print(f"Video playback: {1000 // delay_ms} fps  (use --fps N to change)")
    print("Controls: c/SPACE=capture  r=calibrate  s=save  d=delete last  q=quit\n")

    while True:
        frame = source.read()
        if frame is None:
            time.sleep(0.01)
            continue

        if img_size is None:
            img_size = (frame.shape[1], frame.shape[0])

        # --- Pre-process: adaptive contrast ---
        proc = clahe.apply(frame)

        # --- Detect on upscaled image, then scale corners back ---
        h, w = proc.shape[:2]
        big  = cv2.resize(proc, (w * DETECT_SCALE, h * DETECT_SCALE),
                          interpolation=cv2.INTER_LINEAR)
        found, corners_big = cv2.findChessboardCorners(big, board, det_flags)
        corners = None
        if found:
            corners = corners_big / DETECT_SCALE
            corners = cv2.cornerSubPix(proc, corners, (11, 11), (-1, -1), criteria)

        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if found and corners is not None:
            cv2.drawChessboardCorners(disp, board, corners, found)

        rms_str = (f"  RMS={last_calib['rms_px']:.3f}px" if last_calib else "")
        board_str = "[board found]" if found else "[no board]"
        col       = (0, 220, 0)    if found else (0, 80, 255)
        status    = f"Captured: {len(obj_pts)}/{MIN_FRAMES}  {board_str}{rms_str}"
        cv2.putText(disp, status, (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.imshow('Calibration — AI-deck', disp)

        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord('q'), 27):
            break

        if key in (ord('c'), ord(' ')):
            if found:
                obj_pts.append(objp.copy())
                img_pts.append(corners)
                print(f"  Captured frame {len(obj_pts)}")
            else:
                print("  No checkerboard detected — skipped")

        elif key == ord('d'):
            if obj_pts:
                obj_pts.pop()
                img_pts.pop()
                print(f"  Deleted last frame ({len(obj_pts)} remaining)")

        elif key == ord('r'):
            if len(obj_pts) < MIN_FRAMES:
                print(f"  Need at least {MIN_FRAMES} frames (have {len(obj_pts)})")
            else:
                print(f"  Running calibration on {len(obj_pts)} frames …")
                rms, mtx, dist = _run_calibration(obj_pts, img_pts, img_size)
                last_calib = {'rms_px': rms, 'mtx': mtx, 'dist': dist,
                              'img_w': img_size[0], 'img_h': img_size[1]}
                print(f"  RMS = {rms:.3f} px")

        elif key == ord('s'):
            if last_calib is None:
                if len(obj_pts) < MIN_FRAMES:
                    print(f"  Need at least {MIN_FRAMES} frames first")
                else:
                    print("  Running calibration …")
                    rms, mtx, dist = _run_calibration(obj_pts, img_pts, img_size)
                    last_calib = {'rms_px': rms, 'mtx': mtx, 'dist': dist,
                                  'img_w': img_size[0], 'img_h': img_size[1]}
            if last_calib:
                _save(args.output,
                      last_calib['mtx'], last_calib['dist'],
                      last_calib['img_w'], last_calib['img_h'],
                      last_calib['rms_px'])

    cv2.destroyAllWindows()
    source.release()


if __name__ == '__main__':
    main()
