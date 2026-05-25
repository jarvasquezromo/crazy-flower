"""Camera calibration loading and runtime scaling."""
import json
import os
import cv2
import numpy as np

CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'calibration.json')


def load_calibration(path=None):
    path = path or CALIBRATION_PATH
    with open(path, 'r', encoding='utf-8') as f:
        calib = json.load(f)
    required = ('img_w', 'img_h', 'fx', 'fy', 'cx', 'cy', 'dist_coeffs')
    missing = [k for k in required if k not in calib]
    if missing:
        raise ValueError(f"Missing calibration keys: {missing}")
    return calib


def scaled_calibration(calib, img_w, img_h):
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


def camera_matrix(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_image(rgb_img, cam_mtx, dist_coeffs):
    if dist_coeffs.size == 0 or np.allclose(dist_coeffs, 0.0):
        return rgb_img
    return cv2.undistort(rgb_img.copy(), cam_mtx, dist_coeffs)

