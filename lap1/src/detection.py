"""Gate detection: HSV thresholding + polygon validation."""
import cv2
import numpy as np

from .constants import (
    GREEN_HSV_LO, GREEN_HSV_HI, GREEN_MIN_V, MIN_GREEN_AREA_FRAC,
    MORPH_KERNEL, GATE_MIN_VERTICES, GATE_MAX_VERTICES,
    GATE_ASPECT_MIN, GATE_ASPECT_MAX, GATE_MIN_SOLIDITY, GATE_APPROX_EPS,
)


def _order_corners(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float64)


def _quad_corners(cnt, approx):
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(np.float64)
    else:
        pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float64)
    ordered = _order_corners(pts)
    if len(np.unique(np.round(ordered, 1), axis=0)) != 4:
        return None
    return ordered


def _gate_candidate(cnt, img_w, img_h, cam_cx, cam_cy):
    area = float(cv2.contourArea(cnt))
    if area < (MIN_GREEN_AREA_FRAC * float(img_w * img_h)):
        return None

    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return None

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

    m = cv2.moments(approx)
    if abs(m.get("m00", 0.0)) < 1e-6:
        pts = approx.reshape(-1, 2).astype(np.float64)
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())
    else:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

    ex = (cx - cam_cx) / max(0.5 * img_w, 1.0)
    ey = (cy - cam_cy) / max(0.5 * img_h, 1.0)

    return {
        "found": True, "cx": cx, "cy": cy, "area": area,
        "bbox": (x, y, bw, bh), "ex": ex, "ey": ey,
        "approx": approx, "corners": _quad_corners(cnt, approx),
    }


def detect_green_gate(rgb_img, cam_cx, cam_cy):
    """Detect gate(s). Returns dict with best candidate (rightmost cx)."""
    h, w = rgb_img.shape[:2]
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask_1 = cv2.inRange(hsv, GREEN_HSV_LO, GREEN_HSV_HI)

    if GREEN_MIN_V is not None:
        v_mask = np.where(hsv[:, :, 2] >= int(GREEN_MIN_V), np.uint8(255), np.uint8(0))
        mask = cv2.bitwise_and(mask_1, v_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        cand = _gate_candidate(cnt, w, h, cam_cx, cam_cy)
        if cand is not None:
            candidates.append(cand)

    if not candidates:
        return {"found": False, "mask": rgb_img, "candidates": []}

    # TODO: select the one its in the section that we are loking for
    best = dict(max(candidates, key=lambda c: c["cx"]))
    best["mask"] = mask
    best["candidates"] = candidates
    best["n_candidates"] = len(candidates)
    return best
