#!/usr/bin/env python3
"""Race-course gate zones and a top-down visualisation widget.

Course geometry
---------------
The course is split into 12 angular *zones* (30 deg wide each) on a circle
centred at ``COURSE_CENTER``. Zone 5 straddles the +X axis (centred at 0 deg,
spanning +/- 15 deg) and zone numbers increase counter-clockwise, so zone *k*
is centred at ``(k - 5) * 30`` degrees.

The five race gates occupy every other zone:

    gate i (1-based) -> zone (2*i - 1)
      gate1 -> zone1   gate2 -> zone3   gate3 -> zone5
      gate4 -> zone7   gate5 -> zone9

so gate *i* sits at bearing ``(i - 3) * 60`` deg from the centre
(-120, -60, 0, +60, +120). Ground-truth gate coordinates are read from
``gates_xyz.py`` (rows in gate order 1..5: ``[x, y, z, yaw_deg]``).

Two classes
-----------
``GateZoneMap``    Pure-geometry helper: maps gate index -> zone -> bearing
                   window, and validates/snaps a vision estimate into the
                   expected gate's zone.
``GateMapWidget``  A Qt top-down map drawing the zones, the ground-truth gate
                   positions, the current vision estimate and the drone.
"""
import math
import os
import re

from PyQt6 import QtCore, QtGui, QtWidgets

# --- Course constants ---
COURSE_CENTER = (1.15, 0.0)   # world XY the zones radiate from
ZONE_COUNT = 12               # 12 zones around the full circle
ZONE_WIDTH_DEG = 30.0         # each zone is 30 deg wide
ZONE_HALF_DEG = ZONE_WIDTH_DEG / 2.0   # +/- 15 deg about the zone centre
N_GATES = 5

GATES_XYZ_PATH = os.path.join(os.path.dirname(__file__), 'gates_xyz.py')

# How far outside the strict +/- 15 deg window a detection may land and still
# be accepted (it is then snapped back into the window). Absorbs vision noise.
DEFAULT_ACCEPT_MARGIN_DEG = 15.0


def _wrap_deg(angle):
    """Wrap an angle to (-180, 180] degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def load_gates_xyz(path=GATES_XYZ_PATH):
    """Load ground-truth gate rows from ``gates_xyz.py``.

    The file is hand-edited and may have small syntax slips (e.g. a missing
    comma between rows), so it is parsed leniently by extracting each
    ``[ ... ]`` row of numbers rather than importing/eval'ing it. Returns
    ``{gate_index (1-based): (x, y, z, yaw_deg)}``; on failure returns an empty
    dict so callers can fall back to an idealised ring.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return {}

    gates = {}
    # Innermost bracket groups (rows) contain no nested brackets.
    for i, body in enumerate(re.findall(r'\[([^\[\]]+)\]', text), start=1):
        nums = re.findall(r'-?\d+\.?\d*', body)
        if len(nums) < 3:
            continue
        x, y, z = (float(nums[0]), float(nums[1]), float(nums[2]))
        yaw = float(nums[3]) if len(nums) >= 4 else 0.0
        gates[i] = (x, y, z, yaw)
    return gates


class GateZoneMap:
    """Geometry of the gate zones plus the validate-and-snap check."""

    def __init__(self, center=COURSE_CENTER, gates=None,
                 accept_margin_deg=DEFAULT_ACCEPT_MARGIN_DEG):
        self.cx, self.cy = float(center[0]), float(center[1])
        self.accept_margin_deg = float(accept_margin_deg)
        self.gates = gates if gates is not None else load_gates_xyz()

    # --- index / angle mapping ---
    @staticmethod
    def gate_zone(gate_index):
        """Zone number (1..12) that gate ``gate_index`` (1-based) lives in."""
        return 2 * gate_index - 1

    @staticmethod
    def zone_center_deg(zone):
        """Bearing of a zone centre: zone 5 = 0 deg, +30 deg per zone, CCW."""
        return _wrap_deg((zone - 5) * ZONE_WIDTH_DEG)

    def gate_center_deg(self, gate_index):
        """Expected bearing of gate ``gate_index`` from the course centre."""
        return self.zone_center_deg(self.gate_zone(gate_index))

    # --- per-point geometry relative to the course centre ---
    def bearing_deg(self, x, y):
        return math.degrees(math.atan2(y - self.cy, x - self.cx))

    def radius(self, x, y):
        return math.hypot(x - self.cx, y - self.cy)

    def bearing_error_deg(self, x, y, gate_index):
        """Signed angular distance of (x, y) from gate ``gate_index``'s zone."""
        return _wrap_deg(self.bearing_deg(x, y) - self.gate_center_deg(gate_index))

    def in_zone(self, x, y, gate_index, margin_deg=0.0):
        """True if (x, y) falls within gate's zone (optionally widened)."""
        return abs(self.bearing_error_deg(x, y, gate_index)) <= (ZONE_HALF_DEG + margin_deg)

    def validate_and_snap(self, gate_world, gate_index):
        """Validate a vision estimate against the gate's zone, then snap it in.

        ``gate_world`` is ``(x, y, z)`` in the world frame. Returns a snapped
        ``(x, y, z)`` whose bearing is clamped into the gate's +/- 15 deg zone
        (radius and height preserved), or ``None`` if the estimate is too far
        outside the zone (beyond ``accept_margin_deg``) and should be rejected.
        """
        gx, gy, gz = gate_world
        r = self.radius(gx, gy)
        center = self.gate_center_deg(gate_index)
        diff = _wrap_deg(self.bearing_deg(gx, gy) - center)

        if abs(diff) > (ZONE_HALF_DEG + self.accept_margin_deg):
            return None

        clamped = max(-ZONE_HALF_DEG, min(ZONE_HALF_DEG, diff))
        theta = math.radians(center + clamped)
        nx = self.cx + r * math.cos(theta)
        ny = self.cy + r * math.sin(theta)
        return (nx, ny, gz)

    def real_gate(self, gate_index):
        """Ground-truth ``(x, y, z, yaw_deg)`` for a gate, or ``None``."""
        return self.gates.get(gate_index)


class GateMapWidget(QtWidgets.QWidget):
    """Top-down map: zones, ground-truth gates, vision estimate, drone."""

    # Colours (RGB).
    _BG = QtGui.QColor(25, 25, 30)
    _GRID = QtGui.QColor(70, 70, 80)
    _ZONE = QtGui.QColor(60, 110, 160, 70)
    _ZONE_ACTIVE = QtGui.QColor(90, 180, 90, 110)
    _REAL = QtGui.QColor(60, 220, 90)
    _EST = QtGui.QColor(235, 70, 70)
    _DRONE = QtGui.QColor(80, 160, 255)
    _TEXT = QtGui.QColor(220, 220, 220)

    def __init__(self, zone_map, parent=None):
        super().__init__(parent)
        self.zone_map = zone_map
        self.setMinimumSize(360, 360)

        self._drone = None        # (x, y, yaw_deg)
        self._est_gate = None     # (x, y, z)
        self._target_gate = 1     # 1-based index of the gate being chased
        self._est_in_zone = None  # None / True / False (last estimate verdict)

        # Drawing radius for the zone wedges (a bit past the gate ring).
        rs = [self.zone_map.radius(x, y)
              for (x, y, *_rest) in self.zone_map.gates.values()]
        self._wedge_r = (max(rs) if rs else 1.0) + 0.30

        # transform state, recomputed each paint
        self._minx = self._maxy = 0.0
        self._scale = 1.0
        self._offx = self._offy = 0.0

    # --- public API (call from the control loop) ---
    def update_state(self, drone=None, est_gate=None, target_gate=None,
                     est_in_zone=None):
        if drone is not None:
            self._drone = drone
        if target_gate is not None:
            self._target_gate = int(target_gate)
        self._est_gate = est_gate            # may be None to clear
        self._est_in_zone = est_in_zone
        self.update()

    # --- coordinate transform ---
    def _compute_transform(self):
        cx, cy = self.zone_map.cx, self.zone_map.cy
        xs = [cx - self._wedge_r, cx + self._wedge_r]
        ys = [cy - self._wedge_r, cy + self._wedge_r]
        for (x, y, *_r) in self.zone_map.gates.values():
            xs.append(x)
            ys.append(y)
        if self._drone is not None:
            xs.append(self._drone[0])
            ys.append(self._drone[1])
        if self._est_gate is not None:
            xs.append(self._est_gate[0])
            ys.append(self._est_gate[1])

        pad = 0.35
        minx, maxx = min(xs) - pad, max(xs) + pad
        miny, maxy = min(ys) - pad, max(ys) + pad
        span_x = max(maxx - minx, 1e-3)
        span_y = max(maxy - miny, 1e-3)

        m = 16  # pixel margin
        w, h = self.width() - 2 * m, self.height() - 2 * m
        scale = min(w / span_x, h / span_y)

        self._minx, self._maxy = minx, maxy
        self._scale = scale
        # centre the content within the widget
        self._offx = m + (w - span_x * scale) / 2.0
        self._offy = m + (h - span_y * scale) / 2.0

    def _w2p(self, x, y):
        px = self._offx + (x - self._minx) * self._scale
        py = self._offy + (self._maxy - y) * self._scale   # flip Y (up = up)
        return QtCore.QPointF(px, py)

    def _polar_pt(self, bearing_deg, radius):
        """World point at a bearing/radius from the course centre."""
        a = math.radians(bearing_deg)
        return (self.zone_map.cx + radius * math.cos(a),
                self.zone_map.cy + radius * math.sin(a))

    # --- painting ---
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), self._BG)
        self._compute_transform()
        self._draw_zones(p)
        self._draw_ring(p)
        self._draw_real_gates(p)
        self._draw_est_gate(p)
        self._draw_drone(p)
        self._draw_legend(p)
        p.end()

    def _draw_zones(self, p):
        zm = self.zone_map
        center_px = self._w2p(zm.cx, zm.cy)

        # Faint boundary lines between all 12 zones (edges at 15 + 30k deg).
        p.setPen(QtGui.QPen(self._GRID, 1, QtCore.Qt.PenStyle.DotLine))
        for k in range(ZONE_COUNT):
            edge = ZONE_HALF_DEG + k * ZONE_WIDTH_DEG
            ex, ey = self._polar_pt(edge, self._wedge_r)
            p.drawLine(center_px, self._w2p(ex, ey))

        # Filled wedges for the 5 active gate zones.
        for gate_idx in range(1, N_GATES + 1):
            center = zm.gate_center_deg(gate_idx)
            is_target = (gate_idx == self._target_gate)
            color = self._ZONE_ACTIVE if is_target else self._ZONE
            path = QtGui.QPainterPath(center_px)
            steps = 12
            for s in range(steps + 1):
                ang = center - ZONE_HALF_DEG + (ZONE_WIDTH_DEG * s / steps)
                wx, wy = self._polar_pt(ang, self._wedge_r)
                path.lineTo(self._w2p(wx, wy))
            path.closeSubpath()
            p.fillPath(path, color)

            # Zone label near the wedge mid-arc.
            lx, ly = self._polar_pt(center, self._wedge_r * 0.78)
            p.setPen(self._TEXT)
            lp = self._w2p(lx, ly)
            p.drawText(QtCore.QRectF(lp.x() - 22, lp.y() - 9, 44, 18),
                       QtCore.Qt.AlignmentFlag.AlignCenter,
                       f"Z{zm.gate_zone(gate_idx)}")

    def _draw_ring(self, p):
        zm = self.zone_map
        # Circle through the mean gate radius, plus the centre marker.
        rs = [zm.radius(x, y) for (x, y, *_r) in zm.gates.values()]
        if rs:
            r = sum(rs) / len(rs)
            c = self._w2p(zm.cx, zm.cy)
            rpx = r * self._scale
            p.setPen(QtGui.QPen(self._GRID, 1, QtCore.Qt.PenStyle.DashLine))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawEllipse(c, rpx, rpx)
        cpt = self._w2p(zm.cx, zm.cy)
        p.setPen(QtGui.QPen(self._GRID, 1))
        p.drawLine(QtCore.QPointF(cpt.x() - 5, cpt.y()),
                   QtCore.QPointF(cpt.x() + 5, cpt.y()))
        p.drawLine(QtCore.QPointF(cpt.x(), cpt.y() - 5),
                   QtCore.QPointF(cpt.x(), cpt.y() + 5))

    def _draw_real_gates(self, p):
        zm = self.zone_map
        for idx, (x, y, _z, yaw) in zm.gates.items():
            pt = self._w2p(x, y)
            p.setPen(QtGui.QPen(self._REAL, 2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRect(QtCore.QRectF(pt.x() - 6, pt.y() - 6, 12, 12))
            # Orientation tick (gate yaw / facing).
            a = math.radians(yaw)
            tip = QtCore.QPointF(pt.x() + 14 * math.cos(a),
                                 pt.y() - 14 * math.sin(a))
            p.drawLine(pt, tip)
            p.setPen(self._REAL)
            p.drawText(QtCore.QRectF(pt.x() + 6, pt.y() - 20, 36, 16),
                       QtCore.Qt.AlignmentFlag.AlignLeft, f"G{idx}")

    def _draw_est_gate(self, p):
        if self._est_gate is None:
            return
        ex, ey = self._est_gate[0], self._est_gate[1]
        pt = self._w2p(ex, ey)
        # Green outline if the estimate sits in the expected zone, red if not.
        ok = self._est_in_zone
        pen_col = self._REAL if ok else self._EST
        p.setPen(QtGui.QPen(pen_col, 2))
        p.drawLine(QtCore.QPointF(pt.x() - 7, pt.y() - 7),
                   QtCore.QPointF(pt.x() + 7, pt.y() + 7))
        p.drawLine(QtCore.QPointF(pt.x() - 7, pt.y() + 7),
                   QtCore.QPointF(pt.x() + 7, pt.y() - 7))

    def _draw_drone(self, p):
        if self._drone is None:
            return
        x, y, yaw = self._drone
        pt = self._w2p(x, y)
        p.setBrush(self._DRONE)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(pt, 5, 5)
        # Heading arrow.
        a = math.radians(yaw)
        tip = QtCore.QPointF(pt.x() + 20 * math.cos(a),
                             pt.y() - 20 * math.sin(a))
        p.setPen(QtGui.QPen(self._DRONE, 2))
        p.drawLine(pt, tip)

    def _draw_legend(self, p):
        lines = [
            ("real gate", self._REAL),
            ("estimate", self._EST),
            ("drone", self._DRONE),
        ]
        target = f"target: gate {self._target_gate} (zone {self.zone_map.gate_zone(self._target_gate)})"
        y = 16
        p.setPen(self._TEXT)
        p.drawText(8, y, target)
        y += 18
        for label, col in lines:
            p.setPen(QtGui.QPen(col, 2))
            p.drawLine(10, y - 4, 26, y - 4)
            p.setPen(self._TEXT)
            p.drawText(32, y, label)
            y += 16
