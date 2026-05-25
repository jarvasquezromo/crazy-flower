"""Top-down race-course map widget for real-time drone/gate visualization."""
import math
import os
import re

from PyQt6 import QtCore, QtGui, QtWidgets

COURSE_CENTER = (1.15, 0.0)
ZONE_COUNT = 12
ZONE_WIDTH_DEG = 30.0
ZONE_HALF_DEG = ZONE_WIDTH_DEG / 2.0
N_GATES = 5

GATES_XYZ_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'gates_xyz.py')


def _wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def load_gates_xyz(path=GATES_XYZ_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return {}
    gates = {}
    for i, body in enumerate(re.findall(r'\[([^\[\]]+)\]', text), start=1):
        nums = re.findall(r'-?\d+\.?\d*', body)
        if len(nums) < 3:
            continue
        x, y, z = float(nums[0]), float(nums[1]), float(nums[2])
        yaw = float(nums[3]) if len(nums) >= 4 else 0.0
        gates[i] = (x, y, z, yaw)
    return gates


class GateZoneMap:
    def __init__(self, center=COURSE_CENTER, gates=None):
        self.cx, self.cy = float(center[0]), float(center[1])
        self.gates = gates if gates is not None else load_gates_xyz()

    @staticmethod
    def gate_zone(gate_index):
        return 2 * gate_index - 1

    @staticmethod
    def zone_center_deg(zone):
        return _wrap_deg((zone - 5) * ZONE_WIDTH_DEG)

    def gate_center_deg(self, gate_index):
        return self.zone_center_deg(self.gate_zone(gate_index))

    def bearing_deg(self, x, y):
        return math.degrees(math.atan2(y - self.cy, x - self.cx))

    def radius(self, x, y):
        return math.hypot(x - self.cx, y - self.cy)


class GateMapWidget(QtWidgets.QWidget):
    """Top-down map: zones, ground-truth gates, vision estimate, drone."""

    _BG = QtGui.QColor(25, 25, 30)
    _GRID = QtGui.QColor(70, 70, 80)
    _ZONE = QtGui.QColor(60, 110, 160, 70)
    _ZONE_ACTIVE = QtGui.QColor(90, 180, 90, 110)
    _REAL = QtGui.QColor(60, 220, 90)
    _EST = QtGui.QColor(235, 70, 70)
    _DRONE = QtGui.QColor(80, 160, 255)
    _TEXT = QtGui.QColor(220, 220, 220)

    def __init__(self, zone_map=None, parent=None):
        super().__init__(parent)
        self.zone_map = zone_map or GateZoneMap()
        self.setMinimumSize(300, 300)

        self._drone = None        # (x, y, yaw_deg)
        self._est_gate = None     # (x, y, z)
        self._target_gate = 1

        rs = [self.zone_map.radius(x, y)
              for (x, y, *_) in self.zone_map.gates.values()]
        self._wedge_r = (max(rs) if rs else 1.0) + 0.30

        self._minx = self._maxy = 0.0
        self._scale = 1.0
        self._offx = self._offy = 0.0

    def update_state(self, drone=None, est_gate=None, target_gate=None):
        if drone is not None:
            self._drone = drone
        if target_gate is not None:
            self._target_gate = int(target_gate)
        self._est_gate = est_gate
        self.update()

    def _compute_transform(self):
        cx, cy = self.zone_map.cx, self.zone_map.cy
        xs = [cx - self._wedge_r, cx + self._wedge_r]
        ys = [cy - self._wedge_r, cy + self._wedge_r]
        for (x, y, *_) in self.zone_map.gates.values():
            xs.append(x); ys.append(y)
        if self._drone:
            xs.append(self._drone[0]); ys.append(self._drone[1])
        if self._est_gate:
            xs.append(self._est_gate[0]); ys.append(self._est_gate[1])

        pad = 0.35
        minx, maxx = min(xs) - pad, max(xs) + pad
        miny, maxy = min(ys) - pad, max(ys) + pad
        span_x, span_y = max(maxx - minx, 1e-3), max(maxy - miny, 1e-3)
        m = 12
        w, h = self.width() - 2*m, self.height() - 2*m
        scale = min(w / span_x, h / span_y)
        self._minx, self._maxy, self._scale = minx, maxy, scale
        self._offx = m + (w - span_x * scale) / 2.0
        self._offy = m + (h - span_y * scale) / 2.0

    def _w2p(self, x, y):
        px = self._offx + (x - self._minx) * self._scale
        py = self._offy + (self._maxy - y) * self._scale
        return QtCore.QPointF(px, py)

    def _polar_pt(self, bearing_deg, radius):
        a = math.radians(bearing_deg)
        return (self.zone_map.cx + radius * math.cos(a),
                self.zone_map.cy + radius * math.sin(a))

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), self._BG)
        self._compute_transform()
        self._draw_zones(p)
        self._draw_real_gates(p)
        self._draw_est_gate(p)
        self._draw_drone(p)
        p.end()

    def _draw_zones(self, p):
        zm = self.zone_map
        center_px = self._w2p(zm.cx, zm.cy)
        p.setPen(QtGui.QPen(self._GRID, 1, QtCore.Qt.PenStyle.DotLine))
        for k in range(ZONE_COUNT):
            edge = ZONE_HALF_DEG + k * ZONE_WIDTH_DEG
            ex, ey = self._polar_pt(edge, self._wedge_r)
            p.drawLine(center_px, self._w2p(ex, ey))

        for gate_idx in range(1, N_GATES + 1):
            center = zm.gate_center_deg(gate_idx)
            is_target = (gate_idx == self._target_gate)
            color = self._ZONE_ACTIVE if is_target else self._ZONE
            path = QtGui.QPainterPath(center_px)
            for s in range(13):
                ang = center - ZONE_HALF_DEG + (ZONE_WIDTH_DEG * s / 12)
                wx, wy = self._polar_pt(ang, self._wedge_r)
                path.lineTo(self._w2p(wx, wy))
            path.closeSubpath()
            p.fillPath(path, color)

    def _draw_real_gates(self, p):
        for idx, (x, y, _z, yaw) in self.zone_map.gates.items():
            pt = self._w2p(x, y)
            p.setPen(QtGui.QPen(self._REAL, 2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRect(QtCore.QRectF(pt.x()-5, pt.y()-5, 10, 10))
            a = math.radians(yaw)
            tip = QtCore.QPointF(pt.x() + 12*math.cos(a), pt.y() - 12*math.sin(a))
            p.drawLine(pt, tip)
            p.setPen(self._TEXT)
            p.drawText(int(pt.x()+8), int(pt.y()-6), f"G{idx}")

    def _draw_est_gate(self, p):
        if self._est_gate is None:
            return
        pt = self._w2p(self._est_gate[0], self._est_gate[1])
        p.setPen(QtGui.QPen(self._EST, 2))
        p.drawLine(QtCore.QPointF(pt.x()-6, pt.y()-6), QtCore.QPointF(pt.x()+6, pt.y()+6))
        p.drawLine(QtCore.QPointF(pt.x()-6, pt.y()+6), QtCore.QPointF(pt.x()+6, pt.y()-6))

    def _draw_drone(self, p):
        if self._drone is None:
            return
        x, y, yaw = self._drone
        pt = self._w2p(x, y)
        p.setBrush(self._DRONE)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(pt, 5, 5)
        a = math.radians(yaw)
        tip = QtCore.QPointF(pt.x() + 18*math.cos(a), pt.y() - 18*math.sin(a))
        p.setPen(QtGui.QPen(self._DRONE, 2))
        p.drawLine(pt, tip)
