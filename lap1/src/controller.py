"""FPV window: GUI, video pipeline, Crazyflie connection, keyboard."""
import sys
import threading
import warnings

import cv2
import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from cflib.drivers.crazyradio import _find_devices


from .calibration import load_calibration, scaled_calibration, camera_matrix, undistort_image
from .constants import URI_DEFAULT
from .detection import detect_green_gate
from .gate_map import GateMapWidget
from .state_machine import GateStateMachine
from .video import UdpVideoThread, VideoFileThread

warnings.filterwarnings('ignore', message='.*TYPE_HOVER_LEGACY.*')
warnings.filterwarnings('ignore', message='.*supervisor subsystem requires CRTP.*')


class FPVWindow(QtWidgets.QWidget):
    def __init__(self, video_path=None):
        super().__init__()
        self.setWindowTitle('Crazyflie FPV')
        self._simulation = video_path is not None

        # Calibration
        self._base_calib = load_calibration()
        self._calib = dict(self._base_calib)
        self._img_w = int(self._calib['img_w'])
        self._img_h = int(self._calib['img_h'])
        self._dist_coeffs = np.array(self._calib['dist_coeffs'], dtype=np.float64)

        # UI
        self.image_label = QtWidgets.QLabel()
        self.status_label = QtWidgets.QLabel('Connecting...')
        self.debug_label = QtWidgets.QLabel()
        self.debug_label.setWindowTitle('Debug Mask')
        self.map_widget = GateMapWidget(parent=self)

        layout = QtWidgets.QHBoxLayout()
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.image_label)
        left.addWidget(self.status_label)
        layout.addLayout(left)
        layout.addWidget(self.map_widget)
        self.setLayout(layout)

        # State machine
        self._sm = GateStateMachine(calib=self._calib)
        self._vision_lock = threading.Lock()
        self._vision = {"found": False, "ex": 0.0, "ey": 0.0, "bbox": None, "area": 0.0}
        self._battery_v = None
        self._show_mask = False
        self._debug_mask = None  # stored from video thread, displayed from main thread

        # Crazyflie
        if not self._simulation:
            cflib.crtp.init_drivers()
            URI = uri_helper.uri_from_env(default=URI_DEFAULT)

            # Check if a CrazyRadio dongle is physically connected
            if not _find_devices():
                warnings.warn("CrazyRadio dongle not found — running without drone control.")
                self.cf = None
                self._sm.log_ready = True
            else:
                self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
                self.cf.connected.add_callback(self._connected)
                self.cf.disconnected.add_callback(self._disconnected)
                self.cf.open_link(URI)
                self.cf.supervisor.send_arming_request(True)
        else:
            self.cf = None
            self._sm.log_ready = True

        # Video source
        if self._simulation:
            self.video = VideoFileThread(video_path, loop=True, parent=self)
        else:
            self.video = UdpVideoThread(self._img_w, self._img_h, parent=self)
        self.video.size_detected.connect(self._on_size_detected)
        self.video.frame_ready.connect(self._update_image)
        self.video.start()

        # Control loop
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(20)
        self._timer.start()

    # ─── Calibration ─────────────────────────────────────────────────────

    def _on_size_detected(self, w, h):
        if w == self._base_calib['img_w'] and h == self._base_calib['img_h']:
            self._calib = dict(self._base_calib)
        else:
            self._calib = scaled_calibration(self._base_calib, w, h)
        self._img_w = int(self._calib['img_w'])
        self._img_h = int(self._calib['img_h'])
        self._dist_coeffs = np.array(self._calib['dist_coeffs'], dtype=np.float64)

    @property
    def _cam_mtx(self):
        return camera_matrix(
            float(self._calib['fx']), float(self._calib['fy']),
            float(self._calib['cx']), float(self._calib['cy']))

    # ─── Frame processing ────────────────────────────────────────────────

    def _update_image(self, img):
        color_0 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img
        color = undistort_image(color_0, self._cam_mtx, self._dist_coeffs)

        det = detect_green_gate(color, float(self._calib['cx']), float(self._calib['cy']))
        with self._vision_lock:
            self._vision = {
                "found": bool(det.get("found", False)),
                "ex": float(det.get("ex", 0.0)),
                "ey": float(det.get("ey", 0.0)),
                "bbox": det.get("bbox"),
                "area": float(det.get("area", 0.0)),
                "corners": det.get("corners"),
            }

        # Overlay
        disp = color.copy()
        if det.get("mask") is not None:
            self._debug_mask = det["mask"].copy()
        sel_bbox = det.get("bbox")
        for cand in det.get("candidates", []):
            is_sel = (cand["bbox"] == sel_bbox)
            c = (0, 255, 0) if is_sel else (255, 255, 0)
            t = 2 if is_sel else 1
            bx, by, bw, bh = cand["bbox"]
            cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), c, t)
            if cand.get("approx") is not None:
                cv2.polylines(disp, [cand["approx"]], True, c, t)
        if det.get("found") and sel_bbox:
            cv2.circle(disp, (int(det["cx"]), int(det["cy"])), 4, (255, 0, 0), -1)

        h, w = disp.shape[:2]
        cv2.drawMarker(disp, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(disp,
                    f"state={self._sm.state} gates={self._sm.gates_passed} found={int(det.get('found', False))}",
                    (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if self._battery_v is not None:
            cv2.putText(disp, f"bat={self._battery_v:.2f}V",
                        (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        ch = disp.shape[2] if disp.ndim == 3 else 1
        disp = np.ascontiguousarray(disp)
        self._disp_buf = disp  # prevent GC before Qt paints
        q = QtGui.QImage(self._disp_buf.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    # ─── Control tick ────────────────────────────────────────────────────

    def _tick(self):
        if self._sm.state == "STOP":
            if self.cf:
                self.cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
            return

        with self._vision_lock:
            vision = dict(self._vision)

        done = self._sm.update(vision, self._calib, self._cam_mtx)

        # Update map with drone position and gate estimate
        with self._sm.pos_lock:
            est = self._sm.est
            self.map_widget.update_state(
                drone=(est['x'], est['y'], est['yaw']),
                est_gate=self._sm._gate_world,
                target_gate=self._sm.gates_passed + 1,
            )

        if done:
            if self.cf:
                self.cf.commander.send_stop_setpoint()
            self._timer.stop()

        # Show debug mask window from main thread
        if self._show_mask and self._debug_mask is not None:
            ch = self._debug_mask.shape[2] if self._debug_mask.ndim == 3 else 1
            m = self._debug_mask
            mh, mw = m.shape[:2]
            qi = QtGui.QImage(m.data, mw, mh, mw * ch, QtGui.QImage.Format.Format_RGB888)
            # qi = QtGui.QImage(m.data, mw, mh, mw, QtGui.QImage.Format.Format_Grayscale8)
            self.debug_label.setPixmap(QtGui.QPixmap.fromImage(qi.scaled(mw * 2, mh * 2)))
            if not self.debug_label.isVisible():
                self.debug_label.show()
        elif not self._show_mask and self.debug_label.isVisible():
            self.debug_label.hide()

    # ─── Keyboard ────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Up:    self._sm.pos['x'] += 0.2
        if k == QtCore.Qt.Key.Key_Down:  self._sm.pos['x'] -= 0.2
        if k == QtCore.Qt.Key.Key_Left:  self._sm.pos['y'] += 0.2
        if k == QtCore.Qt.Key.Key_Right: self._sm.pos['y'] -= 0.2
        if k == QtCore.Qt.Key.Key_W:     self._sm.pos['z'] += 0.1
        if k == QtCore.Qt.Key.Key_S:     self._sm.pos['z'] -= 0.1
        if k == QtCore.Qt.Key.Key_A:     self._sm.pos['yaw'] -= 15.0
        if k == QtCore.Qt.Key.Key_D:     self._sm.pos['yaw'] += 15.0
        if k == QtCore.Qt.Key.Key_Escape:
            self._sm.state = "STOP"
        if k == QtCore.Qt.Key.Key_M:
            self._show_mask = not self._show_mask
        if k == QtCore.Qt.Key.Key_Space:
            self._sm.state = "DONE"
            if self.cf:
                self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    # ─── Crazyflie callbacks ─────────────────────────────────────────────

    def _connected(self, uri):
        self.status_label.setText(f'Connected to {uri}')
        lc = LogConfig('StateEst', period_in_ms=20)
        lc.add_variable('stateEstimate.x', 'float')
        lc.add_variable('stateEstimate.y', 'float')
        lc.add_variable('stateEstimate.z', 'float')
        lc.add_variable('stabilizer.yaw', 'float')
        try:
            self.cf.log.add_config(lc)
        except Exception as e:
            print(f'Could not add log config: {e}')
            return
        lc.data_received_cb.add_callback(self._on_log)
        lc.start()
        self._log_cfg = lc

        # Battery logging
        bat_cfg = LogConfig('Battery', period_in_ms=1000)
        bat_cfg.add_variable('pm.vbat', 'float')
        try:
            self.cf.log.add_config(bat_cfg)
            bat_cfg.data_received_cb.add_callback(self._on_battery)
            bat_cfg.start()
            self._bat_cfg = bat_cfg
        except (KeyError, AttributeError) as e:
            print(f'Could not setup battery log: {e}')

    def _on_log(self, _ts, data, _lc):
        with self._sm.pos_lock:
            self._sm.est['x'] = data['stateEstimate.x']
            self._sm.est['y'] = data['stateEstimate.y']
            self._sm.est['z'] = data['stateEstimate.z']
            self._sm.est['yaw'] = data['stabilizer.yaw']
            if not self._sm.log_ready:
                self._sm.seed_position(
                    self._sm.est['x'], self._sm.est['y'],
                    self._sm.est['z'], self._sm.est['yaw'])

    def _on_battery(self, _ts, data, _lc):
        vbat = data['pm.vbat']
        if self._battery_v is None:
            print(f"Battery: {vbat:.2f}V")
        self._battery_v = vbat

    def _disconnected(self, uri):
        print('Disconnected')
        sys.exit(1)

    def closeEvent(self, event):
        self._timer.stop()
        if self.cf:
            if hasattr(self, '_log_cfg'):
                self._log_cfg.stop()
            if hasattr(self, '_bat_cfg'):
                self._bat_cfg.stop()
            self.cf.commander.send_stop_setpoint()
            self.cf.close_link()
