#!/usr/bin/env python3
import logging
import struct
import sys
import threading
import warnings
from click import command
import numpy as np
import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from PyQt6 import QtCore, QtWidgets, QtGui
import cv2
from follower import Follower
from trajectory import Trajectory

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore', message='.*TYPE_HOVER_LEGACY.*')
warnings.filterwarnings('ignore', message='.*supervisor subsystem requires CRTP.*')

URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E705')
CAM_WIDTH = 324
CAM_HEIGHT = 244
SPEED = 0.6


class ImageThread(threading.Thread):
    def __init__(self, cpx, callback):
        super().__init__(daemon=True)
        self._cpx = cpx
        self._cb = callback

    def run(self):
        while True:
            p = self._cpx.receivePacket(CPXFunction.APP)
            [magic, width, height, depth, fmt, size] = struct.unpack('<BHHBBI', p.data[0:11])
            if magic == 0xBC:
                buf = bytearray()
                while len(buf) < size:
                    buf.extend(self._cpx.receivePacket(CPXFunction.APP).data)
                self._cb(np.frombuffer(buf, dtype=np.uint8))


class FPVWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.follower = Follower()

        self.setWindowTitle('Crazyflie FPV')

        self.image_label = QtWidgets.QLabel()
        self.status_label = QtWidgets.QLabel('Connecting...')

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': 0.0}

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.open_link(URI)

        if not self.cf.link:
            print('Could not connect')
            sys.exit(1)

        self._img_thread = ImageThread(self.cf.link.cpx, self._update_image)
        self._img_thread.start()

        self.log_conf = LogConfig(name='Position', period_in_ms=100)
        self.log_conf.add_variable('stateEstimate.x', 'float')
        self.log_conf.add_variable('stateEstimate.y', 'float')
        self.log_conf.add_variable('stateEstimate.z', 'float')

        self.cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self.log_pos_callback)

        self.log_conf.start()

        self.cf.supervisor.send_arming_request(True)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.send_trajectory_setpoint)
        self._timer.setInterval(100)
        self._timer.start()

    def _update_image(self, img):
        bayer = img.reshape((CAM_HEIGHT, CAM_WIDTH))
        color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        h, w, ch = color.shape
        q = QtGui.QImage(color.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    def send_trajectory_setpoint(self):
        if self.self.follow_trajectory:
            command = self.follower.follow_planned_trajectory(self.sensor_data)
            self.cf.commander.send_position_setpoint(command[0], command[1], command[2], command[3])

    def log_pos_callback(self, timestamp, data, logconf):
        x = data['stateEstimate.x']
        y = data['stateEstimate.y']
        z = data['stateEstimate.z']
        print(f"Position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_F:
            self.follow_trajectory = not self.follow_trajectory
        if k == QtCore.Qt.Key.Key_Space:
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    def _connected(self, uri):
        self.status_label.setText(f'Connected to {uri}')

    def _disconnected(self, uri):
        print('Disconnected')
        sys.exit(1)

    def closeEvent(self, event):
        self.cf.close_link()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    app.exec()