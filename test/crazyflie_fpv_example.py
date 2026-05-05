#!/usr/bin/env python3
import logging
import struct
import sys
import threading
import warnings
import numpy as np
import cflib.crtp
from cflib.cpx import CPXFunction
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from PyQt6 import QtCore, QtWidgets, QtGui
import cv2

logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore', message='.*TYPE_HOVER_LEGACY.*')
warnings.filterwarnings('ignore', message='.*supervisor subsystem requires CRTP.*')

URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')
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

        self.cf.supervisor.send_arming_request(True)

        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(100)
        self._timer.start()

    def _update_image(self, img):
        bayer = img.reshape((CAM_HEIGHT, CAM_WIDTH))
        color = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)
        h, w, ch = color.shape
        q = QtGui.QImage(color.data, w, h, w * ch, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q.scaled(w * 2, h * 2)))

    def _send_setpoint(self):
        self.cf.commander.send_hover_setpoint(
            self.hover['x'], self.hover['y'], self.hover['yaw'], self.hover['height'])

    def _set_hover(self, key, value):
        if key == 'height':
            self.hover[key] += value
        else:
            self.hover[key] = value * SPEED

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Up:       self._set_hover('x',   1)
        if k == QtCore.Qt.Key.Key_Down:     self._set_hover('x',  -1)
        if k == QtCore.Qt.Key.Key_Left:     self._set_hover('y',   1)
        if k == QtCore.Qt.Key.Key_Right:    self._set_hover('y',  -1)
        if k == QtCore.Qt.Key.Key_A:        self._set_hover('yaw', -70)
        if k == QtCore.Qt.Key.Key_D:        self._set_hover('yaw',  70)
        if k == QtCore.Qt.Key.Key_W:        self._set_hover('height',  0.1)
        if k == QtCore.Qt.Key.Key_S:        self._set_hover('height', -0.1)
        if k == QtCore.Qt.Key.Key_Space:
            self.cf.commander.send_stop_setpoint()
            self._timer.stop()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):    self._set_hover('x', 0)
        if k in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):  self._set_hover('y', 0)
        if k in (QtCore.Qt.Key.Key_A, QtCore.Qt.Key.Key_D):         self._set_hover('yaw', 0)
        if k in (QtCore.Qt.Key.Key_W, QtCore.Qt.Key.Key_S):         self._set_hover('height', 0)

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