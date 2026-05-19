#!/usr/bin/env python3
"""
Crazyflie FPV example.

What happens when this script runs:
  * Fly commands are sent to the Crazyflie with cflib over Crazyradio using URI.
    Modify URI below to match your Crazyflie setup.
  * Vision runs separately over WiFi: the laptop must be connected to the
    AI-deck WiFi. The AI-deck sends camera frames as a UDP stream to LOCAL_PORT.
  * Each frame is received as grayscale JPEG data, decoded with OpenCV, and
    shown in a PyQt window. This example expects 324 x 244 images.
  * The UDP socket and Qt event queue can briefly cache received data. If the
    network or GUI falls behind, displayed frames may be delayed instead of
    always showing the newest camera frame immediately. The images may also be
    slightly distorted, though they looked fine in our experimental environment.
  * If the script cannot start because LOCAL_PORT is already in use, a previous
    run may still be alive. See the FAQ for how to free the port on your OS.

Keys:  arrows = pitch/roll,  A/D = yaw,  W/S = up/down,  Space = stop.
       R = start recording,  E = stop recording and save.
"""
import os
import cv2
import sys
import time
import socket
import struct
import datetime
import contextlib
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig

from PyQt6 import QtCore, QtGui, QtWidgets

@contextlib.contextmanager
def _muted_stderr():
    saved = os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)

# --- Configure these for your setup ---
URI         = 'radio://0/70/2M/E7E7E7E705'
AIDECK_IP   = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT  = 5001
START_MAGIC = b'FER'
SPEED       = 0.6

CPX_HEADER_SIZE  = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE  = 11
IMG_WIDTH        = 324
IMG_HEIGHT       = 244
MIN_JPEG_BYTES   = 5000

# --- Recording settings ---
RECORDING_FPS     = 10       # Approximate FPS for the saved video
RECORDINGS_DIR    = 'recordings'  # Folder where videos will be saved


class UdpVideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.bind(('0.0.0.0', LOCAL_PORT))
        sock.sendto(START_MAGIC, (AIDECK_IP, AIDECK_PORT))

        buffer = bytearray()
        expected_size = 0
        receiving = False

        while True:
            data, _ = sock.recvfrom(2048)
            if len(data) < CPX_HEADER_SIZE:
                continue
            payload = data[CPX_HEADER_SIZE:]

            if len(payload) >= IMG_HEADER_SIZE and payload[0] == IMG_HEADER_MAGIC:
                _, w, h, _, _, size = struct.unpack(
                    '<BHHBBI', payload[:IMG_HEADER_SIZE])
                if w == IMG_WIDTH and h == IMG_HEIGHT and 0 < size < 65536:
                    expected_size = size
                    buffer = bytearray()
                    receiving = True
                    continue

            if not receiving:
                continue

            buffer.extend(payload)

            if len(buffer) >= expected_size:
                self._decode_and_emit(buffer)
                receiving = False

    def _decode_and_emit(self, buffer):
        soi = buffer.find(b'\xff\xd8')
        eoi = buffer.rfind(b'\xff\xd9')
        if soi < 0 or eoi <= soi:
            return
        jpeg_len = eoi + 2 - soi
        if jpeg_len < MIN_JPEG_BYTES:
            return
        jpeg = np.frombuffer(buffer, np.uint8, count=jpeg_len, offset=soi)
        with _muted_stderr():
            img = cv2.imdecode(jpeg, cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (IMG_HEIGHT, IMG_WIDTH):
            return
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.frame_ready.emit(img)


class FPVWindow(QtWidgets.QWidget):
    _connected_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Crazyflie FPV')

        self.image_label = QtWidgets.QLabel('Waiting for video...')
        self.status_label = QtWidgets.QLabel(f'Connecting to {URI}...')
        self.record_label = QtWidgets.QLabel('')
        self.battery_label = QtWidgets.QLabel('')
        self.record_label.setStyleSheet('color: red; font-weight: bold;')

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.record_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.battery_label)

        self.hover = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'height': 0.3}
        self._held = set()

        # --- Recording state ---
        self._video_writer = None
        self._is_recording = False
        os.makedirs(RECORDINGS_DIR, exist_ok=True)

        self.video = UdpVideoThread(self)
        self.video.frame_ready.connect(self._show_frame)
        self.video.start()

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(rw_cache='cache')
        self._connected_signal.connect(self._on_connected)
        self.cf.connected.add_callback(self._connected_signal.emit)
        self.cf.open_link(URI)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._send_setpoint)
        self._timer.setInterval(100)

        self._last_battery_level = None
        self.fly_mode = False

    def _on_connected(self, uri):
        self.status_label.setText(f'Connected to {uri}  |  R = record  |  E = stop & save')
        self.cf.supervisor.send_arming_request(True)
        self._timer.start()

        self._setup_battery_logging()
        self._start_time = time.time()

    def _battery_callback(self, timestamp, data, logconf):
        vbat = data['pm.vbat']
        if self._last_battery_level is None:
            print (f"Battery: {vbat:.2f}V")
        self._last_battery_level = vbat

    def _setup_battery_logging(self):
        # Create a log configuration with a 1-second (1000ms) period
        log_conf = LogConfig(name='Battery', period_in_ms=1000)
        log_conf.add_variable('pm.vbat', 'float')
        
        try:
            self.cf.log.add_config(log_conf)
            # Register the callback
            log_conf.data_received_cb.add_callback(self._battery_callback)
            # Start the logging
            log_conf.start()
        except KeyError as e:
            print(f'Could not setup log configuration: {str(e)}')
        except AttributeError:
            print('Crazyflie not connected or TOC not downloaded')

    def _show_frame(self, img):
        if img.ndim == 2:
            h, w = img.shape
            qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
        else:
            h, w, _ = img.shape
            qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.scaled(w * 2, h * 2)))

        # Feed the current frame into the video writer if recording
        if self._is_recording and self._video_writer is not None:
            # VideoWriter expects BGR; img is RGB (or grayscale)
            if img.ndim == 2:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self._video_writer.write(frame_bgr)

        if self._last_battery_level is not None:
            self.battery_label.setText(f"Battery: {self._last_battery_level:.2f}V")

    def _start_recording(self):
        if self._is_recording:
            return  # Already recording

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(RECORDINGS_DIR, f'fpv_{timestamp}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(
            filepath, fourcc, RECORDING_FPS, (IMG_WIDTH, IMG_HEIGHT))

        if not self._video_writer.isOpened():
            self.status_label.setText('ERROR: Could not open video writer.')
            self._video_writer = None
            return

        self._is_recording = True
        self._current_recording_path = filepath
        self.record_label.setText(f'● REC  →  {filepath}')
        self.status_label.setText('Recording…  Press E to stop and save.')

    def _stop_recording(self):
        if not self._is_recording:
            return  # Not currently recording

        self._is_recording = False
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        self.record_label.setText('')
        self.status_label.setText(
            f'Saved: {self._current_recording_path}  |  R = record  |  E = stop & save')

    def _send_setpoint(self):
        if self.fly_mode:
            self.cf.commander.send_hover_setpoint(
                self.hover['x'], self.hover['y'],
                self.hover['yaw'], self.hover['height'])

    def _update_velocity(self):
        K = QtCore.Qt.Key
        vx = (K.Key_Up in self._held) * SPEED - (K.Key_Down in self._held) * SPEED
        vy = (K.Key_Left in self._held) * SPEED - (K.Key_Right in self._held) * SPEED
        yaw = (K.Key_D in self._held) * 70.0 - (K.Key_A in self._held) * 70.0
        self.hover['x'], self.hover['y'], self.hover['yaw'] = vx, vy, yaw

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        k = event.key()
        if k == QtCore.Qt.Key.Key_Space:
            if self.fly_mode:
                self.cf.commander.send_stop_setpoint()
                self._timer.stop()
            else:
                self.fly_mode = True
        elif k == QtCore.Qt.Key.Key_W:
            self.hover['height'] += 0.1
        elif k == QtCore.Qt.Key.Key_S:
            self.hover['height'] -= 0.1
        elif k == QtCore.Qt.Key.Key_R:
            self._start_recording()
        elif k == QtCore.Qt.Key.Key_E:
            self._stop_recording()
        else:
            self._held.add(k)
            self._update_velocity()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        self._held.discard(event.key())
        self._update_velocity()

    def closeEvent(self, event):
        if self._last_battery_level is not None:
            print(f"Battery: {self._last_battery_level:.2f}V")
        print (f"Time of connection: {time.time() - self._start_time:.2f} seg")

        self._timer.stop()
        # Ensure any active recording is cleanly saved on window close
        if self._is_recording:
            self._stop_recording()
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow()
    win.show()
    sys.exit(app.exec())