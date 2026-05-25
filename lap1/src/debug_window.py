"""Debug window: shows detection pipeline stages and parameter sliders."""
import collections

import numpy as np
import cv2
from PyQt6 import QtCore, QtWidgets, QtGui

from .constants import GREEN_HSV_LO, GREEN_HSV_HI, GREEN_MIN_V
from .detection import detect_green_gate

FRAME_BUFFER_SIZE = 120  # ~4 s at 30 fps


class DebugWindow(QtWidgets.QWidget):
    """Shows HSV, raw mask, morphed mask panels + parameter tuning sliders.

    Supports pause (freeze frame), prev/next to scrub through a ring buffer,
    and re-runs detection with current slider params when paused.
    """

    params_changed = QtCore.pyqtSignal(dict)

    def __init__(self, cam_cx=None, cam_cy=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gate Detection Debug")
        self._cam_cx = cam_cx
        self._cam_cy = cam_cy
        self._paused = False
        self._frame_buf = collections.deque(maxlen=FRAME_BUFFER_SIZE)
        self._buf_idx = -1  # index into buffer when paused

        self._params = {
            "h_lo": int(GREEN_HSV_LO[0]), "h_hi": int(GREEN_HSV_HI[0]),
            "s_lo": int(GREEN_HSV_LO[1]), "s_hi": int(GREEN_HSV_HI[1]),
            "v_lo": int(GREEN_HSV_LO[2]), "v_hi": int(GREEN_HSV_HI[2]),
            "min_v": int(GREEN_MIN_V) if GREEN_MIN_V is not None else 0,
            "kernel": 5, "min_area_frac": 0.01,
        }

        # --- Image panels ---
        self._panels = {}
        panels_layout = QtWidgets.QHBoxLayout()
        for name in ("HSV", "Raw Mask", "Morphed Mask"):
            lbl = QtWidgets.QLabel(name)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumSize(200, 150)
            lbl.setStyleSheet("border: 1px solid gray;")
            self._panels[name] = lbl
            col = QtWidgets.QVBoxLayout()
            title = QtWidgets.QLabel(name)
            title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            col.addWidget(title)
            col.addWidget(lbl)
            panels_layout.addLayout(col)

        # --- Buttons ---
        btn_layout = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("◀ Prev")
        self._btn_pause = QtWidgets.QPushButton("⏸ Pause")
        self._btn_next = QtWidgets.QPushButton("Next ▶")
        self._frame_label = QtWidgets.QLabel("")
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_pause.clicked.connect(self._on_pause)
        self._btn_next.clicked.connect(self._on_next)
        btn_layout.addWidget(self._btn_prev)
        btn_layout.addWidget(self._btn_pause)
        btn_layout.addWidget(self._btn_next)
        btn_layout.addWidget(self._frame_label)
        btn_layout.addStretch()

        # --- Sliders ---
        sliders_layout = QtWidgets.QGridLayout()
        self._sliders = {}
        slider_defs = [
            ("h_lo", "H Low", 0, 179),
            ("h_hi", "H High", 0, 179),
            ("s_lo", "S Low", 0, 255),
            ("s_hi", "S High", 0, 255),
            ("v_lo", "V Low", 0, 255),
            ("v_hi", "V High", 0, 255),
            ("min_v", "Min V (brightness)", 0, 255),
            ("kernel", "Morph Kernel", 1, 21),
        ]
        for row, (key, label, lo, hi) in enumerate(slider_defs):
            lbl = QtWidgets.QLabel(label)
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(self._params[key])
            val_lbl = QtWidgets.QLabel(str(self._params[key]))
            val_lbl.setMinimumWidth(30)
            slider.valueChanged.connect(lambda v, k=key, vl=val_lbl: self._on_slider(k, v, vl))
            sliders_layout.addWidget(lbl, row, 0)
            sliders_layout.addWidget(slider, row, 1)
            sliders_layout.addWidget(val_lbl, row, 2)
            self._sliders[key] = slider

        # Min area slider (float, scaled x1000)
        row = len(slider_defs)
        lbl = QtWidgets.QLabel("Min Area %")
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(self._params["min_area_frac"] * 1000))
        val_lbl = QtWidgets.QLabel(f"{self._params['min_area_frac']:.3f}")
        val_lbl.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v, vl=val_lbl: self._on_area_slider(v, vl))
        sliders_layout.addWidget(lbl, row, 0)
        sliders_layout.addWidget(slider, row, 1)
        sliders_layout.addWidget(val_lbl, row, 2)

        # --- Layout ---
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(panels_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(sliders_layout)
        self.setLayout(main_layout)

    # --- Public API ---

    @property
    def paused(self):
        return self._paused

    @property
    def params(self):
        return dict(self._params)

    def set_camera(self, cx, cy):
        self._cam_cx = cx
        self._cam_cy = cy

    def push_frame(self, rgb_img):
        """Store a frame in the ring buffer. If not paused, run detection and update panels."""
        self._frame_buf.append(rgb_img.copy())
        if not self._paused:
            self._buf_idx = len(self._frame_buf) - 1
            self._run_detection(rgb_img)
            self._update_frame_label()

    def update_stages(self, stages):
        """Direct stage update (used when not paused, called from controller)."""
        if self._paused:
            return
        self._show_stages(stages)

    # --- Internal ---

    def _run_detection(self, rgb_img):
        """Run detection on a frame with current params and display stages."""
        if rgb_img is None or self._cam_cx is None:
            return
        det = detect_green_gate(rgb_img, self._cam_cx, self._cam_cy, params=self._params)
        self._show_stages(det.get("stages"))

    def _show_stages(self, stages):
        if stages is None:
            return
        mapping = {
            "HSV": stages.get("hsv"),
            "Raw Mask": stages.get("mask_raw"),
            "Morphed Mask": stages.get("mask"),
        }
        for name, img in mapping.items():
            if img is None:
                continue
            lbl = self._panels[name]
            if img.ndim == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
                h, w = rgb.shape[:2]
                qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0],
                                    QtGui.QImage.Format.Format_RGB888)
            else:
                h, w = img.shape[:2]
                qimg = QtGui.QImage(img.data, w, h, img.strides[0],
                                    QtGui.QImage.Format.Format_Grayscale8)
            pixmap = QtGui.QPixmap.fromImage(qimg.copy())
            lbl.setPixmap(pixmap.scaled(
                lbl.width(), lbl.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio))

    def _current_frame(self):
        if not self._frame_buf or self._buf_idx < 0:
            return None
        idx = max(0, min(self._buf_idx, len(self._frame_buf) - 1))
        return self._frame_buf[idx]

    def _update_frame_label(self):
        total = len(self._frame_buf)
        cur = self._buf_idx + 1 if self._buf_idx >= 0 else 0
        state = "PAUSED" if self._paused else "LIVE"
        self._frame_label.setText(f"{state}  frame {cur}/{total}")

    # --- Button callbacks ---

    def _on_pause(self):
        self._paused = not self._paused
        self._btn_pause.setText("▶ Resume" if self._paused else "⏸ Pause")
        if self._paused:
            self._buf_idx = len(self._frame_buf) - 1
        self._update_frame_label()

    def _on_prev(self):
        if not self._paused:
            self._paused = True
            self._btn_pause.setText("▶ Resume")
            self._buf_idx = len(self._frame_buf) - 1
        if self._buf_idx > 0:
            self._buf_idx -= 1
        self._run_detection(self._current_frame())
        self._update_frame_label()

    def _on_next(self):
        if not self._paused:
            return
        if self._buf_idx < len(self._frame_buf) - 1:
            self._buf_idx += 1
        self._run_detection(self._current_frame())
        self._update_frame_label()

    # --- Slider callbacks ---

    def _on_slider(self, key, value, val_lbl):
        if key == "kernel" and value % 2 == 0:
            value += 1
            self._sliders[key].blockSignals(True)
            self._sliders[key].setValue(value)
            self._sliders[key].blockSignals(False)
        self._params[key] = value
        val_lbl.setText(str(value))
        self.params_changed.emit(dict(self._params))
        if self._paused:
            self._run_detection(self._current_frame())

    def _on_area_slider(self, value, val_lbl):
        self._params["min_area_frac"] = value / 1000.0
        val_lbl.setText(f"{self._params['min_area_frac']:.3f}")
        self.params_changed.emit(dict(self._params))
        if self._paused:
            self._run_detection(self._current_frame())
