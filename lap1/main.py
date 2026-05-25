#!/usr/bin/env python3
"""Lap1 autonomous gate-flying entry point.

Usage:
  python -m lap1.main                          # live flight (AI-deck + Crazyflie)
  python -m lap1.main --video path/to/file.mp4 # video simulation (no drone)
"""
import argparse
import sys

from PyQt6 import QtWidgets

from src.controller import FPVWindow


def main():
    parser = argparse.ArgumentParser(description='Lap1 gate-flying pipeline')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to video file for simulation (skips Crazyflie)')
    parser.add_argument('--replay', type=str, default=None,
                        help='Path to recording directory for synchronized replay')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = FPVWindow(video_path=args.video, replay_dir=args.replay)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
