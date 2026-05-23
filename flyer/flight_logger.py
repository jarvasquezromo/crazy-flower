"""Flight logging utilities and replay helper.

This module records planned and actual flight data to JSON and can replay
that log with the Visualisation class.
"""
import json
import os
import time
from datetime import datetime

import numpy as np

from trajectory import Trajectory
from visualisation import Visualisation


class FlightLogger:
    def __init__(self, output_dir="logs", flight_id=None, metadata=None):
        self.output_dir = output_dir
        self.flight_id = flight_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metadata = metadata or {}
        self._planned_points = []
        self._samples = []
        self._start_time = time.time()

    def set_planned_points(self, points):
        self._planned_points = [list(p) for p in points]

    def log_sample(self, sensor_data, cmd=None, target_pos=None, target_vel=None, note=None):
        entry = {
            "t": float(sensor_data.get("t", 0.0)),
            "pos": [
                float(sensor_data.get("x_global", 0.0)),
                float(sensor_data.get("y_global", 0.0)),
                float(sensor_data.get("z_global", 0.0)),
            ],
            "yaw": float(sensor_data.get("yaw", 0.0)),
        }
        if cmd is not None:
            entry["cmd"] = [float(cmd[0]), float(cmd[1]), float(cmd[2]), float(cmd[3])]
        if target_pos is not None:
            entry["target_pos"] = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
        if target_vel is not None:
            entry["target_vel"] = [float(target_vel[0]), float(target_vel[1]), float(target_vel[2])]
        if note:
            entry["note"] = str(note)

        self._samples.append(entry)

    def save_json(self):
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            "flight_id": self.flight_id,
            "created_at": datetime.now().isoformat(),
            "metadata": self.metadata,
            "planned_points": self._planned_points,
            "samples": self._samples,
            "duration_s": time.time() - self._start_time,
        }
        path = os.path.join(self.output_dir, f"flight_{self.flight_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path


def replay_log(json_path, speed=1.0):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    planned = data.get("planned_points", [])
    samples = data.get("samples", [])

    traj = Trajectory()
    if planned:
        traj.generate_trajectory(planned)

    vis = Visualisation()
    if planned:
        vis.set_trajectory(traj)
    vis.start()

    last_t = None
    for sample in samples:
        t = float(sample.get("t", 0.0))
        if last_t is not None:
            dt = max(t - last_t, 0.0)
            time.sleep(dt / max(speed, 1e-3))
        last_t = t

        pos = sample.get("pos", [0.0, 0.0, 0.0])
        yaw = sample.get("yaw", 0.0)
        vis.update_sensor({
            "x_global": pos[0],
            "y_global": pos[1],
            "z_global": pos[2],
            "yaw": yaw,
        })
        target_pos = sample.get("target_pos")
        target_vel = sample.get("target_vel")
        if target_pos is not None or target_vel is not None:
            vis.update_target(target_pos, target_vel)
        if traj is not None:
            est_pos = traj.get_position(t)
            if est_pos is not None:
                vis.update_estimated_position(est_pos)
        vis.refresh()

    # Keep the window open until closed by user
    try:
        while True:
            vis.refresh()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        vis.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python flight_logger.py <path-to-flight.json> [speed]")
        sys.exit(1)

    path = sys.argv[1]
    speed = 1.0
    if len(sys.argv) > 2:
        try:
            speed = float(sys.argv[2])
        except ValueError:
            pass

    replay_log(path, speed=speed)
