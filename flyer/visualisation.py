"""Live 3D visualization for planned trajectory and drone pose.

Usage:
    from visualisation import Visualisation

    vis = Visualisation()
    vis.set_trajectory(traj)   # traj is an instance of Trajectory
    vis.start()

    # In your log callback or loop:
    vis.update_sensor(sensor_data)

    # When done:
    vis.stop()
"""
import threading
import numpy as np


class Visualisation:
    def __init__(self, trajectory=None, rate_hz=10):
        self.trajectory = trajectory
        self.rate_hz = rate_hz
        self._pos = np.array([0.0, 0.0, 0.0])
        self._path = []
        self._est_pos = None
        self._est_path = []
        self._target_pos = None
        self._target_vel = None
        self._lock = threading.Lock()
        self._sampled = None
        self._fig = None
        self._ax = None
        self._traj_line = None
        self._path_line = None
        self._drone_scatter = None
        self._est_path_line = None
        self._est_scatter = None
        self._target_scatter = None
        self._target_vel_line = None
        self._vel_scale = 0.5

    def set_trajectory(self, trajectory):
        """Attach a Trajectory instance and pre-sample it for plotting."""
        self.trajectory = trajectory
        self._sampled = self._sample_trajectory()

    def _sample_trajectory(self):
        if not self.trajectory or not getattr(self.trajectory, 'segment_times', None):
            return np.zeros((0, 3))

        total_t = sum(self.trajectory.segment_times)
        if total_t <= 0:
            return np.zeros((0, 3))

        n = max(int(total_t * 20), 50)
        ts = np.linspace(0.0, total_t, n)
        pts = []
        for t in ts:
            p = self.trajectory.get_position(t)
            if p is None:
                break
            pts.append(p)
        if len(pts) == 0:
            return np.zeros((0, 3))
        return np.vstack(pts)

    def update_sensor(self, sensor_data):
        """Call from your log callback with sensor_data containing x_global,y_global,z_global."""
        with self._lock:
            self._pos = np.array([
                float(sensor_data.get('x_global', 0.0)),
                float(sensor_data.get('y_global', 0.0)),
                float(sensor_data.get('z_global', 0.0)),
            ])
            self._path.append(self._pos.copy())

    def update_estimated_position(self, pos):
        if pos is None:
            return
        with self._lock:
            self._est_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
            self._est_path.append(self._est_pos.copy())

    def update_target(self, pos, vel=None):
        if pos is None:
            return
        with self._lock:
            self._target_pos = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
            if vel is not None:
                self._target_vel = np.array([float(vel[0]), float(vel[1]), float(vel[2])])

    def start(self):
        if self._fig is not None:
            return

        # Import and create the GUI on the caller thread.
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        plt.ion()
        self._fig = plt.figure(figsize=(8, 6))
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        self._ax.set_zlabel('Z')
        self._ax.set_title('Planned trajectory and drone position')

        if self._sampled is None and self.trajectory is not None:
            self._sampled = self._sample_trajectory()

        if self._sampled is not None and self._sampled.shape[0] > 0:
            xs, ys, zs = self._sampled[:, 0], self._sampled[:, 1], self._sampled[:, 2]
            self._traj_line, = self._ax.plot(xs, ys, zs, '-', color='tab:blue', linewidth=1.5, label='planned')
            self._ax.legend()
            self._ax.set_xlim(np.min(xs) - 0.5, np.max(xs) + 0.5)
            self._ax.set_ylim(np.min(ys) - 0.5, np.max(ys) + 0.5)
            self._ax.set_zlim(max(0, np.min(zs) - 0.5), np.max(zs) + 0.5)

        with self._lock:
            px, py, pz = self._pos.tolist()
        self._drone_scatter = self._ax.scatter([px], [py], [pz], color='red', s=60)
        self._path_line, = self._ax.plot([px], [py], [pz], '-', color='tab:red', linewidth=1.0, label='actual')

        with self._lock:
            est = self._est_pos
            if est is not None:
                ex, ey, ez = est.tolist()
            else:
                ex, ey, ez = px, py, pz
        self._est_scatter = self._ax.scatter([ex], [ey], [ez], color='tab:green', s=40)
        self._est_path_line, = self._ax.plot([ex], [ey], [ez], '--', color='tab:green', linewidth=1.0, label='estimated')

        with self._lock:
            target = self._target_pos
            vel = self._target_vel
            if target is not None:
                tx, ty, tz = target.tolist()
            else:
                tx, ty, tz = px, py, pz
            if vel is not None:
                vx, vy, vz = vel.tolist()
            else:
                vx, vy, vz = 0.0, 0.0, 0.0
        self._target_scatter = self._ax.scatter([tx], [ty], [tz], color='tab:orange', s=50)
        self._target_vel_line, = self._ax.plot(
            [tx, tx + vx * self._vel_scale],
            [ty, ty + vy * self._vel_scale],
            [tz, tz + vz * self._vel_scale],
            '-',
            color='tab:orange',
            linewidth=1.2,
            label='target',
        )
        self._ax.legend()

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def stop(self):
        try:
            import matplotlib.pyplot as plt

            if self._fig is not None:
                plt.close(self._fig)
        except Exception:
            pass

        self._fig = None
        self._ax = None
        self._traj_line = None
        self._path_line = None
        self._drone_scatter = None
        self._est_path_line = None
        self._est_scatter = None
        self._target_scatter = None
        self._target_vel_line = None

    def refresh(self):
        if self._fig is None or self._ax is None or self._drone_scatter is None:
            return

        with self._lock:
            px, py, pz = self._pos.tolist()
            path = np.array(self._path) if len(self._path) else None
            est = self._est_pos.copy() if self._est_pos is not None else None
            est_path = np.array(self._est_path) if len(self._est_path) else None
            target = self._target_pos.copy() if self._target_pos is not None else None
            target_vel = self._target_vel.copy() if self._target_vel is not None else None

        try:
            self._drone_scatter._offsets3d = ([px], [py], [pz])
            if self._path_line is not None and path is not None and path.shape[0] > 1:
                self._path_line.set_data(path[:, 0], path[:, 1])
                self._path_line.set_3d_properties(path[:, 2])
            if self._est_scatter is not None and est is not None:
                self._est_scatter._offsets3d = ([est[0]], [est[1]], [est[2]])
            if self._est_path_line is not None and est_path is not None and est_path.shape[0] > 1:
                self._est_path_line.set_data(est_path[:, 0], est_path[:, 1])
                self._est_path_line.set_3d_properties(est_path[:, 2])
            if self._target_scatter is not None and target is not None:
                self._target_scatter._offsets3d = ([target[0]], [target[1]], [target[2]])
            if self._target_vel_line is not None and target is not None:
                vx, vy, vz = (target_vel.tolist() if target_vel is not None else (0.0, 0.0, 0.0))
                self._target_vel_line.set_data(
                    [target[0], target[0] + vx * self._vel_scale],
                    [target[1], target[1] + vy * self._vel_scale],
                )
                self._target_vel_line.set_3d_properties([target[2], target[2] + vz * self._vel_scale])
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            pass
