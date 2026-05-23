#!/usr/bin/env python3
#!/usr/bin/env python3
"""Command-line Crazyflie launcher that plans a trajectory through gates and follows it.

This version does not handle any image packets — it only logs position from the
Crazyflie, plans the trajectory using `trajectory.Trajectory`, and follows it with
`follower.Follower`. It waits for an Enter keypress in the console before starting
to send setpoints so you can validate the plan.
"""
import logging
import sys
import time
import threading
from threading import Timer
import select

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from pynput import keyboard

from follower import Follower
from trajectory import Trajectory
from visualisation import Visualisation
from flight_logger import FlightLogger

logging.basicConfig(level=logging.ERROR)

URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E705')


HOME_POINT = [-0.0, 0.0, 1.5]
LAND_POINT = [-0.0, 0.0, 0.0]

GATE_PREPOST_DISTANCE = 0.3

GATES = [
    HOME_POINT,  
  [0.68, -0.79, 1.28, -19],
  [1.74, -0.89, 1.15, -1],
  [2.22, 0.08, 1.43, 102],
  [1.64, 0.82, 1.18, 165],
  [0.64, 0.93, 1.26, -165],
    HOME_POINT,
  [0.68, -0.79, 1.28, -19],
  [1.74, -0.89, 1.15, -1],
  [2.22, 0.08, 1.43, 102],
  [1.64, 0.82, 1.18, 165],
  [0.64, 0.93, 1.26, -165],
    HOME_POINT,
]


def _build_waypoints(gates, home_point, prepost_distance):
    waypoints = []
    for gate in gates:
        pos = np.array(gate[:3], dtype=float)
        angle_deg = gate[3] if len(gate) > 3 else None
        is_home = np.allclose(pos, home_point)
        if angle_deg is None or is_home or prepost_distance <= 0.0:
            waypoints.append(pos.tolist())
            continue

        angle = np.deg2rad(float(angle_deg))
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        pre = pos - direction * prepost_distance
        post = pos + direction * prepost_distance
        waypoints.extend([pre.tolist(), pos.tolist(), post.tolist()])

    return waypoints



class Launcher:
    def __init__(self, uri):
        self.uri = uri
        self.cf = Crazyflie(rw_cache='./cache')
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

        self.follower = Follower()
        self.waypoints = _build_waypoints(GATES, HOME_POINT, GATE_PREPOST_DISTANCE)
        self.follower.trajectory.generate_trajectory(self.waypoints)
        self.follower.reset_trajectory()

        self.logger = FlightLogger(metadata={
            "uri": self.uri,
            "home": HOME_POINT,
            "land": LAND_POINT,
        })
        self.logger.set_planned_points(self.waypoints)

        # Visualization: show planned trajectory and live drone pose
        try:
            self.vis = Visualisation()
            self.vis.set_trajectory(self.follower.trajectory)
            self.vis.set_waypoints(self.waypoints)
            self.vis.start()
        except Exception as e:
            print('Visualization unavailable:', e)
            self.vis = None

        self.sensor_data = {
            't': 0.0,
            'x_global': 0.0,
            'y_global': 0.0,
            'z_global': 0.0,
            'yaw': 0.0,
        }

        self.is_connected = False
        self._log_ready = False
        self._emergency_stop = False
        self._log_saved = False

    def start(self):
        print('Initializing drivers')
        cflib.crtp.init_drivers()
        print('Opening link to %s' % self.uri)
        self.cf.open_link(self.uri)

        # Wait for connection (timeout)
        wait_t = 0.0
        while not self.is_connected and wait_t < 10.0:
            time.sleep(0.1)
            wait_t += 0.1

        if not self.is_connected:
            print('Could not connect to Crazyflie')
            return

        # Reset Kalman filter as a warmup
        try:
            self.cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            self.cf.param.set_value('kalman.resetEstimation', '0')
            time.sleep(1.0)
        except Exception:
            pass

        # Start emergency stop listener
        threading.Thread(target=self._start_emergency_listener, daemon=True).start()
        threading.Thread(target=self._start_terminal_emergency_listener, daemon=True).start()

        # Wait for operator confirmation
        try:
            input('Press Enter in the console to START following the planned trajectory...')
        except Exception:
            print('Console input unavailable; aborting start')
            return

        # Wait until log stream is ready
        while not self._log_ready:
            print('Waiting for log stream...')
            time.sleep(0.1)

        print('Going to home point before trajectory')
        self._goto_position(HOME_POINT, duration=4.0, rate_hz=20)

        print('Starting trajectory following')
        self.follower.reset_trajectory()

        # Main follow loop
        try:
            while self.is_connected:
                if self._emergency_stop:
                    print('Emergency stop requested; exiting control loop')
                    break
                # Only run follower if log is ready
                if self._log_ready:
                    cmd, loop_wrapped = self.follower.follow_planned_trajectory(self.sensor_data)
                    target_pos = self.follower.trajectory.get_position(
                        self.follower.virtual_time + self.follower.look_ahead
                    )
                    target_vel = self.follower.trajectory.get_velocity(
                        self.follower.virtual_time + self.follower.look_ahead
                    )
                    self.logger.log_sample(
                        self.sensor_data,
                        cmd=cmd,
                        target_pos=target_pos,
                        target_vel=target_vel,
                        note="follow",
                    )
                    # send_position_setpoint(x, y, z, yaw)
                    self.cf.commander.send_position_setpoint(cmd[0], cmd[1], cmd[2], cmd[3])
                    if loop_wrapped:
                        print('Trajectory finished')
                        break
                if hasattr(self, 'vis') and self.vis:
                    self.vis.refresh()
                time.sleep(0.01)

            if self.is_connected:
                print('Returning to home point')
                self._goto_position(HOME_POINT, duration=3.0, rate_hz=20)
                print('Landing')
                self._goto_position(LAND_POINT, duration=3.0, rate_hz=20)
        finally:
            try:
                self.cf.commander.send_stop_setpoint()
            except Exception:
                print("Failed to send stop setpoint")
            try:
                log_path = self.logger.save_json()
                print(f"Flight log saved to {log_path}")
                self._log_saved = True
            except Exception as e:
                print('Failed to save flight log:', e)
            try:
                if hasattr(self, 'vis') and self.vis:
                    self.vis.refresh()
                    if not self._emergency_stop:
                        self.vis.stop()
            except Exception:
                pass
            self.cf.close_link()

    def _goto_position(self, target, duration=3.0, rate_hz=20):
        if not self._log_ready:
            return

        start = np.array([
            self.sensor_data.get('x_global', target[0]),
            self.sensor_data.get('y_global', target[1]),
            self.sensor_data.get('z_global', target[2]),
        ], dtype=float)
        end = np.array(target, dtype=float)
        steps = max(int(duration * rate_hz), 1)

        for i in range(steps + 1):
            if not self.is_connected:
                break
            t = i / steps
            pos = (1.0 - t) * start + t * end
            yaw = float(self.sensor_data.get('yaw', 0.0))
            self.logger.log_sample(
                self.sensor_data,
                cmd=[float(pos[0]), float(pos[1]), float(pos[2]), yaw],
                target_pos=end,
                target_vel=[0.0, 0.0, 0.0],
                note="goto",
            )
            self.cf.commander.send_position_setpoint(float(pos[0]), float(pos[1]), float(pos[2]), yaw)
            if hasattr(self, 'vis') and self.vis:
                self.vis.refresh()
            time.sleep(1.0 / rate_hz)

    def _start_emergency_listener(self):
        # Uses pynput to listen for 'q' to emergency stop
        def on_press(key):
            try:
                if key.char == 'q':
                    self._trigger_emergency_stop('pynput')
                    return False
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    def _start_terminal_emergency_listener(self):
        # Listen for 'q' from the terminal (STDIN) for environments where pynput fails
        if not sys.stdin or not sys.stdin.isatty():
            return

        if sys.platform.startswith('win'):
            try:
                import msvcrt
            except Exception:
                return
            while self.is_connected and not self._emergency_stop:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if isinstance(ch, str) and ch.lower() == 'q':
                        self._trigger_emergency_stop('terminal')
                        break
                time.sleep(0.05)
        else:
            while self.is_connected and not self._emergency_stop:
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                except Exception:
                    break
                if ready:
                    ch = sys.stdin.read(1)
                    if isinstance(ch, str) and ch.lower() == 'q':
                        self._trigger_emergency_stop('terminal')
                        break

    def _trigger_emergency_stop(self, source):
        print(f'Emergency stop (q) received from {source}')
        self._emergency_stop = True
        if not self._log_saved:
            try:
                log_path = self.logger.save_json()
                print(f"Flight log saved to {log_path}")
                self._log_saved = True
            except Exception as e:
                print('Failed to save flight log:', e)
        try:
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass

    def _connected(self, link_uri):
        print('Connected to %s' % link_uri)
        self.is_connected = True

        # Setup logging
        self.log_conf = LogConfig(name='Position', period_in_ms=100)
        self.log_conf.add_variable('stateEstimate.x', 'float')
        self.log_conf.add_variable('stateEstimate.y', 'float')
        self.log_conf.add_variable('stateEstimate.z', 'float')
        self.log_conf.add_variable('stabilizer.yaw', 'float')

        try:
            self.cf.log.add_config(self.log_conf)
            self.log_conf.data_received_cb.add_callback(self._log_cb)
            self.log_conf.error_cb.add_callback(self._log_error)
            self.log_conf.start()
            self._log_ready = True
        except Exception as e:
            print('Failed to start log config:', e)

        # Optional auto-disconnect timer (comment out if undesired)
        # t = Timer(50, self.cf.close_link)
        # t.start()

    def _log_cb(self, timestamp, data, logconf):
        x = data.get('stateEstimate.x', 0.0)
        y = data.get('stateEstimate.y', 0.0)
        z = data.get('stateEstimate.z', 0.0)
        yaw = data.get('stabilizer.yaw', 0.0)
        self.sensor_data = {
            't': float(timestamp) / 1000.0,
            'x_global': float(x),
            'y_global': float(y),
            'z_global': float(z),
            'yaw': float(yaw),
        }
        # update visualization if available
        try:
            if hasattr(self, 'vis') and self.vis:
                self.vis.update_sensor(self.sensor_data)
        except Exception:
            pass
        print(f"[{timestamp}] pos x={x:.2f} y={y:.2f} z={z:.2f}")

    def _log_error(self, logconf, msg):
        print('Log error:', logconf.name, msg)

    def _connection_failed(self, link_uri, msg):
        print('Connection to %s failed: %s' % (link_uri, msg))

    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False
        try:
            if hasattr(self, 'vis') and self.vis:
                if not self._emergency_stop:
                    self.vis.stop()
        except Exception:
            pass


def emergency_stop_callback(cf):
    # kept for compatibility with older scripts, but we use pynput in Launcher
    def on_press(key):
        try:
            if key.char == 'q':
                print('Emergency stop triggered!')
                cf.commander.send_stop_setpoint()
                cf.close_link()
                return False
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    launcher = Launcher(URI)
    launcher.start()
