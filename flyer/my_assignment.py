import numpy as np

from assignment.flyer2 import Flyer2
from assignment.surveyer import Surveyer
from assignment.visualizer import Visualizer


class FlyerAdapter:
    def __init__(self):
        self._under = Flyer2()
        self.trajectory_ready = False
        self.trajectory_points = None
        self.closest_point_world = None
        self.lookahead_point_world = None
        self.lookahead_tangent_world = None
        self.trajectory_loop_count = 0
        self.trajectory_finished = True
        self.start_time = None
        self.look_ahead = 0.05
        self.tau = 0.5
        self.tau_z = 0.20
        # Clock synchronization parameters
        self.last_real_time = None
        self.virtual_time = 0.0
        self.min_lag = 0.2  # Distance threshold for full speed
        self.max_lag = 1  # Distance threshold for minimum speed
        self.min_speed = 0.5  # Minimum time scale factor

    def plan_trajectory(self, home, gates):
        gate_angles = [float(getattr(self, 'detected_angles', [])[i]) + np.pi / 2 for i in range(len(gates))]
        gates_with_angles = [[home[0], home[1], gates[0][2], gate_angles[0]]]
        for i, g in enumerate(gates):
            try:
                angle = gate_angles[i]
            except Exception:
                angle = 0.0
            gates_with_angles.append([g[0], g[1], g[2], angle])

        gates_with_angles.append([home[0], home[1], gates[-1][2], gate_angles[-1]])
        self._under.generate_trajectory(gates_with_angles)

        pts = []
        samples_per_seg = 20
        total_time = np.sum(self._under.segment_times)
        ts = np.linspace(0, total_time, num=samples_per_seg*len(self._under.segment_times))
        for t in ts:
            pts.append(self._under.get_setpoint(t))

        self.trajectory_points = pts
        self.trajectory_ready = True
        self.trajectory_finished = False
        self.start_time = None
        self.virtual_time = 0.0
        self.last_real_time = None

    def restart_trajectory(self, t):
        try:
            self._under.restart_trajectory(t)
        except Exception:
            pass
        self.start_time = t
        self.trajectory_finished = False
        self.virtual_time = 0.0
        self.last_real_time = None

    def update_clock(self, current_real_time, current_pos):
        # Initialize on first call
        if self.last_real_time is None:
            self.last_real_time = current_real_time
            return self.virtual_time
        
        # 1. Calculate how much real time has passed
        dt = current_real_time - self.last_real_time
        self.last_real_time = current_real_time
        
        # 2. Find where we SHOULD be according to our virtual clock
        target_p = self._under.get_setpoint(self.virtual_time)
        
        if target_p is None:
            return self.virtual_time # Trajectory finished

        # 3. Calculate the distance error
        dist_error = np.linalg.norm(target_p - current_pos)
        
        # 4. Calculate Time Scaling Factor (K_time)
        # If error is small, factor is 1.0. If error is large, factor approaches min_speed.
        # We use a linear interpolation (lerp) or a smooth step
        if dist_error < self.min_lag:
            time_scale = 1.0
        else:
            # Scale reduces as error increases
            time_scale = 1.0 - (dist_error - self.min_lag) / (self.max_lag - self.min_lag)
            time_scale = max(time_scale, self.min_speed)
        
        # 5. Advance the virtual clock
        self.virtual_time += dt * time_scale
        
        return self.virtual_time

    def follow_planned_trajectory(self, sensor_data, home):
        if self.start_time is None:
            self.start_time = float(sensor_data.get('t', 0.0)) - 0.001
            self.virtual_time = 0.0
            self.last_real_time = None

        current_real_time = float(sensor_data.get('t', 0.0))
        current_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Update virtual clock based on tracking error
        self.update_clock(current_real_time, current_pos)
        
        # Get trajectory setpoint at virtual time
        target_pos = self._under.get_setpoint(self.virtual_time + self.look_ahead)
        target_vel = self._under.get_velocity(self.virtual_time + self.look_ahead)

        if target_pos is None:
            self.trajectory_finished = True
            loop_wrapped = True
            cmd = [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data.get('yaw', 0.0)]
            return cmd, loop_wrapped

        desired_yaw = np.arctan2(target_vel[1], target_vel[0])
        shifted_p = target_pos + np.array([
            target_vel[0] * self.tau,
            target_vel[1] * self.tau,
            target_vel[2] * self.tau_z,
        ])
        cmd = [float(shifted_p[0]), float(shifted_p[1]), float(shifted_p[2]), desired_yaw]
        loop_wrapped = False
        return cmd, loop_wrapped


class MyAssignment:
    def __init__(self):
        self.state = 0
        self.home = None
        self.lap = 0
        self.surveyer = Surveyer()
        self.flyer = FlyerAdapter()
        self.visualizer = Visualizer()
        self.wait_home = False
        self.home_settle_start_time = None
        self.home_wait_start_time = None
        self.home_wait_altitude = 0.9
        self.home_wait_duration = 1.0
        self.home_wait_max_duration = 6.0

    def _racing_start_command(self, sensor_data):
        start_z = self.home_wait_altitude
        if self.surveyer.gates:
            start_z = float(self.surveyer.gates[0][2])

        start_yaw = float(sensor_data.get('yaw', 0.0))
        if self.surveyer.detected_gate_angles:
            start_yaw = float(self.surveyer.detected_gate_angles[0]) + np.pi / 2

        return [self.home[0], self.home[1], start_z, start_yaw]

    @staticmethod
    def _yaw_error(current_yaw, target_yaw):
        return abs(np.arctan2(np.sin(current_yaw - target_yaw), np.cos(current_yaw - target_yaw)))

    def _release_home_wait(self):
        self.wait_home = False
        self.home_settle_start_time = None
        self.home_wait_start_time = None
        self.flyer.trajectory_finished = False
        self.flyer.trajectory_loop_count = 0
        self.flyer.closest_point_world = None
        self.flyer.lookahead_point_world = None
        self.flyer.lookahead_tangent_world = None
        self.flyer.trajectory_points = None
        self.flyer.trajectory_ready = False
        self.start_time = None

    def set_ground_truth_gates(self, gate_positions, gate_orientations):
        self.ground_truth_gates = (gate_positions, gate_orientations)
        self.visualizer.set_ground_truth_gates(gate_positions, gate_orientations)

    def update_visualization(self, sensor_data, force=False):
        self.visualizer.update_visualization(
            sensor_data,
            self.surveyer,
            self.flyer,
            self.lap,
            force=force,
        )

    def should_show_visualization(self):
        return False
        return self.lap >= 1 or self.surveyer.gate_to_go >= 5

    def close_visualization(self):
        self.visualizer.close()

    def compute_command(self, sensor_data, camera_data, dt):
        if sensor_data['z_global'] < 0.49:
            return [sensor_data['x_global'], sensor_data['y_global'], 1, sensor_data['yaw']]

        if self.home is None:
            self.home = [sensor_data['x_global'], sensor_data['y_global'], self.home_wait_altitude]

        if self.wait_home:
            control_command = self._racing_start_command(sensor_data)
            if self.home_wait_start_time is None:
                self.home_wait_start_time = sensor_data['t']

            at_home = self.surveyer.reached_target(sensor_data, control_command, threshold=0.3)

            if at_home:
                if self.home_settle_start_time is None:
                    self.home_settle_start_time = sensor_data['t']
                elif (sensor_data['t'] - self.home_settle_start_time) >= self.home_wait_duration:
                    self._release_home_wait()
            else:
                self.home_settle_start_time = None

            if self.wait_home and self.home_wait_start_time is not None:
                if (sensor_data['t'] - self.home_wait_start_time) >= self.home_wait_max_duration:
                    self._release_home_wait()

            return control_command

        if self.lap == 0:
            if self.surveyer.gate_to_go < 5:
                return self.surveyer.map_gate(sensor_data, camera_data)

            control_command = self._racing_start_command(sensor_data)
            if self.surveyer.reached_target(sensor_data, control_command):
                if self._yaw_error(sensor_data['yaw'], control_command[3]) > 0.2:
                    return control_command

                try:
                    self.flyer.detected_angles = list(self.surveyer.detected_gate_angles)
                except Exception:
                    self.flyer.detected_angles = []

                self.flyer.plan_trajectory(self.home, self.surveyer.gates)
                self.lap = 1
                self.surveyer.gate_to_go = 0
                self.wait_home = True
                self.home_wait_start_time = None

            return control_command

        if self.lap == 1 or self.lap == 2:
            if not self.flyer.trajectory_ready:
                try:
                    self.flyer.detected_angles = list(self.surveyer.detected_gate_angles)
                except Exception:
                    self.flyer.detected_angles = []
                self.flyer.plan_trajectory(self.home, self.surveyer.gates)

            control_command, loop_wrapped = self.flyer.follow_planned_trajectory(sensor_data, self.home)
            if loop_wrapped:
                self.lap = 1 + self.flyer.trajectory_loop_count
                self.wait_home = True
                self.home_wait_start_time = None
                return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]

            return control_command

        return [sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global'], sensor_data['yaw']]


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)


def set_ground_truth_gates(gate_positions, gate_orientations):
    _controller.set_ground_truth_gates(gate_positions, gate_orientations)


def update_visualization(sensor_data, force=False):
    _controller.update_visualization(sensor_data, force=force)


def should_show_visualization():
    return _controller.should_show_visualization()


def close_visualization():
    _controller.close_visualization()