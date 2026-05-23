import numpy as np
from trajectory import Trajectory

class Follower:
    def __init__(self):
        self.trajectory = Trajectory()
        self.last_real_time = None
        self.virtual_time = 0.0
        self.start_time = None
        self.trajectory_finished = False

        # constants for time scaling based on tracking error

        self.min_lag = 0.1  # Distance threshold for full speed
        self.max_lag = 1  # Distance threshold for minimum speed
        self.min_speed = 0.35  # Minimum time scale factor

        # constants for trajectory following
        
        self.look_ahead = 0.05  # how far ahead in time on the trajectory to look for the setpoint
        self.tau = 0.25  # feedforward for velocity in XY
        self.tau_z = 0.2 # feedforward for velocity in Z

        
    def update_clock(self, current_real_time, current_pos):
        # Initialize on first call
        if self.last_real_time is None:
            self.last_real_time = current_real_time
            return self.virtual_time
        
        # 1. Calculate how much real time has passed
        dt = current_real_time - self.last_real_time
        self.last_real_time = current_real_time
        
        # 2. Find where we SHOULD be according to our virtual clock
        target_p = self.trajectory.get_position(self.virtual_time)
        
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
    

    def reset_trajectory(self):
        self.virtual_time = 0.0
        self.start_time = None
        self.last_real_time = None


    def follow_planned_trajectory(self, sensor_data):
        if self.start_time is None:
            self.start_time = float(sensor_data.get('t', 0.0)) - 0.001
            self.virtual_time = 0.0
            self.last_real_time = None

        current_real_time = float(sensor_data.get('t', 0.0))
        current_pos = np.array([sensor_data['x_global'], sensor_data['y_global'], sensor_data['z_global']])
        
        # Update virtual clock based on tracking error
        self.update_clock(current_real_time, current_pos)
        
        # Get trajectory setpoint at virtual time
        target_pos = self.trajectory.get_position(self.virtual_time + self.look_ahead)
        target_vel = self.trajectory.get_velocity(self.virtual_time + self.look_ahead)

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