import numpy as np
from scipy import linalg

class Trajectory:
    def __init__(self):
        self.segments = []
        self.segment_times = []
        self.coeffs = []
        self.slowdown = 0.5
        self.v_xy_max = 2.25 * self.slowdown
        self.v_z_max = 0.75 * self.slowdown

    def estimate_segment_time(self, p0, p1):
        """
        Estimates flight time by considering the drone's velocity limits 
        on different axes as concurrent rather than additive.
        """
        dp = np.array(p1) - np.array(p0)
        dist_xy = np.linalg.norm(dp[:2])
        dist_z = abs(dp[2])
        # 2. Calculate time required for each component
        t_xy = dist_xy / self.v_xy_max
        t_z = dist_z / self.v_z_max
        
        # 3. The "Diagonal" estimation
        # Since the drone moves in 3D, the time is the maximum of the two requirements,
        # plus a small overhead for the combined vector (acceleration/drag).
        t_base = max(t_xy, t_z)

        # 4. Add an acceleration/cornering penalty
        # It takes time to change direction or start from a standstill.
        # We add a constant 't_turn' or scale by the distance.
        t_accel = 0.5  # Seconds lost to inertia
        
        T = t_base + t_accel
        
        return max(T, 1.0)
    

    def solve_trajectory(self, waypoints):
        num_waypoints = len(waypoints)
        num_segments = num_waypoints - 1
        
        # Pre-calculate segment times
        # Assuming self.estimate_segment_time exists; velocities could be passed as zeros or averages
        self.segment_times = []
        for i in range(num_segments):
            T = self.estimate_segment_time(waypoints[i], waypoints[i+1])
            self.segment_times.append(T)

        # Solve for each axis (X, Y, Z) independently
        all_coeffs = [] # Stores [segment][axis]
        for axis in range(3):
            # 6 coefficients per segment (a5*t^5 + ... + a0)
            n_vars = 6 * num_segments
            A = np.zeros((n_vars, n_vars))
            b = np.zeros(n_vars)
            row = 0

            # 1. Start Waypoint: Position, Velocity=0, Acceleration=0
            # p(0) = p0
            A[row, 0:6] = [0, 0, 0, 0, 0, 1]; b[row] = waypoints[0][axis]; row += 1
            # v(0) = 0
            A[row, 0:6] = [0, 0, 0, 0, 1, 0]; b[row] = 0; row += 1
            # a(0) = 0
            A[row, 0:6] = [0, 0, 0, 2, 0, 0]; b[row] = 0; row += 1

            # 2. Intermediate Waypoints and Continuity
            for i in range(num_segments - 1):
                T = self.segment_times[i]
                off = i * 6
                next_off = (i + 1) * 6
                
                # Position at end of segment i must be waypoint i+1
                A[row, off:off+6] = [T**5, T**4, T**3, T**2, T, 1]
                b[row] = waypoints[i+1][axis]; row += 1
                
                # Position at start of segment i+1 must be waypoint i+1
                A[row, next_off:next_off+6] = [0, 0, 0, 0, 0, 1]
                b[row] = waypoints[i+1][axis]; row += 1

                # Continuity: Vel, Acc, Jerk, Snap (1st, 2nd, 3rd, 4th derivatives)
                # v_i(T) - v_{i+1}(0) = 0
                A[row, off:off+6] = [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0]
                A[row, next_off+4] = -1; row += 1
                
                # a_i(T) - a_{i+1}(0) = 0
                A[row, off:off+6] = [20*T**3, 12*T**2, 6*T, 2, 0, 0]
                A[row, next_off+3] = -2; row += 1
                
                # j_i(T) - j_{i+1}(0) = 0 (3rd)
                A[row, off:off+6] = [60*T**2, 24*T, 6, 0, 0, 0]
                A[row, next_off+2] = -6; row += 1
                
                # s_i(T) - s_{i+1}(0) = 0 (4th)
                A[row, off:off+6] = [120*T, 24, 0, 0, 0, 0]
                A[row, next_off+1] = -24; row += 1

            # 3. End Waypoint: Position, Velocity=0, Acceleration=0
            T_last = self.segment_times[-1]
            last_off = (num_segments - 1) * 6
            # p(T) = p_last
            A[row, last_off:last_off+6] = [T_last**5, T_last**4, T_last**3, T_last**2, T_last, 1]
            b[row] = waypoints[-1][axis]; row += 1
            # v(T) = 0
            A[row, last_off:last_off+6] = [5*T_last**4, 4*T_last**3, 3*T_last**2, 2*T_last, 1, 0]
            b[row] = 0; row += 1
            # a(T) = 0
            A[row, last_off:last_off+6] = [20*T_last**3, 12*T_last**2, 6*T_last, 2, 0, 0]
            b[row] = 0; row += 1

            # Solve the system
            x = linalg.solve(A, b)
            all_coeffs.append(x.reshape(num_segments, 6))

        # Reorganize into segment objects
        for s in range(num_segments):
            seg_coeffs = [all_coeffs[0][s], all_coeffs[1][s], all_coeffs[2][s]]
            vel_xyz = [np.polyder(np.poly1d(c)) for c in seg_coeffs]
            
            self.segments.append({
                'pos': [np.poly1d(c) for c in seg_coeffs],
                'vel': vel_xyz
            })
            self.coeffs.append(seg_coeffs)

    def get_velocity(self, time):
        elapsed = 0
        for i, T in enumerate(self.segment_times):
            if time <= elapsed + T:
                t = time - elapsed
                # Evaluate the pre-calculated velocity polynomials
                v_x = self.segments[i]['vel'][0](t)
                v_y = self.segments[i]['vel'][1](t)
                v_z = self.segments[i]['vel'][2](t)
                return np.array([v_x, v_y, v_z])
            elapsed += T
        
        # If time exceeds trajectory, return zero velocity
        return np.array([0.0, 0.0, 0.0])

    def get_position(self, time):
        # Same logic as get_velocity but using ['pos']
        elapsed = 0
        for i, T in enumerate(self.segment_times):
            if time <= elapsed + T:
                t = time - elapsed
                p_x = self.segments[i]['pos'][0](t)
                p_y = self.segments[i]['pos'][1](t)
                p_z = self.segments[i]['pos'][2](t)
                return np.array([p_x, p_y, p_z])
            elapsed += T
        return None


    def generate_trajectory(self, gates):
        self.solve_trajectory([g[:3] for g in gates])