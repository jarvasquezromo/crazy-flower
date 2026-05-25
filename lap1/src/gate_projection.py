"""Gate position estimation via multi-view triangulation.

Adapted from GatesDetectorTriangulation: accumulates observations of gate
corners from different drone positions, then triangulates the gate centre
using the two-line midpoint method (ray intersection).

When USE_MONO_HEIGHT_PROJECTION is True, the state machine uses this module
instead of the single-frame PnP/bbox method.
"""
import numpy as np

from .constants import GATE_PHYS_H, MIN_HEIGHT, MAX_HEIGHT


class GateTriangulator:
    """Accumulates gate corner observations and triangulates position."""

    # Minimum drone displacement between stored observations (metres)
    MIN_BASELINE = 0.10

    def __init__(self, calib):
        """
        Args:
            calib: dict with fx, fy, cx, cy, img_w, img_h.
        """
        fx = float(calib['fx'])
        fy = float(calib['fy'])
        cx = float(calib['cx'])
        cy = float(calib['cy'])
        self.k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.inv_k = np.linalg.inv(self.k)
        self._observations = []  # list of (corners, yaw_rad, pos_xyz)

    def reset(self):
        self._observations.clear()

    def add_observation(self, corners, est):
        """Store an observation if the drone moved enough since the last one.

        Args:
            corners: (4,2) ordered [TL, TR, BR, BL] image points.
            est: dict with x, y, z, yaw.
        """
        if corners is None:
            return
        pos = np.array([est['x'], est['y'], est['z']])
        if self._observations:
            last_pos = self._observations[-1][2]
            if np.linalg.norm(pos - last_pos) < self.MIN_BASELINE:
                return
        self._observations.append((
            np.asarray(corners, dtype=np.float64).reshape(4, 2),
            np.deg2rad(est['yaw']),
            pos.copy(),
        ))
        # Keep bounded
        if len(self._observations) > 10:
            self._observations = self._observations[-10:]

    def triangulate(self, est):
        """Triangulate gate centre from accumulated observations.

        Returns:
            (gx, gy, gz) or None if not enough data.
        """
        if len(self._observations) < 2:
            return self._mono_fallback(est)

        # Triangulate left and right posts across the two most recent observations
        obs1 = self._observations[-2]
        obs2 = self._observations[-1]
        left_3d = self._triangulate_post(obs1, obs2, 0, 3)   # TL, BL
        right_3d = self._triangulate_post(obs1, obs2, 1, 2)  # TR, BR

        if left_3d is None or right_3d is None:
            return self._mono_fallback(est)

        center = (left_3d + right_3d) / 2.0
        gz = float(np.clip(center[2], MIN_HEIGHT, MAX_HEIGHT))
        return float(center[0]), float(center[1]), gz

    def _triangulate_post(self, obs1, obs2, top_idx, bot_idx):
        """Triangulate a single post midpoint from two observations."""
        corners1, yaw1, pos1 = obs1
        corners2, yaw2, pos2 = obs2

        # Midpoint pixel of the post in each frame
        mid1 = (corners1[top_idx] + corners1[bot_idx]) / 2.0
        mid2 = (corners2[top_idx] + corners2[bot_idx]) / 2.0

        # Camera positions (camera = drone position, forward-facing)
        P = pos1
        Q = pos2

        # Bearing rays in world frame
        r = self._pixel_to_world_ray(mid1, yaw1)
        s = self._pixel_to_world_ray(mid2, yaw2)

        # Solve for closest points on the two rays (least-squares)
        A = np.column_stack([r, -s])  # 3x2
        b = Q - P
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        lam, mu = result

        F = P + lam * r
        G = Q + mu * s
        residual = float(np.linalg.norm(F - G))

        # Reject if rays are too far apart (bad triangulation)
        if residual > 0.5:
            return None

        return (F + G) / 2.0

    def _pixel_to_world_ray(self, pixel, yaw_rad):
        """Convert pixel to unit bearing vector in world frame.

        Camera convention: z-forward, x-right, y-down.
        Body: x-forward, y-left, z-up.
        World: rotated by yaw.
        """
        px_h = np.array([pixel[0], pixel[1], 1.0])
        v_cam = self.inv_k @ px_h  # direction in camera frame

        # Camera → body: x_body = z_cam, y_body = -x_cam, z_body = -y_cam
        v_body = np.array([v_cam[2], -v_cam[0], -v_cam[1]])

        # Body → world (yaw rotation)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        v_world = np.array([
            v_body[0] * cos_y - v_body[1] * sin_y,
            v_body[0] * sin_y + v_body[1] * cos_y,
            v_body[2],
        ])
        return v_world / np.linalg.norm(v_world)

    def _mono_fallback(self, est):
        """Single-frame depth estimate using known gate height (fallback)."""
        if not self._observations:
            return None
        corners, yaw_rad, pos = self._observations[-1]

        fy = self.k[1, 1]
        fx = self.k[0, 0]
        cx = self.k[0, 2]
        cy = self.k[1, 2]

        # Left post depth from pixel height
        left_h_px = float(np.linalg.norm(corners[0] - corners[3]))
        right_h_px = float(np.linalg.norm(corners[1] - corners[2]))
        if left_h_px < 1 or right_h_px < 1:
            return None

        left_depth = GATE_PHYS_H * fy / left_h_px
        right_depth = GATE_PHYS_H * fy / right_h_px

        left_mid = (corners[0] + corners[3]) / 2.0
        right_mid = (corners[1] + corners[2]) / 2.0

        left_cam = np.array([
            (left_mid[0] - cx) * left_depth / fx,
            (left_mid[1] - cy) * left_depth / fy,
            left_depth,
        ])
        right_cam = np.array([
            (right_mid[0] - cx) * right_depth / fx,
            (right_mid[1] - cy) * right_depth / fy,
            right_depth,
        ])

        # Camera → world
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        left_world = self._cam_to_world(left_cam, pos, cos_y, sin_y)
        right_world = self._cam_to_world(right_cam, pos, cos_y, sin_y)

        center = (left_world + right_world) / 2.0
        gz = float(np.clip(center[2], MIN_HEIGHT, MAX_HEIGHT))
        return float(center[0]), float(center[1]), gz

    @staticmethod
    def _cam_to_world(p_cam, pos, cos_y, sin_y):
        dx_b = p_cam[2]
        dy_b = -p_cam[0]
        dz_b = -p_cam[1]
        gx = pos[0] + dx_b * cos_y - dy_b * sin_y
        gy = pos[1] + dx_b * sin_y + dy_b * cos_y
        gz = pos[2] + dz_b
        return np.array([gx, gy, gz])
