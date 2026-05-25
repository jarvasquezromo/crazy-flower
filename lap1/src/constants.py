"""All tuning constants for the lap1 gate-flying pipeline."""
import numpy as np

# --- Network / protocol ---
AIDECK_IP = '192.168.4.1'
AIDECK_PORT = 5000
LOCAL_PORT = 5001
START_MAGIC = b'FER'
SPEED = 0.6

CPX_HEADER_SIZE = 4
IMG_HEADER_MAGIC = 0xBC
IMG_HEADER_SIZE = 11
MIN_JPEG_BYTES = 5000

# --- Gate detection / HSV ---
GREEN_HSV_LO = np.array([25, 0, 0], dtype=np.uint8)
GREEN_HSV_HI = np.array([118, 73, 255], dtype=np.uint8)
GREEN_MIN_V = 181

MIN_GREEN_AREA_FRAC = 0.01
CENTER_TOL_X = 0.10
CENTER_TOL_Y = 0.12

# --- Gate-shape acceptance ---
GATE_MIN_VERTICES = 4
GATE_MAX_VERTICES = 8
GATE_ASPECT_MIN = 0.45
GATE_ASPECT_MAX = 2.2
GATE_MIN_SOLIDITY = 0.80
GATE_APPROX_EPS = 0.04
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# --- Flight control ---
SEARCH_YAWRATE = -20.0
MAX_YAWRATE = 70.0
K_YAW = 80.0
FORWARD_SPEED = 0.35
PUSH_DURATION_S = 1.0
SEARCH_HEIGHT = 0.8
TAKEOFF_START_HEIGHT = 0.1
TAKEOFF_RATE = 0.4
MAX_GATES = 5
MIN_HEIGHT = 0.2
MAX_HEIGHT = 2.0
K_HEIGHT = 1.2
MAX_DH_PER_S = 0.6

# --- Gate projection method ---
# True  = mono-height method (from GatesDetector: uses known gate height + pinhole)
# False = PnP/bbox method (solvePnP with 4 corners, fallback to bbox-width range)
USE_MONO_HEIGHT_PROJECTION = False
GATE_PHYS_H = 0.4  # metres, physical gate height (for mono-height method)

# --- Camera / world-frame projection ---
GATE_PHYS_W = 0.4  # metres, physical gate width (spec: 0.3–0.5m)
TRAJ_N_STEPS = 5
TRAJ_OVERSHOOT = 0.30
WAYPOINT_TOL = 0.15
PASS_AREA_FRAC = 0.15
CHASE_TIMEOUT = 8.0
GATE_EMA_ALPHA = 0.35

# --- Gate locking / outlier rejection ---
GATE_LOCK_MIN_OBS = 3
GATE_LOCK_MAX_OBS = 5
GATE_LOCK_MAX_AGE_S = 1.0
GATE_LOCK_MAX_SPREAD_XY = 0.30
CHASE_UPDATE_MAX_JUMP_XY = 0.60
PNP_RANGE_MIN = 0.25
PNP_RANGE_MAX = 5.00
PNP_LATERAL_MAX = 3.00

# --- Crazyflie URI ---
URI_DEFAULT = 'radio://0/70/2M/E7E7E7E705'
