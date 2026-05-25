# Lap1 — Autonomous Gate Flying

Autonomous gate detection and traversal pipeline for the Crazyflie drone with AI-deck camera.

## Architecture

```
main.py                 Entry point (argparse → FPVWindow)
src/
├── controller.py       FPV GUI window, video pipeline, Crazyflie link, keyboard control
├── state_machine.py    Flight state machine: WAIT → TAKEOFF → SEARCH → CHASE → PUSH → DONE
├── detection.py        Green gate detection (HSV threshold + polygon validation)
├── gate_projection.py  Gate world-position estimation (PnP, bbox fallback, multi-view triangulation)
├── video.py            Video sources: UDP AI-deck stream, video file, recording replay
├── recorder.py         Record synchronized frames + pose for offline replay
├── debug_window.py     Debug GUI: detection stages, parameter sliders, pause/prev/next
├── gate_map.py         2D map widget showing drone + gate positions
├── calibration.py      Camera calibration loader
├── constants.py        All tuning parameters
config/
├── calibration.json    Camera intrinsics (fx, fy, cx, cy, distortion)
├── gates_xyz.py        Known gate positions (if available)
```

## Installation

```bash
pip install cflib opencv-python numpy PyQt6
```

Hardware requirements:
- Crazyflie 2.1 with Lighthouse positioning
- AI-deck (camera streams JPEG over UDP at 192.168.4.1:5000)
- CrazyRadio PA dongle

## Usage

### Live flight

Connect to the AI-deck Wi-Fi, plug in the CrazyRadio, then:

```bash
cd crazy-flower
python -m lap1.main
```

### Video simulation (no drone)

```bash
python -m lap1.main --video path/to/recording.mp4
```

### Record a session

During live flight, press **R** to start/stop recording. Saves to `lap1/recordings/rec_YYYYMMDD_HHMMSS/` with:
- `frames/` — timestamped JPEG frames
- `frames.csv` — frame timestamps
- `pose.csv` — 50Hz pose log (t, x, y, z, yaw)

### Replay a recording (with synchronized pose)

```bash
python -m lap1.main --replay lap1/recordings/rec_20260525_112800
```

Replays frames and pose at original timing so the state machine runs identically to the live flight.

## Keyboard Controls

| Key | Action |
|-----|--------|
| **R** | Start/stop recording |
| **G** | Toggle debug window (detection stages + parameter sliders) |
| **M** | Toggle mask overlay |
| **W/S** | Altitude up/down |
| **A/D** | Yaw left/right |
| **Arrows** | Manual position nudge (x/y) |
| **Space** | Emergency stop (land) |
| **Esc** | Stop motors |

## Debug Window

Press **G** to open. Shows:
- HSV image, raw mask, morphed mask (detection pipeline stages)
- Sliders to tune all detection parameters in real-time
- **Pause** — freezes the frame for parameter tuning
- **Prev/Next** — scrub through buffered frames while paused

## Detection Pipeline

1. RGB → HSV conversion
2. `inRange` threshold (H, S, V bounds) + minimum brightness filter
3. Morphological close + open (removes noise, fills gaps)
4. Contour extraction → polygon approximation
5. Shape validation: vertex count (4–8), aspect ratio (0.45–2.2), solidity (≥0.80)
6. Best candidate selection (rightmost centroid)

## Gate Position Estimation

Two methods (selected by `USE_MONO_HEIGHT_PROJECTION` in constants):

- **PnP method** (default): `solvePnP` with 4 ordered corners + known gate size (0.4m). Falls back to bbox-width ranging.
- **Multi-view triangulation**: Accumulates observations from different drone positions, triangulates via ray intersection.

Gate locking requires 3+ consistent observations within 1s with <30cm spread before committing to CHASE.

## Tuning

All parameters are in `src/constants.py`. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GREEN_HSV_LO/HI` | [35,50,50]–[85,255,255] | HSV gate color range |
| `GREEN_MIN_V` | 240 | Minimum brightness threshold |
| `GATE_PHYS_W/H` | 0.4m | Physical gate dimensions |
| `PASS_AREA_FRAC` | 0.15 | Gate area/image ratio to trigger PUSH |
| `SEARCH_YAWRATE` | -20°/s | Yaw speed during gate search |
| `FORWARD_SPEED` | 0.35 m/s | Speed through gate |
