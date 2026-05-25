# Crazy Flower Visual-Control Code Guide

This document explains the current `test/lap1_test_visual.py` controller: how the vision pipeline works, how the state machine decides what to do, what every important tuning parameter means, and how to tune the system safely.

The short version: the drone does not fly to a computed world-frame gate point. It uses the camera image directly. It rotates until it finds the expected gate, validates that gate against the course map, visually centers the gate in the image, creeps forward only when centered, then pushes straight through.

## Main Files

- `test/lap1_test_visual.py`: main autonomous flight script.
- `test/gate_map.py`: course-zone validation and top-down map widget.
- `test/gates_xyz.py`: approximate ground-truth gate positions used by the map and zone validation.
- `test/calibration.json`: AI-deck camera calibration used for undistortion and world projection.

Run with:

```bash
python3 test/lap1_test_visual.py
```

The Crazyflie URI is read from `CRAZYFLIE_URI`; otherwise it defaults to:

```text
radio://0/70/2M/E7E7E7E705
```

## Coordinate And Image Conventions

Body-frame commands sent through `send_hover_setpoint` use:

- `x`: forward velocity in m/s.
- `y`: left velocity in m/s.
- `yaw`: yaw rate in deg/s.
- `height`: absolute height setpoint in meters.

Image errors are normalized around the calibrated optical center:

```python
ex = (cx - CAMERA_CX) / (0.5 * img_w)
ey = (cy - CAMERA_CY) / (0.5 * img_h)
```

- `ex > 0`: gate centroid appears to the right of the camera center.
- `ex < 0`: gate centroid appears to the left.
- `ey > 0`: gate centroid appears below the camera center.
- `ey < 0`: gate centroid appears above the camera center.

Important consequence: if the gate appears above the image center, the drone is usually too low and must climb. The code computes:

```python
ey_err = ey_t - APPROACH_TARGET_EY
dh = -K_HEIGHT * height_scale * ey_err
```

So if `ey_t` is below the target value, `ey_err` is negative and `dh` becomes positive, increasing the height setpoint.

## High-Level Algorithm

The program is a Qt GUI with two concurrent data streams:

- Video frames arrive from the AI-deck over UDP in `UdpVideoThread`.
- Crazyflie pose and quaternion logs arrive at 50 Hz through cflib log configs.

Every video frame is processed by `_update_image()`:

1. Convert grayscale/BGR image into RGB if needed.
2. Undistort it using `calibration.json`.
3. Detect bright gate-like blobs with `_detect_green_gate()`.
4. Store the latest detection in `self._vision`.
5. Draw the FPV overlay, candidate rectangles, crosshair, state, and calibration numbers.

Every 20 ms, `_send_setpoint()` runs the control loop:

1. Read the latest vision result.
2. Read the latest state estimate.
3. Update the map widget.
4. Run the autonomy state machine.
5. Send the final body-frame hover setpoint.

## State Machine

The main states are:

```text
WAIT -> TAKEOFF -> SEARCH -> CHASE -> PUSH -> SEARCH -> ... -> DONE
                         \-> CHASE_RECOVER -> CHASE or SEARCH
STOP is a manual safety state.
```

### WAIT

This is only an initial transition state. The code immediately switches to `TAKEOFF`.

### TAKEOFF

The drone climbs to `TAKEOFF_HEIGHT` and rotates to `TAKEOFF_YAW_DEG`.

Relevant code:

```python
self.hover['height'] = min(self.hover['height'] + TAKEOFF_RATE * dt, TAKEOFF_HEIGHT)
yaw_err = ((TAKEOFF_YAW_DEG - est['yaw'] + 180.0) % 360.0) - 180.0
yaw_cmd = clip(K_TAKEOFF_YAW * yaw_err, -MAX_YAWRATE, MAX_YAWRATE)
```

The transition to `SEARCH` happens when:

```python
height >= TAKEOFF_HEIGHT
abs(yaw_err) <= TAKEOFF_YAW_TOL
```

### SEARCH

The drone yaws in place at `SEARCH_YAWRATE` until it repeatedly sees the expected gate.

Detection is not accepted just because a bright shape is visible. The code projects the detected gate corners into the world with `_gate_to_world()` and asks `GateZoneMap.validate_and_snap()` if that estimate falls inside the expected gate zone.

The script requires `SEARCH_CONFIRM_FRAMES` consecutive in-zone detections before switching to `CHASE`. This avoids locking onto one noisy false detection.

If validation passes enough times:

```python
self._gate_world = average_of_snapped_detections
self._gate_state = "CHASE"
```

### CHASE

This is the main visual-servo state.

The controller selects the target gate candidate, computes image errors, and commands:

- Yaw from horizontal error.
- Lateral velocity from horizontal error.
- Height correction from vertical error.
- Forward creep only when the gate is centered.

Core equations:

```python
yaw_cmd = clip(-K_YAW * ex_t, -MAX_YAWRATE, MAX_YAWRATE)
y_cmd = clip(-K_LATERAL * ex_t, -MAX_LATERAL, MAX_LATERAL)
ey_err = ey_t - APPROACH_TARGET_EY
dh = clip(-K_HEIGHT * height_scale * ey_err, -max_dh, max_dh)
self.hover['height'] += dh * dt
```

The gate is considered centered when:

```python
centred = abs(ex_t) <= APPROACH_TOL_X and abs(ey_err) <= APPROACH_TOL_Y
```

If centered but still far away, the drone creeps forward:

```python
x_cmd = CHASE_FORWARD
```

If not centered, it holds forward motion at zero while continuing yaw/lateral/height correction.

The normal transition to `PUSH` requires the gate to be both large and centered for several consecutive frames:

```python
if centred and area_frac > PASS_AREA_FRAC:
    self._push_confirm += 1
```

There is also a fallback push for cases where the drone is close enough but area growth stalls. This fallback currently requires vertical alignment too:

```python
fallback_push = (
    abs(ey_err) <= APPROACH_TOL_Y
    and area_frac >= PUSH_FALLBACK_AREA_FRAC
    and self._chase_area_stall >= PUSH_AREA_STALL_FRAMES
)
```

Note: this fallback checks vertical alignment but does not require horizontal alignment. If the drone pushes sideways through the gate, consider changing it to require `centred` instead of only `abs(ey_err) <= APPROACH_TOL_Y`.

### CHASE_RECOVER

If `CHASE` times out, the drone performs a local recovery search before returning to the wider `SEARCH` scan.

Recovery sequence:

1. Move height up by `CHASE_RECOVER_Z_DELTA`.
2. Move height down below the original recovery height.
3. Return to the original recovery height.
4. Yaw one direction, then the other.
5. If still not found, return to `SEARCH`.

If the target gate is detected again during recovery, the code immediately returns to `CHASE`.

### PUSH

The drone stops visual servoing and simply drives forward:

```python
x_cmd = FORWARD_SPEED
```

It exits `PUSH` when either:

- Odometry says it has travelled `PASS_THROUGH_DIST` from the push start point.
- The safety timeout `PUSH_MAX_DURATION` expires.

After each push, `_gates_passed` increments. If `_gates_passed >= MAX_GATES`, the drone enters `DONE` and stops.

### STOP And DONE

- Escape sets state to `STOP`, which sends zero world velocity continuously.
- Space sets state to `DONE`, sends stop setpoint, and stops the timer.

## Vision Pipeline

### Frame Reception

`UdpVideoThread` receives AI-deck UDP packets, reconstructs JPEG frames, decodes them with OpenCV, converts BGR to RGB, and emits each frame to the Qt GUI.

The runtime image size is read from the AI-deck frame header. If it differs from the calibration size, `_set_runtime_calibration()` scales `fx`, `fy`, `cx`, and `cy` in memory.

### Undistortion

Before detection, frames are undistorted with:

```python
cv2.undistort(rgb_img, _camera_matrix(), DIST_COEFFS)
```

This matters because the world projection uses calibrated image rays. Bad calibration causes poor zone validation.

### Detection

Despite the variable names `GREEN_HSV_LO` and `GREEN_HSV_HI`, the current detector thresholds by grayscale brightness:

```python
gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
threshold = max(GREEN_HSV_LO[2], GREEN_MIN_V)
mask = gray >= threshold
```

Then it closes the mask with a morphology kernel and extracts external contours.

Each contour is accepted only if it passes `_gate_candidate()` checks:

- Large enough area: `MIN_GREEN_AREA_FRAC`.
- Polygon vertex count between `GATE_MIN_VERTICES` and `GATE_MAX_VERTICES`.
- Bounding-box aspect ratio between `GATE_ASPECT_MIN` and `GATE_ASPECT_MAX`.
- Solidity above `GATE_MIN_SOLIDITY`.

For every accepted candidate, the code computes:

- Polygon centroid `cx`, `cy`.
- Normalized image errors `ex`, `ey`.
- Area.
- Bounding box.
- Ordered quad corners for world projection.

The selected detection returned by `_detect_green_gate()` is the rightmost candidate, but in `CHASE` the controller calls `_select_target_candidate()` and prefers candidates that validate into the expected gate zone.

## World Projection And Zone Validation

The world projection is used only for gate identity and the map. It is not used to command the drone toward a world point.

`_gate_to_world()` works like this:

1. Reject the candidate if its corners touch the image border, because clipped corners give bad geometry.
2. Convert each image corner into a calibrated body-frame ray.
3. Rotate rays into the world frame using the logged attitude quaternion.
4. Solve the depth of the left and right vertical gate edges by enforcing the known physical height `GATE_PHYS_H`.
5. Average the four world corners to estimate the gate center.

The output `(gx, gy, gz)` is passed to `GateZoneMap.validate_and_snap()`.

`GateZoneMap` divides the course into 12 angular zones around `COURSE_CENTER`. Gate `i` is expected in zone `2*i - 1`. If the estimated gate bearing is close enough to the expected zone, it is accepted and snapped into that zone. If it is too far outside, it is rejected as the wrong gate or a false detection.

## Parameter Reference And Tuning

Tune only one group at a time. After every change, use the debug overlay and printed `[calib]` lines to confirm what changed.

### Radio And Video

`URI`

- Crazyflie radio URI.
- Usually set by `CRAZYFLIE_URI` instead of editing the file.

`AIDECK_IP`, `AIDECK_PORT`, `LOCAL_PORT`

- UDP video stream connection settings.
- Change only if your AI-deck network setup differs.

`MIN_JPEG_BYTES`

- Rejects tiny/corrupt JPEG frames.
- Lower only if valid frames are being dropped. Raise if corrupted frames pass through.

### Camera Calibration

`CALIBRATION_PATH`

- Path to `test/calibration.json`.

`CAMERA_FX`, `CAMERA_FY`, `CAMERA_CX`, `CAMERA_CY`, `DIST_COEFFS`

- Loaded from calibration.
- Do not tune manually unless recalibrating the camera.

If zone validation is inconsistent even though the visible detection looks correct, suspect calibration, camera mounting, or `GATE_PHYS_H` before changing control gains.

### Brightness Detection

`GREEN_HSV_LO`, `GREEN_HSV_HI`

- Historical names. The current code only uses `GREEN_HSV_LO[2]` as part of the grayscale threshold.
- `GREEN_HSV_HI` is currently unused.

`GREEN_MIN_V`

- Main brightness threshold.
- Increase if the detector picks up too many background highlights.
- Decrease if the gate is visible but not detected.
- Current value is strict: `240`.

`MORPH_KERNEL`

- Kernel used for morphological closing.
- Larger kernel bridges larger gaps but may merge nearby objects.
- Smaller kernel preserves details but can break a gate into multiple contours.

`MIN_GREEN_AREA_FRAC`

- Minimum contour area as fraction of image area.
- Increase to reject small noise.
- Decrease if gates are missed when far away.
- If too low, SEARCH may validate more false candidates.

### Gate Shape Filtering

`GATE_MIN_VERTICES`, `GATE_MAX_VERTICES`

- Allowed polygon vertex count after approximation.
- Widen if valid gates are rejected because contours are noisy.
- Tighten if non-gate blobs pass as candidates.

`GATE_ASPECT_MIN`, `GATE_ASPECT_MAX`

- Allowed bounding-box width/height ratio.
- Widen if gates are seen from strong perspective angles.
- Tighten if elongated blobs are accepted.

`GATE_MIN_SOLIDITY`

- Contour area divided by convex hull area.
- Higher values reject irregular shapes.
- Lower values accept more broken/noisy gates.

`GATE_APPROX_EPS`

- Polygon simplification strength.
- Larger values simplify more aggressively and may turn noisy contours into clean quads.
- Too large can distort the gate shape.

`GATE_BORDER_MARGIN`

- Rejects world projection when gate corners are too close to image edge.
- Increase if clipped gates produce bad map estimates.
- Decrease if close gates are often rejected too early.

### Search And Gate Locking

`SEARCH_YAWRATE`

- Yaw scan speed in deg/s during `SEARCH`.
- Increase to find gates faster.
- Decrease if the drone scans past gates before collecting enough confirmed frames.

`SEARCH_CONFIRM_FRAMES`

- Number of consecutive in-zone detections required before `CHASE`.
- Increase to reduce false locks.
- Decrease if the drone sees the gate but never transitions to `CHASE`.

`CHASE_TIMEOUT`

- Time in seconds before `CHASE` gives up and enters `CHASE_RECOVER`.
- Increase if the gate is briefly lost but reacquired reliably.
- Decrease if the drone wastes time holding after a bad lock.

`CHASE_RETURN_K`, `CHASE_RETURN_MAX_SPEED`, `CHASE_RETURN_TOL`

- Used when the target gate is temporarily lost during `CHASE`.
- The drone tries to return to the last pose where the target gate was visible.
- Increase `CHASE_RETURN_K` or max speed if it returns too slowly.
- Decrease if it oscillates or moves too aggressively.

### Takeoff

`TAKEOFF_HEIGHT`

- Initial target height.
- Should roughly match the course/gate height but can be lower if the visual servo should climb into position.

`TAKEOFF_YAW_DEG`

- Initial yaw target before searching.
- Set to face the likely first gate direction.

`TAKEOFF_YAW_TOL`

- Allowed yaw error before entering `SEARCH`.
- Decrease for more precise initial heading.
- Increase if takeoff spends too long rotating.

`K_TAKEOFF_YAW`

- Proportional yaw gain during takeoff.
- If the drone rotates the wrong way, the sign convention is wrong.
- If it oscillates, reduce it.

`TAKEOFF_RATE`

- Climb rate during takeoff.
- Increase to take off faster.
- Decrease for gentler takeoff.

### Horizontal Visual Servo

`K_YAW`

- Converts horizontal image error `ex` into yaw rate.
- Increase if the gate stays off-center for too long.
- Decrease if yaw oscillates or overshoots.

`MAX_YAWRATE`

- Clamp for yaw rate.
- Increase only if the drone clearly cannot turn fast enough.
- Decrease if yaw motion is too aggressive.

`K_LATERAL`

- Converts horizontal image error into lateral body velocity.
- This nudges the drone sideways while yaw also turns toward the gate.
- Increase if yaw alone cannot center the gate.
- Decrease if the drone slides too much or approaches diagonally.

`MAX_LATERAL`

- Clamp for lateral speed.
- Increase carefully; excessive lateral motion can miss the gate.

`APPROACH_TOL_X`

- Horizontal centering tolerance before the drone is allowed to creep forward.
- Decrease for stricter centering and safer gate entry.
- Increase if the drone refuses to approach because `ex` jitters around the threshold.

### Vertical Visual Servo

`APPROACH_TARGET_EY`

- Desired vertical image offset during approach.
- Positive means the gate should appear below image center, so the drone flies slightly higher.
- Increase if the drone is still too low before pushing.
- Decrease if the drone is too high or hits the top of the gate.

`APPROACH_TOL_Y`

- Allowed vertical error around `APPROACH_TARGET_EY`.
- Decrease if the drone pushes while still too high/low.
- Increase if the drone never creeps forward because vertical detection is noisy.

`K_HEIGHT`

- Converts vertical image error into height correction speed.
- Increase if height converges too slowly.
- Decrease if height oscillates.

`MAX_DH_PER_S`

- Maximum height change rate.
- Increase if the drone cannot climb fast enough before reaching the gate.
- Decrease if height changes are too abrupt.

`HEIGHT_SLOWDOWN_AREA_FRAC`

- Area fraction where height correction starts being reduced near the gate.
- Increase if height corrections become noisy too early.
- Decrease if the drone needs full height control until it is closer.

`HEIGHT_MIN_SCALE`

- Minimum fraction of height gain/rate kept near pass-through.
- Increase if the drone is still below the gate right before push.
- Decrease if close-range height estimates are noisy and cause vertical oscillations.

`MIN_HEIGHT`, `MAX_HEIGHT`

- Hard clamps for the height setpoint.
- If the drone needs to fly higher but cannot, check `MAX_HEIGHT`.
- Keep safety margins for the room and gate setup.

### Approach And Push

`CHASE_FORWARD`

- Forward speed during centered approach before `PUSH`.
- Increase to approach faster.
- Decrease if the drone does not have enough time to align vertically/horizontally.

`PASS_AREA_FRAC`

- Area fraction threshold for normal push commit.
- Increase if the drone pushes too early.
- Decrease if it gets very close but never pushes.
- The overlay prints both the live area and this threshold.

`PASS_CONFIRM_FRAMES`

- Consecutive big-and-centered frames required before normal push.
- Increase for robustness.
- Decrease if push is delayed too much.

`PUSH_FALLBACK_AREA_FRAC`

- Minimum area fraction for fallback push if area growth stalls.
- Should usually be below `PASS_AREA_FRAC`.
- Increase if fallback push happens too early.
- Decrease if the drone gets stuck close to the gate without pushing.

`PUSH_AREA_STALL_FRAMES`

- Number of frames without meaningful area growth before fallback push.
- Increase to wait longer.
- Decrease to push sooner when stuck.

`PUSH_AREA_GROWTH_EPS`

- Minimum area increase required to reset the stall counter.
- Increase if small noisy area changes prevent fallback.
- Decrease if noise resets the stall counter too easily.

`FORWARD_SPEED`

- Forward speed during `PUSH`.
- Increase if the drone does not clear the gate reliably.
- Decrease if push is too aggressive.

`PASS_THROUGH_DIST`

- Odometry distance to travel during `PUSH`.
- Increase if the drone exits push before clearing the gate.
- Decrease if it pushes too far after passing.

`PUSH_MAX_DURATION`

- Safety timeout for push.
- Should be long enough to cover `PASS_THROUGH_DIST / FORWARD_SPEED` plus margin.
- Example with current values: `1.2 m / 0.1 m/s = 12 s`, but `PUSH_MAX_DURATION` is `10 s`, so the timeout may end push before distance does. Either increase `PUSH_MAX_DURATION` or reduce `PASS_THROUGH_DIST`/increase `FORWARD_SPEED` if this is not intended.

### Course And Zone Validation

`GATE_PHYS_H`

- Physical gate height in meters.
- Critical for `_gate_to_world()` depth estimation.
- If wrong, zone validation can reject valid gates or accept bad ones.

`VERT_PAIR_RESIDUAL_MAX`

- Maximum allowed residual when solving vertical gate edges.
- Increase if valid gates often fail projection due to noise.
- Decrease if bad geometry produces unstable map estimates.

`COURSE_CENTER` in `gate_map.py`

- Center point of the circular zone model.
- If wrong, zone validation can reject correct gates.

`DEFAULT_ACCEPT_MARGIN_DEG` in `gate_map.py`

- Extra angular margin outside the strict zone.
- Increase if valid detections are slightly outside the expected zone because projection is noisy.
- Decrease if wrong gates are accepted.

`MAX_GATES`

- Number of gates to pass before stopping.
- Current `gates_xyz.py` contains four gate rows, and `MAX_GATES` is currently `4`.

## Practical Tuning Workflow

### 1. Confirm Video And Detection

Start with the drone safe and watch the FPV window.

Check:

- The image is not upside down or badly delayed.
- The gate is outlined in green/yellow.
- The printed `[calib]` line shows reasonable `area`, `ex`, and `ey`.

If the gate is not detected:

- Lower `GREEN_MIN_V` if the gate is too dim.
- Lower `MIN_GREEN_AREA_FRAC` if the gate is far and small.
- Relax shape filters only after brightness detection is working.

If false objects are detected:

- Raise `GREEN_MIN_V`.
- Raise `MIN_GREEN_AREA_FRAC`.
- Tighten `GATE_ASPECT_MIN/MAX` or increase `GATE_MIN_SOLIDITY`.

### 2. Confirm Zone Validation

In `SEARCH`, the drone should only switch to `CHASE` after seeing the expected gate for `SEARCH_CONFIRM_FRAMES` frames.

If the gate is visibly detected but never enters `CHASE`:

- Check printed `[gate ray]` estimates.
- Check `test/calibration.json`.
- Check `GATE_PHYS_H`.
- Check `COURSE_CENTER` and `gates_xyz.py`.
- Reduce `SEARCH_CONFIRM_FRAMES` only if the validation is good but too intermittent.
- Increase `DEFAULT_ACCEPT_MARGIN_DEG` only if the gate estimate is close but just outside the zone.

### 3. Tune Horizontal Centering

In `CHASE`, watch `ex`.

If the drone is slow to face the gate:

- Increase `K_YAW`.
- Increase `MAX_YAWRATE` only if the clamp is limiting.

If it oscillates left/right:

- Decrease `K_YAW`.
- Decrease `MAX_YAWRATE`.
- Consider decreasing `K_LATERAL`.

If it faces the gate but approaches from the side:

- Increase `K_LATERAL` slightly.
- Decrease `CHASE_FORWARD` to give lateral correction more time.

### 4. Tune Vertical Centering

Watch `ey` and compare it to `APPROACH_TARGET_EY`.

If the drone is too low before pushing:

- Increase `APPROACH_TARGET_EY`.
- Increase `K_HEIGHT`.
- Increase `MAX_DH_PER_S`.
- Increase `HEIGHT_MIN_SCALE` so close-range height correction remains stronger.
- Decrease `CHASE_FORWARD` so the drone has more time to climb.
- Decrease `APPROACH_TOL_Y` so it cannot push while vertically misaligned.

If the drone is too high:

- Decrease `APPROACH_TARGET_EY`.
- Decrease `K_HEIGHT` if it overshoots.

If height oscillates near the gate:

- Decrease `K_HEIGHT`.
- Decrease `HEIGHT_MIN_SCALE`.
- Increase `HEIGHT_SLOWDOWN_AREA_FRAC` so slowdown begins earlier.
- Increase `APPROACH_TOL_Y` only if the oscillation is small and acceptable.

### 5. Tune Push Timing

Use the overlay `area=... PASS=...` while manually observing the gate size at the desired push moment.

If it pushes too early:

- Increase `PASS_AREA_FRAC`.
- Increase `PASS_CONFIRM_FRAMES`.
- Increase `PUSH_FALLBACK_AREA_FRAC`.
- Increase `PUSH_AREA_STALL_FRAMES`.

If it never pushes:

- Decrease `PASS_AREA_FRAC`.
- Decrease `PASS_CONFIRM_FRAMES`.
- Decrease `PUSH_FALLBACK_AREA_FRAC`.
- Check whether `APPROACH_TOL_X` or `APPROACH_TOL_Y` is too strict.

If it pushes before being horizontally centered because of fallback:

Change fallback from vertical-only alignment:

```python
abs(ey_err) <= APPROACH_TOL_Y
```

to full centering:

```python
centred
```

### 6. Tune Push Distance

If the drone does not clear the gate:

- Increase `PASS_THROUGH_DIST`.
- Increase `FORWARD_SPEED`.
- Increase `PUSH_MAX_DURATION`.

If it flies too far after the gate:

- Decrease `PASS_THROUGH_DIST`.
- Decrease `FORWARD_SPEED`.

Check the relationship:

```text
PUSH_MAX_DURATION >= PASS_THROUGH_DIST / FORWARD_SPEED
```

With the current values, the timeout is shorter than the distance-based duration, so timeout may dominate.

## Debug Output

`DEBUG_CALIB = True` prints live detection values:

```text
[calib] area= ... bbox= ... ex=... ey=... bbox_px=... aspect=... cands=...
```

Use this to tune:

- `PASS_AREA_FRAC` from `area`.
- `APPROACH_TOL_X` from `ex` jitter.
- `APPROACH_TARGET_EY` and `APPROACH_TOL_Y` from `ey`.
- Shape filters from `bbox_px`, `aspect`, and candidate count.

`DEBUG_GATE_POSE = True` prints world-projection estimates:

```text
[gate ray] range=... angle=... drone(...) -> gate(...)
```

Use this to debug:

- Bad calibration.
- Wrong `GATE_PHYS_H`.
- Bad course center or gate positions.
- Wrong candidate being validated.

## Common Problems

### Drone Is Under The Gate Before Push

Likely causes:

- `APPROACH_TARGET_EY` too low.
- `K_HEIGHT` too low.
- `MAX_DH_PER_S` too low.
- `HEIGHT_MIN_SCALE` too low near the gate.
- `CHASE_FORWARD` too high, leaving too little time to climb.
- `APPROACH_TOL_Y` too loose.
- Fallback push triggering before full alignment.

Current fallback now requires vertical alignment, but only vertical alignment. If needed, require full `centred` there.

### Drone Sees Gate But Does Not Chase

Likely causes:

- Detection passes image filters but `_gate_to_world()` fails.
- Gate is clipped by the image border.
- Camera calibration or `GATE_PHYS_H` is wrong.
- Zone map rejects the estimate.
- `SEARCH_CONFIRM_FRAMES` is too high for the scan speed.

### Drone Chases Wrong Gate

Likely causes:

- Zone margin too permissive.
- Ground-truth gate positions/course center are wrong.
- World projection is noisy.
- Multiple bright objects pass shape filtering.

Tune detection first, then projection, then zone margin.

### Drone Oscillates Around Gate Center

Likely causes:

- `K_YAW`, `K_LATERAL`, or `K_HEIGHT` too high.
- Tolerances too tight.
- Close-range image centroid is noisy.
- `HEIGHT_MIN_SCALE` too high near pass-through.

### Drone Gets Stuck Close To Gate

Likely causes:

- `PASS_AREA_FRAC` too high.
- `APPROACH_TOL_X` or `APPROACH_TOL_Y` too strict.
- `PUSH_FALLBACK_AREA_FRAC` too high.
- `PUSH_AREA_STALL_FRAMES` too high.
- Area growth noise keeps resetting the stall counter; increase `PUSH_AREA_GROWTH_EPS`.

## Safety Notes

- Keep Escape ready: it enters `STOP` and commands zero velocity.
- Space cuts autonomy harder by sending a stop setpoint and stopping the timer.
- Tune at low speed first: `CHASE_FORWARD` and `FORWARD_SPEED` are the main approach/push speed knobs.
- Do not raise `MAX_HEIGHT`, `MAX_DH_PER_S`, or speed parameters without checking the flight area.
- Change one parameter at a time and use the printed logs to verify the effect.

## Suggested Initial Tuning Order

1. Detection: `GREEN_MIN_V`, `MIN_GREEN_AREA_FRAC`, shape filters.
2. Zone validation: calibration, `GATE_PHYS_H`, `COURSE_CENTER`, `DEFAULT_ACCEPT_MARGIN_DEG`.
3. Search lock: `SEARCH_YAWRATE`, `SEARCH_CONFIRM_FRAMES`.
4. Horizontal servo: `K_YAW`, `K_LATERAL`, `APPROACH_TOL_X`.
5. Vertical servo: `APPROACH_TARGET_EY`, `K_HEIGHT`, `MAX_DH_PER_S`, `HEIGHT_MIN_SCALE`, `APPROACH_TOL_Y`.
6. Approach speed: `CHASE_FORWARD`.
7. Push trigger: `PASS_AREA_FRAC`, `PASS_CONFIRM_FRAMES`, fallback parameters.
8. Push clearing: `FORWARD_SPEED`, `PASS_THROUGH_DIST`, `PUSH_MAX_DURATION`.
