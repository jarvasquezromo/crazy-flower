# -*- coding: utf-8 -*-
"""
Crazyflie Position Snapshot Logger
===================================
Connects to a Crazyflie, streams position (x, y, z) and yaw continuously.
Press ENTER to capture a snapshot:
  - Takes the 10 samples BEFORE the keypress (already buffered)
  - Waits for 10 samples AFTER the keypress
  - Computes mean and variance for each variable
  - Appends the result to 'snapshots.csv'

Press 'q' + ENTER to quit (or Ctrl+C).

Usage:
    pip install cflib pynput
    python log_position_snapshot.py
"""

import os
import csv
import time
import logging
import numpy as np

from collections import deque
from threading import Event, Lock, Thread

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

# ── Configuration ────────────────────────────────────────────────────────────

# TODO: change to match your Crazyflie address / radio channel
URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E705')
# URI = uri_helper.uri_from_env(default='tcp://192.168.4.1:5000')

LOG_PERIOD_MS  = 50          # logging period (ms) → 20 Hz
WINDOW_SIZE    = 10          # samples before AND after the keypress
OUTPUT_FILE    = 'snapshots.csv'
GATES_FILE     = 'gates_xyz.py'
VARIABLES      = ['stateEstimate.x', 'stateEstimate.y',
                  'stateEstimate.z', 'stabilizer.yaw']
SHORT_NAMES    = ['x', 'y', 'z', 'yaw']   # used in CSV headers
VAR_ACCEPTED   = [0.01, 0.01, 0.01, 0.1]

logging.basicConfig(level=logging.ERROR)

# ── Statistics helpers ────────────────────────────────────────────────────────

def stats(samples: list[dict], key: str, num_dig: int=3) -> tuple[float, float]:
    vals = [s[key] for s in samples]
    return np.mean(vals).round(num_dig), np.std(vals).round(num_dig + 1)

# ── CSV output ────────────────────────────────────────────────────────────────

def build_csv_headers() -> list[str]:
    headers = ['snapshot_id', 'trigger_timestamp']
    for name in SHORT_NAMES:
        headers += [
            f'{name}_mean',    f'{name}_var',
        ]
    return headers

def write_snapshot(snapshot_id: int, trigger_ts: float,
                   before: list[dict], after: list[dict]) -> None:
    all_samples = before + after
    file_exists = os.path.isfile(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=build_csv_headers())
        if not file_exists:
            writer.writeheader()

        row = {'snapshot_id': snapshot_id,
               'trigger_timestamp': f'{trigger_ts:.3f}'}

        for var, name in zip(VARIABLES, SHORT_NAMES):
            m_all, v_all = stats(all_samples, var, num_dig=3)
            row[f'{name}_mean']    = f'{m_all:.2f}'
            row[f'{name}_var']     = f'{v_all:.3f}'

        writer.writerow(row)

        # Pretty-print to terminal
        print(f'\n── Snapshot #{snapshot_id} saved (trigger @ {trigger_ts:.3f}s) ──')
        print(f'  {"Variable":<6} {"mean":>10} {"var":>10} {"check":>10}')
        for i, var, name in zip(range(len(VAR_ACCEPTED)), VARIABLES, SHORT_NAMES):
            names = (name + '_mean', name + '_var')
            check = '✔' if VAR_ACCEPTED[i] >= float(row[names[1]]) else '✘'
            print(f"  {name:<6} {row[names[0]]:>12s} {row[names[1]]:>12s} {check:>6}")
    print(f'  → appended to {OUTPUT_FILE}\n')

    # Only write a gate if x,y,z variances are within accepted thresholds.
    good_xyz = all(
        float(row[f'{name}_var']) <= VAR_ACCEPTED[idx]
        for idx, name in enumerate(SHORT_NAMES[:3])
    )
    if good_xyz:
        append_gate_point(
            float(row['x_mean']),
            float(row['y_mean']),
            float(row['z_mean']),
        )


def append_gate_point(x: float, y: float, z: float) -> None:
    line = f"  [{x:.2f}, {y:.2f}, {z:.2f}],\n"

    if not os.path.isfile(GATES_FILE):
        with open(GATES_FILE, 'w', newline='') as f:
            f.write('[\n')
            f.write(line)
            f.write(']\n')
        print(f'  → gate appended to {GATES_FILE}')
        return

    with open(GATES_FILE, 'r', newline='') as f:
        lines = f.readlines()

    if not lines:
        lines = ['[\n', ']\n']

    # Ensure trailing closing bracket exists, then insert before it.
    if lines[-1].strip() != ']':
        lines.append(']\n')
    lines.insert(len(lines) - 1, line)

    with open(GATES_FILE, 'w', newline='') as f:
        f.writelines(lines)

    print(f'  → gate appended to {GATES_FILE}')

# ── Main logger class ─────────────────────────────────────────────────────────

class SnapshotLogger:
    def __init__(self, link_uri: str):
        self._uri          = link_uri
        self._cf           = Crazyflie(rw_cache='./cache')
        self._lock         = Lock()
        self._buffer       = deque(maxlen=WINDOW_SIZE)  # rolling pre-trigger window
        self._capturing    = False     # True while collecting post-trigger samples
        self._post_samples : list[dict] = []
        self._pre_snapshot : list[dict] = []
        self._trigger_ts   = 0.0
        self._snapshot_id  = 0
        self._snapshot_ready = Event()  # signals that post-window is full
        self.is_connected  = False

        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)

        print(f'Connecting to {link_uri} …')
        self._cf.open_link(link_uri)

    # ── Crazyflie callbacks ──────────────────────────────────────────────────

    def _connected(self, link_uri: str) -> None:
        print(f'Connected to {link_uri}')
        self.is_connected = True

        lg = LogConfig(name='Stabilizer', period_in_ms=LOG_PERIOD_MS)
        for var in VARIABLES:
            lg.add_variable(var, 'float')

        try:
            self._cf.log.add_config(lg)
            lg.data_received_cb.add_callback(self._on_data)
            lg.error_cb.add_callback(self._on_error)
            lg.start()
            print('Logging started.')
            print('Press ENTER to capture a snapshot.')
            print('Press q + ENTER (or Ctrl-C) to quit.\n')
        except (KeyError, AttributeError) as e:
            print(f'Could not start log config: {e}')

    def _disconnected(self, link_uri: str) -> None:
        print(f'Disconnected from {link_uri}')
        self.is_connected = False

    def _connection_failed(self, link_uri: str, msg: str) -> None:
        print(f'Connection failed to {link_uri}: {msg}')
        self.is_connected = False

    def _connection_lost(self, link_uri: str, msg: str) -> None:
        print(f'Connection lost to {link_uri}: {msg}')

    # ── Data callback ────────────────────────────────────────────────────────

    def _on_data(self, timestamp: int, data: dict, logconf) -> None:
        sample = dict(data)          # e.g. {'stateEstimate.x': 0.1, …}
        sample['_ts'] = timestamp

        with self._lock:
            if self._capturing:
                self._post_samples.append(sample)
                if len(self._post_samples) >= WINDOW_SIZE:
                    self._capturing = False
                    self._snapshot_ready.set()
            else:
                self._buffer.append(sample)   # rolling pre-trigger ring buffer

    def _on_error(self, logconf, msg: str) -> None:
        print(f'Log error [{logconf.name}]: {msg}')

    # ── Trigger ──────────────────────────────────────────────────────────────

    def trigger_snapshot(self) -> None:
        """Call this when the user presses the key."""
        with self._lock:
            if self._capturing:
                print('  (snapshot already in progress, please wait)')
                return
            if len(self._buffer) < WINDOW_SIZE:
                print(f'  (need at least {WINDOW_SIZE} samples first, '
                      f'have {len(self._buffer)} — wait a moment)')
                return

            self._pre_snapshot = list(self._buffer)   # freeze the last 10
            self._post_samples = []
            self._trigger_ts   = time.time()
            self._capturing    = True
            self._snapshot_ready.clear()
            print(f'  Snapshot #{self._snapshot_id + 1}: capturing {WINDOW_SIZE} '
                  f'post-trigger samples …')

        # Wait for post-window to fill (blocking the caller thread is fine)
        self._snapshot_ready.wait()

        with self._lock:
            pre  = self._pre_snapshot
            post = list(self._post_samples)
            ts   = self._trigger_ts
            self._snapshot_id += 1
            sid  = self._snapshot_id

        write_snapshot(sid, ts, pre, post)

    def disconnect(self) -> None:
        self._cf.close_link()

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cflib.crtp.init_drivers()
    logger = SnapshotLogger(URI)

    # Wait until connected before entering the input loop
    while not logger.is_connected:
        time.sleep(0.1)

    try:
        while logger.is_connected:
            try:
                key = input()          # blocks; user presses ENTER to submit
            except EOFError:
                break

            key = key.strip().lower()
            if key in ('q', 'quit', 'exit'):
                break
            else:
                # Any other input (including bare ENTER / SPACE+ENTER) triggers
                logger.trigger_snapshot()

    except KeyboardInterrupt:
        pass
    finally:
        print('Disconnecting …')
        logger.disconnect()
        print('Done.')

if __name__ == '__main__':
    main()