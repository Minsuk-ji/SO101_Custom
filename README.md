# SO101_Custom

A from-scratch reimplementation of the SO-101 robot arm pipeline, built to understand and
customize each step instead of depending on the [lerobot](https://github.com/huggingface/lerobot)
library as a black box. Each stage (motor mapping, calibration, teleoperation, data recording,
training, evaluation) is added incrementally, in its own folder.

## Progress

- [x] `motor_setup/` — motor ID mapping
- [x] `calibration/` — homing offset + range of motion
- [x] `teleoperate/` — leader -> follower real-time control loop
- [ ] data recording
- [ ] training
- [ ] evaluation

## `motor_setup/`

Assigns a unique ID (1-6) to each STS3215 servo of an SO-101 arm over its serial bus. Ported
from lerobot's `motors_bus.py` / `feetech.py`, trimmed down to only what motor ID mapping needs.

| File | What it does |
|---|---|
| `sts3215_table.py` | STS3215 control-table register addresses (ID, Baud_Rate, Lock, ...), baud-rate table, and the SO-101 joint-name -> target ID mapping |
| `feetech_bus.py` | Low-level serial bus wrapper: open/close port, scan baud rates, ping, and write the ID/baud-rate registers (via `scservo_sdk`) |
| `setup_motors.py` | Interactive CLI — prompts you to connect one motor at a time and burns its target ID |
| `scan_bus.py` | Pings IDs 1-252 on an already-assembled arm and prints which joint each ID answers as, to verify the mapping |

### Usage

Requires the `scservo_sdk` package (installed in the `lerobot` conda env here):

Run from the repo root (`so101_custom/`), so `motor_setup` resolves as a package:

```bash
# Assign IDs one motor at a time (connect only the named motor when prompted)
conda run -n lerobot python -m motor_setup.setup_motors --port /dev/ttyACM0

# Verify: scan a fully assembled arm and list id -> joint name
conda run -n lerobot python -m motor_setup.scan_bus --port /dev/ttyACM0
```

## `calibration/`

Computes and stores each joint's zero point (homing offset) and range of motion, so raw
encoder ticks can be mapped to normalized joint positions. Ported from lerobot's
`SOLeader.calibrate()` / `MotorsBus.set_half_turn_homings()` / `record_ranges_of_motion()`.
Builds on `motor_setup/feetech_bus.py` and `sts3215_table.py` rather than duplicating them.

| File | What it does |
|---|---|
| `sign_magnitude.py` | Encode/decode the sign-magnitude format STS3215 uses for signed registers (`Homing_Offset`, `Present_Position`) |
| `calibration_io.py` | Save/load a calibration as JSON under `calibration/data/<name>.json` |
| `calibrate.py` | Interactive CLI — center the arm to set the homing offset, then move each joint through its range to record min/max, and save |
| `apply_calibration.py` | Re-writes a saved calibration JSON back onto the motors (restore/verify without repeating the physical procedure) |

### Usage

```bash
# Run the interactive calibration (needs a display/keyboard nearby to move the arm by hand)
conda run -n lerobot python -m calibration.calibrate --port /dev/ttyACM0 --name leader

# Re-apply a previously saved calibration and print each joint's current position
conda run -n lerobot python -m calibration.apply_calibration --port /dev/ttyACM0 --name leader
```

## `teleoperate/`

Reads the leader arm's joint positions and mirrors them onto the follower arm in real time.
Ported from lerobot's `lerobot_teleoperate.py` teleop loop, with the camera/visualization/processor
pipeline stripped out. Builds on `motor_setup/` and `calibration/` rather than duplicating them.

| File | What it does |
|---|---|
| `normalization.py` | Converts raw encoder ticks <-> normalized position (-100..100, gripper 0..100) using each arm's own calibration, plus an optional per-step relative-motion clamp |
| `teleoperate.py` | Main loop: read leader (raw) -> normalize with leader calibration -> unnormalize with follower calibration -> write to follower `Goal_Position` |

Normalizing in between matters because the leader and follower have different `range_min`/`range_max`
(assembly variance, see `calibration/`) — copying raw ticks directly would send the follower to the
wrong angle. Converting through a common -100..100 scale makes "50% of the leader's range" map to
"50% of the follower's range".

### Usage

Requires both arms calibrated first (`calibration.calibrate`, once per arm):

```bash
conda run -n lerobot python -m teleoperate.teleoperate \
    --leader-port /dev/ttyACM1 --leader-name leader \
    --follower-port /dev/ttyACM0 --follower-name follower
```
