# SO101_Custom

A from-scratch reimplementation of the SO-101 robot arm pipeline, built to understand and
customize each step instead of depending on the [lerobot](https://github.com/huggingface/lerobot)
library as a black box. Each stage (motor mapping, calibration, teleoperation, data recording,
training, evaluation) is added incrementally, in its own folder.

## Progress

- [x] `motor_setup/` — motor ID mapping
- [ ] calibration
- [ ] teleoperation
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
