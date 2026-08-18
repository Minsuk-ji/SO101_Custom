# SO101_Custom

A from-scratch reimplementation of the SO-101 robot arm pipeline, built to understand and
customize each step instead of depending on the [lerobot](https://github.com/huggingface/lerobot)
library as a black box. Each stage (motor mapping, calibration, teleoperation, data recording,
training, evaluation) is added incrementally, in its own folder.

## Progress

- [x] `motor_setup/` — motor ID mapping
- [x] `calibration/` — homing offset + range of motion
- [x] `teleoperate/` — leader -> follower real-time control loop
- [x] `record/` — camera-inclusive episode recording (rerun viz, local or HF Hub storage)
- [x] `training/` — pluggable multi-policy training (mlp_bc, ACT)
- [ ] evaluation

## Setup

```bash
conda create -n so101_custom python=3.12 -y
conda activate so101_custom
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-enabled `torch` build first (see
[pytorch.org](https://pytorch.org/get-started/locally/) — pip's default wheel may resolve to a
CPU-only build depending on your platform), then run the `pip install -r requirements.txt` above;
it will leave an already-satisfied `torch` alone.

All commands below assume this environment is active (no `conda run` prefix needed once you've
run `conda activate so101_custom`).

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

Run from the repo root (`so101_custom/`), so `motor_setup` resolves as a package:

```bash
# Assign IDs one motor at a time (connect only the named motor when prompted)
python -m motor_setup.setup_motors --port /dev/ttyACM0

# Verify: scan a fully assembled arm and list id -> joint name
python -m motor_setup.scan_bus --port /dev/ttyACM0
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
python -m calibration.calibrate --port /dev/ttyACM0 --name leader

# Re-apply a previously saved calibration and print each joint's current position
python -m calibration.apply_calibration --port /dev/ttyACM0 --name leader
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
python -m teleoperate.teleoperate \
    --leader-port /dev/ttyACM1 --leader-name leader \
    --follower-port /dev/ttyACM0 --follower-name follower
```

## `record/`

Teleoperates the follower while recording (observation, action) pairs plus camera video into
episodes, and shows them live in [rerun](https://rerun.io/). Ported from lerobot's
`lerobot_record.py` core loop; `LeRobotDataset`'s multi-episode chunking/streaming video encoding
is dropped in favor of one plain folder per episode (simpler to read back, and unnecessary at
personal-project scale). Storage is local-first; pushing to the HF Hub is a separate opt-in step
using `huggingface_hub` directly — not the `lerobot` library.

| File | What it does |
|---|---|
| `camera.py` | OpenCV camera wrapper — connect, read an RGB frame, disconnect |
| `dataset_writer.py` | Writes one episode per folder: `frames.npz` (state/action arrays) + `videos/<camera>.mp4` (via `cv2.VideoWriter`), plus a dataset-level `info.json` |
| `hub_upload.py` | Uploads a local dataset folder to a HF Hub dataset repo (`huggingface_hub.HfApi.upload_folder`) |
| `record.py` | Main CLI — connects leader+follower (reusing `motor_setup`/`calibration`/`teleoperate`) and cameras, records `--num-episodes` episodes, logs every frame to rerun, and optionally pushes to the Hub at the end |

Frame convention (matches lerobot): `observation.state` is the follower's position *before* the
action is applied, `action` is the leader's position for that frame — so `(state_t, action_t) ->
state_{t+1}` pairs are ready for imitation learning as-is.

### Usage

```bash
python -m record.record \
    --leader-port /dev/ttyACM1 --leader-name leader \
    --follower-port /dev/ttyACM0 --follower-name follower \
    --camera front=0 --camera wrist=2 \
    --root ~/so101_data/pick_cube --task "pick up the red cube" \
    --num-episodes 20 --episode-time 15 --reset-time 5 \
    --push-to-hub --repo-id myuser/so101-pick-cube   # optional
```

## `training/`

Trains a policy on a `record/`-produced dataset via imitation learning (behavior cloning). The
key design goal here is **not being locked into one model**: `train.py` only depends on the
`Policy` interface (`policy_base.py`) and a name -> class `registry.py`, so adding a new policy
never requires touching the training script — drop a file in `policies/`, decorate the class with
`@register_policy("name")`, and `--policy name` picks it up.

| File | What it does |
|---|---|
| `policy_base.py` | Abstract `Policy` interface every policy implements: `forward(batch) -> (loss, logs)`, `select_action(batch) -> action`, `reset()` |
| `registry.py` | `{name: class}` dict + `@register_policy` decorator + lookup — the pluggability mechanism |
| `dataset.py` | `SO101Dataset` — reads `record/` output, decodes each episode's video once and caches it, yields `(state, action_chunk, images)` samples |
| `checkpoint_io.py` | Save/load a checkpoint locally (`torch.save`), or push/pull it to a HF Hub model repo |
| `policies/mlp_bc.py` | Baseline: small per-camera CNN + MLP, single-step (or short-chunk) action prediction — fast sanity check / comparison point |
| `policies/act.py` | Trimmed [ACT](https://tonyzhaozh.github.io/aloha/) (Action Chunking Transformer): CNN backbone per camera -> transformer encoder (+ CVAE latent from the target action chunk during training) -> transformer decoder predicting a whole action chunk at once |
| `train.py` | Policy-agnostic training loop: dataset -> `DataLoader` -> `policy.forward(batch)` -> backward -> checkpoint, identical regardless of `--policy` |

### Usage

```bash
# ACT
python -m training.train \
    --policy act --dataset ~/so101_data/pick_cube \
    --checkpoint-dir ~/so101_checkpoints/pick_cube_act \
    --epochs 100 --batch-size 8

# Simple baseline, same dataset, same command shape
python -m training.train \
    --policy mlp_bc --dataset ~/so101_data/pick_cube \
    --checkpoint-dir ~/so101_checkpoints/pick_cube_mlp \
    --push-to-hub --repo-id myuser/so101-pick-cube-mlp   # optional
```
