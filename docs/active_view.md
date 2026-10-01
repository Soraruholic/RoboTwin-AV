# Active-view usage

## Installation

Install RoboTwin 2.0 and its assets using the upstream instructions in the
root README. Use Linux and the RoboTwin simulation environment. The default
AV configuration supports `aloha-agilex` (six joints and one gripper per arm).
The OpenPI model server runs in its own environment; the simulator only needs
`openpi-client` when connecting to that server.

The extension is based on upstream commit
`ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`. It retains the upstream task
success functions, model enumeration, joint-path validation and native scripts.
Setting `active_view.enabled` to false leaves the native physics path active.

## Collection

```bash
bash collect_av.sh adjust_bottle demo_clean 0 --episodes 50 --max-attempts 250
bash collect_av.sh handover_block demo_clean 0 --mode tracking --episodes 10 \
  --output data/av_tracking
bash collect_av.sh stack_blocks_two demo_clean 0 --mode fixed --episodes 10 \
  --output data/av_fixed
```

The arguments follow `collect_data.sh`: task name, task configuration and GPU.
Extra arguments include `--seed`, `--split`, `--initial-head PAN TILT`,
`--timeout`, `--min-free-gib`, and `--settle-retries`. Collection defaults to
seeds starting at 1000. Use disjoint seeds and separate episode families for
training, validation and testing. `--dry-run` prints the resolved collection
options without starting a simulator.

Each episode runs in a separate process. Resume requires the exact same
arguments. The manifest includes only successful episodes passing the data
audit. Failed attempts, logs and partially recorded trajectories remain on
disk for inspection. A stale `.collection.lock` requires checking that its
recorded PID has exited before removing it. An incomplete episode directory
without a receipt must also be inspected before resuming.

```text
data/active_view/train/adjust_bottle/phase/
  collection_config.json
  manifest.json
  seed_001000.log
  seed_001000/
    episode.hdf5
    result.json
    audit.json
```

The gaze modes are:

| Mode | Head control |
| --- | --- |
| `fixed` | Keep the initial pan/tilt |
| `tracking` | Track the selected object's visible surface |
| `phase` | Frame task objects and end effectors at manipulation stages |
| `external` | Execute policy pan/tilt commands; no gaze teacher |

The collection teacher uses simulator instance segmentation, visible depth,
task object identities and execution stages. Learned policy observations
contain only RGB, robot/head state and the instruction. These are different
information sets; collecting demonstrations does not constitute policy evaluation.

Head limits, acceleration, speed, delay and settling are configured in
`active_view/config.yml`. They describe a kinematic camera mount, not a
collision-enabled physical neck. Optical zoom is absent. A fixed optical
center changes framing, and does not generally remove line-of-sight occlusion.

## HDF5 contract

RGB observations are captured at 25 Hz by default and at explicit capture
boundaries. Dense applied controls are recorded at 250 Hz. Timestamps and
`command_offset` are authoritative, including irregular final frames.

| Key | Contents |
| --- | --- |
| `frames/rgb/{head_camera,left_camera,right_camera}` | uint8 RGB, 240×320×3 |
| `frames/robot_state` | 14 values: measured arm joints and commanded gripper values |
| `frames/finger_qpos` | Measured physical finger joints |
| `frames/head_qpos`, `frames/head_velocity` | Simulated measured pan/tilt state |
| `frames/head_target`, `frames/head_settled` | Requested head target and settling flag |
| `frames/intrinsics/*`, `frames/extrinsics_cv/*` | Per-camera calibration |
| `frames/time_s`, `frames/step`, `frames/command_offset` | Observation time and dense-command boundary |
| `commands/action` | 16 targets: left6, left gripper, right6, right gripper, pan, tilt |
| `commands/arm_velocity_target` | 12 joint velocity feedforward targets |
| `commands/time_s`, `commands/step` | Dense control timestamps |

Units are radians, radians/second and seconds; grippers use [0,1]. For policy
training/deployment concatenate the 16 targets with 12 velocity targets into
28 channels. Keep the dense rate: taking every tenth target and holding it
changes the controller. `active_view.dataset.load_window` returns the exact
controls between observation timestamps.

```bash
python scripts/audit_active_view.py data/active_view
python scripts/replay_active_view.py PATH_TO_EPISODE --output eval_result/replay.json
python -m unittest discover -s tests -v
```

Replay is a controller/data check, not a learned-policy result. Old archives
may differ in asset enumeration or initial scene setup; matching seeds across
different source revisions does not guarantee the same scene.

## Evaluation

```bash
bash eval_av.sh adjust_bottle policy/hold.yml 0 --episodes 1 \
  --output eval_result/hold_check
bash eval_av.sh adjust_bottle policy/pi05/deploy.yml 0 --episodes 50 \
  --output eval_result/pi05_eval
bash eval_av.sh adjust_bottle policy/easywam/deploy.yml 0 --episodes 50 \
  --output eval_result/easywam_eval
```

Evaluation uses `external` mode and never calls the manipulation demonstration
or gaze teacher. Each requested seed is attempted once, without successful-seed
selection. Time limits are in simulated seconds. The policy server must
advertise the 28-D action, 16-D state and 250-Hz contract. Action chunks execute
one physics tick per row; the default replanning interval is ten ticks.
Inference pauses simulation: elapsed simulator time is not wall-clock latency.

The evaluator saves its config, per-seed logs/results and `summary.json`.
It reports valid episodes, errors, successes, valid-episode success rate and
successes divided by all attempts. Inspect error counts before comparing
results. Use a fresh output directory for every evaluation.

The supplied `hold` policy is a negative control. See the
[π0.5](../policy/pi05/README.md) and [EasyWAM](../policy/easywam/README.md)
guides for fine-tuning and starting their policy servers.
Native RoboTwin evaluation continues to use
`scripts/eval_policy.sh` and XPolicyLab. Its 14-D/TOPP-based action protocol
is separate from this AV dense-control protocol.
