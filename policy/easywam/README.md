# EasyWAM with RoboTwin-AV

This adapter uses [OpenMOSS/EasyWAM](https://github.com/OpenMOSS/EasyWAM),
revision `a8ec082360e101d22601af4fdf02a82eab506819`. It runs the upstream
EasyWAM-Unified Wan2.2 model, LoRA implementation, LeRobot v3.0 loader, and
training loop. The adapter sets the AV interface to 16 state channels and
28 action channels, including pan/tilt and arm velocity targets.

## Install

Use a separate environment from the RoboTwin simulator. Follow upstream
EasyWAM's installation and Wan2.2 weight download instructions, then install
the websocket client:

```bash
git clone https://github.com/OpenMOSS/EasyWAM.git /path/to/EasyWAM
git -C /path/to/EasyWAM checkout a8ec082360e101d22601af4fdf02a82eab506819
pip install -e /path/to/EasyWAM
pip install openpi-client==0.1.2 'websockets>=14,<16'
```

Run the following commands from the RoboTwin-AV directory using that environment.

## Convert data

```bash
python policy/easywam/convert.py \
  --input data/active_view/train --root /path/to/robotwin-av-v3
```

The converter writes successful, audited episodes as LeRobot v3.0 Parquet
and H.264 videos. Each row retains one 250-Hz control command. State and RGB
hold the latest observation at or before that command. This preserves all
dense controls while matching the upstream reader. Convert splits separately.
Use `--limit 1` to check one episode before converting the full collection.

## Fine-tune

```bash
export EASYWAM_ROOT=/path/to/EasyWAM
export WAN_ROOT=/path/to/Wan2.2-TI2V-5B
export AV_DATA=/path/to/robotwin-av-v3
export AV_RUN=/path/to/runs/easywam-av

python policy/easywam/run.py text \
  --easywam-root "$EASYWAM_ROOT" --data "$AV_DATA" \
  --backbone "$WAN_ROOT" --output "$AV_RUN"

python policy/easywam/run.py train \
  --easywam-root "$EASYWAM_ROOT" --data "$AV_DATA" \
  --backbone "$WAN_ROOT" --output "$AV_RUN" \
  --base-checkpoint /path/to/easywam_unified_wan22.pt
```

`--base-checkpoint` loads the video backbone from the upstream pretrained
EasyWAM-Unified checkpoint. State/action projections are initialized for the
AV control space and trained with the LoRA parameters. Without this option,
the model initializes from Wan2.2. The default `context` state placement
matches the original EasyWAM-Unified checkpoint.

[finetune.yaml](finetune.yaml) supplies the defaults: rank-16 LoRA,
80 dense actions, 9 video frames, three cameras composed as a 384×320 image,
and 30,000 updates. Head is on top; left/right wrists are below. Training
computes normalization from the training episodes and saves it in the run
directory alongside `config.yaml` and checkpoints. Five percent of episodes
are held out by the upstream loader.

Append Hydra-style overrides, for example `max_steps=1000 batch_size=2`.
For a single-episode smoke run, also set
`data.train.val_set_proportion=0 data.val.val_set_proportion=0`.
Resume with `resume=/path/to/run/checkpoints/state/step_...`.

## Serve and evaluate

```bash
python policy/easywam/serve.py \
  --easywam-root "$EASYWAM_ROOT" --run-dir "$AV_RUN" \
  --checkpoint "$AV_RUN/checkpoints/weights/step_030000.pt" --port 8001
```

The run directory supplies model settings and normalization statistics.
If weights have moved, pass `--backbone` and `--base-checkpoint` with their
new paths. `--inference-steps` controls the number of denoising steps.

In the RoboTwin simulator environment:

```bash
pip install openpi-client==0.1.2
bash eval_av.sh adjust_bottle policy/easywam/deploy.yml 0 \
  --episodes 50 --output eval_result/easywam_adjust_bottle
```

Set `host` and `port` in [deploy.yml](deploy.yml) to match the server.
`execute_steps` sets how many 250-Hz actions execute before the next observation;
it must be a positive multiple of 10 and no larger than the trained horizon.
