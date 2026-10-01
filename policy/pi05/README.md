# π0.5 with RoboTwin-AV

This adapter uses upstream OpenPI and all three RGB cameras. State has 16
channels; output has 28 channels including arm velocity feedforward and head
targets. The model retains 32 channels internally for base-weight compatibility
and discards only the four padding channels at output.
Normalized robot/head state enters the π0.5 discrete state tokens
(discrete_state_input=True); it is not dropped from model conditioning.

Use OpenPI revision `215abfb217dbac7d5f1273282331b9b1866c0479` and its pinned
LeRobot dependency (`0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`, v2.1 API).
Install its environment following [OpenPI](https://github.com/Physical-Intelligence/openpi).
Do not install the JAX/model dependencies into the RoboTwin simulator environment.

## Convert

Run in the OpenPI environment, with `h5py` available:

```bash
python scripts/convert_av_to_lerobot.py \
  --input data/active_view/train \
  --repo-id local/robotwin_av \
  --root /path/to/lerobot-data/local/robotwin_av
```

The converter accepts only successful, audited episodes and rejects mixed
splits. Each row corresponds to one 250-Hz control tick. RGB and state hold
the latest observation at or before that tick, so there is no future-frame
leakage. Repeated images do not represent additional observations or episodes.
H.264 stores the 250-Hz video timebase (SVT-AV1 rejects this rate).
This format preserves all dense controls but has more rows and video frames
than the source 25-Hz RGB stream. Start with `--limit 1` before bulk conversion.

## Fine-tune

From the RoboTwin-AV directory using OpenPI's Python environment:

```bash
export HF_LEROBOT_HOME=/path/to/lerobot-data
export AV_REPO_ID=local/robotwin_av
python policy/pi05/run.py norm --openpi-root /path/to/openpi
python policy/pi05/run.py train --openpi-root /path/to/openpi \
  --exp-name robotwin_av --batch-size 32
```

`AV_BASE_PARAMS` optionally replaces the default π0.5 base parameter URI.
The starter configuration uses 50 dense actions (0.2 simulated seconds),
absolute joint/head targets, 32 batch size, and 30,000 updates. It does not
apply ALOHA real-hardware sign/gripper conversion or a 14-D output slice.
Use the normalization statistics saved by training when serving the checkpoint.

## Serve and evaluate

Start the model server in the same OpenPI environment and use the same repo ID:

```bash
export AV_REPO_ID=local/robotwin_av
python policy/pi05/serve.py --checkpoint /path/to/checkpoint --port 8000
```

In the RoboTwin simulation environment, install only the client from that
OpenPI checkout, then run:

```bash
pip install -e /path/to/openpi/packages/openpi-client
bash eval_av.sh adjust_bottle policy/pi05/deploy.yml 0 \
  --episodes 50 --output eval_result/pi05_adjust_bottle
```

Set `host` and `port` in `deploy.yml` to match the server. A native 14-D
RoboTwin/ALOHA checkpoint does not supply trained head or velocity channels.
