"""Export dense controls to LeRobot 2.1 with causal, held observations."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from active_view.dataset import audit_episode
from active_view.policy import CAMERAS, CONTROL_HZ


def causal_indices(frame_steps, command_count):
    steps = np.asarray(frame_steps)
    if steps.ndim != 1 or len(steps) < 2 or steps[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError('Frame steps must start at zero and increase strictly')
    if steps[-1] != command_count:
        raise ValueError('Missing final observation or incomplete controls')
    return np.searchsorted(steps, np.arange(command_count), side='right') - 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--repo-id', default='local/robotwin_av')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.root.exists():
        raise FileExistsError(args.root)
    if args.limit is not None and args.limit < 1:
        p.error('limit must be positive')
    files = sorted(args.input.rglob('episode.hdf5'))
    if not files:
        raise ValueError('No episodes found')
    sources = []
    split = None
    for path in files:
        with h5py.File(path) as file:
            if not file.attrs.get('complete') or not file.attrs.get('success'):
                continue
            meta = json.loads(file.attrs['metadata_json'])
        if not audit_episode(path)['all_checks_pass']:
            raise ValueError(f'Invalid episode: {path}')
        if not np.isclose(meta['physics_dt_s'], 1 / CONTROL_HZ):
            raise ValueError('Expected 250 Hz dense controls')
        if split is not None and split != meta['split']:
            raise ValueError('Do not mix train, validation, and test episodes')
        split = meta['split']
        sources.append((path, meta))
        if args.limit and len(sources) >= args.limit:
            break
    if not sources:
        raise ValueError('No successful complete episodes')
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.video_utils import encode_video_frames
    # SVT-AV1 rejects the native 250-Hz timebase; H.264 supports it.
    lerobot_dataset.encode_video_frames = partial(encode_video_frames, vcodec='h264')
    LeRobotDataset = lerobot_dataset.LeRobotDataset
    with h5py.File(sources[0][0]) as file:
        features = {'observation.state': dict(dtype='float32', shape=(16,), names=['state']),
                    'action': dict(dtype='float32', shape=(28,), names=['action'])}
        for name in CAMERAS:
            features['observation.images.' + name] = dict(
                dtype='video', shape=file['frames/rgb/' + name].shape[1:], names=['height', 'width', 'channels'])
    dataset = LeRobotDataset.create(repo_id=args.repo_id, root=args.root, fps=CONTROL_HZ,
                                    robot_type='aloha_agilex_av', features=features, use_videos=True)
    for path, meta in sources:
        with h5py.File(path) as file:
            indices = causal_indices(file['frames/step'][:], len(file['commands/step']))
            last = -1
            for command, index in enumerate(indices):
                if index != last:
                    held = {'observation.state': np.r_[file['frames/robot_state'][index],
                                                       file['frames/head_qpos'][index]].astype(np.float32)}
                    for name in CAMERAS:
                        held['observation.images.' + name] = file['frames/rgb/' + name][index]
                    last = index
                action = np.r_[file['commands/action'][command],
                               file['commands/arm_velocity_target'][command]].astype(np.float32)
                dataset.add_frame(dict(held, action=action, task=meta['instruction']))
        dataset.save_episode()
    (args.root / 'av_contract.json').write_text(json.dumps(dict(
        action_type='dense_joint_position_velocity', action_dim=28, state_dim=16,
        control_hz=CONTROL_HZ, observation_rule='latest frame at or before command',
        split=split, sources=[str(p) for p, _ in sources]), indent=2))


if __name__ == '__main__':
    main()
