"""Convert successful AV HDF5 episodes to LeRobot v3.0 for EasyWAM."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from active_view.dataset import audit_episode
from active_view.policy import CAMERAS, CONTROL_HZ, validate_actions
from scripts.convert_av_to_lerobot import causal_indices


def write_video(path, frames, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = frames.shape[1:]
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=CONTROL_HZ)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast'}
        previous, rgb = -1, None
        for tick, index in enumerate(indices):
            if index != previous:
                rgb, previous = frames[index], index
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = tick, Fraction(1, CONTROL_HZ)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('limit must be positive')
    sources, split = [], None
    for path in sorted(args.input.rglob('episode.hdf5')):
        with h5py.File(path) as f:
            if not f.attrs.get('complete') or not f.attrs.get('success'):
                continue
            meta = json.loads(f.attrs['metadata_json'])
        if not audit_episode(path)['all_checks_pass']:
            raise ValueError(f'Invalid episode: {path}')
        if not np.isclose(meta['physics_dt_s'], 1 / CONTROL_HZ):
            raise ValueError('Expected 250 Hz controls')
        if split is not None and split != meta['split']:
            raise ValueError('Convert each train/validation/test split separately')
        split = meta['split']
        sources.append((path.resolve(), meta))
        if args.limit and len(sources) >= args.limit:
            break
    if not sources:
        raise ValueError('No successful complete episodes')
    args.root.mkdir(parents=True, exist_ok=False)
    meta_dir = args.root / 'meta'
    (meta_dir / 'episodes/chunk-000').mkdir(parents=True)
    tasks = list(dict.fromkeys(meta['instruction'] for _, meta in sources))
    pq.write_table(pa.table({'task_index': range(len(tasks)), 'task': tasks}), meta_dir / 'tasks.parquet')
    features = {name: dict(dtype='int64', shape=[1], names=None)
                for name in ('index', 'episode_index', 'frame_index', 'task_index')}
    features['timestamp'] = dict(dtype='float32', shape=[1], names=None)
    features['observation.state'] = dict(dtype='float32', shape=[16], names=None)
    features['action'] = dict(dtype='float32', shape=[28], names=None)
    episodes, offset, moments = [], 0, {}
    for episode, (path, meta) in enumerate(sources):
        chunk, file_index = divmod(episode, 1000)
        with h5py.File(path) as f:
            count = len(f['commands/step'])
            indices = causal_indices(f['frames/step'][:], count)
            state = np.concatenate((f['frames/robot_state'][:], f['frames/head_qpos'][:]), axis=-1)[indices]
            action = validate_actions(np.concatenate((f['commands/action'][:], f['commands/arm_velocity_target'][:]), axis=-1))
            for key, values in (('observation.state', state), ('action', action)):
                values = values.astype(np.float64)
                mean, m2 = values.mean(0), ((values - values.mean(0)) ** 2).sum(0)
                if key not in moments:
                    moments[key] = dict(count=count, mean=mean, m2=m2, min=values.min(0), max=values.max(0))
                else:
                    current = moments[key]
                    total = current['count'] + count
                    delta = mean - current['mean']
                    current['m2'] += m2 + delta ** 2 * current['count'] * count / total
                    current['mean'] += delta * count / total
                    current['count'] = total
                    current['min'] = np.minimum(current['min'], values.min(0))
                    current['max'] = np.maximum(current['max'], values.max(0))
            table = pa.table({
                'index': np.arange(offset, offset + count, dtype=np.int64),
                'episode_index': np.full(count, episode, dtype=np.int64),
                'frame_index': np.arange(count, dtype=np.int64),
                'task_index': np.full(count, tasks.index(meta['instruction']), dtype=np.int64),
                'timestamp': np.arange(count, dtype=np.float32) / CONTROL_HZ,
                'observation.state': pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32(), 16)),
                'action': pa.array(action.tolist(), type=pa.list_(pa.float32(), 28)),
            })
            data_path = args.root / f'data/chunk-{chunk:03d}/file-{file_index:03d}.parquet'
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, data_path)
            row = {'episode_index': episode, 'length': count, 'tasks': [meta['instruction']],
                   'data/chunk_index': chunk, 'data/file_index': file_index,
                   'dataset_from_index': offset, 'dataset_to_index': offset + count}
            for camera in CAMERAS:
                key = 'observation.images.' + camera
                frames = f['frames/rgb/' + camera]
                shape = list(frames.shape[1:])
                feature = dict(dtype='video', shape=shape, names=['height', 'width', 'channels'],
                               info={'video.height': shape[0], 'video.width': shape[1],
                                     'video.fps': CONTROL_HZ, 'video.channels': 3,
                                     'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                                     'video.is_depth_map': False, 'has_audio': False})
                if key in features and features[key] != feature:
                    raise ValueError(f'Inconsistent camera dimensions: {camera}')
                features[key] = feature
                write_video(args.root / f'videos/{key}/chunk-{chunk:03d}/file-{file_index:03d}.mp4', frames, indices)
                row.update({f'videos/{key}/chunk_index': chunk, f'videos/{key}/file_index': file_index,
                            f'videos/{key}/from_timestamp': 0., f'videos/{key}/to_timestamp': count / CONTROL_HZ})
            episodes.append(row)
            offset += count
        print(f'{episode + 1}/{len(sources)}: {path}', flush=True)
    pq.write_table(pa.Table.from_pylist(episodes), meta_dir / 'episodes/chunk-000/file-000.parquet')
    info = dict(codebase_version='v3.0', robot_type='aloha_agilex_av', fps=CONTROL_HZ,
                total_episodes=len(episodes), total_frames=offset, total_tasks=len(tasks),
                total_videos=len(episodes) * 3, chunks_size=1000,
                splits={split: f'0:{len(episodes)}'}, features=features,
                data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (meta_dir / 'info.json').write_text(json.dumps(info, indent=2))
    stats = {key: dict(min=value['min'].tolist(), max=value['max'].tolist(),
                       mean=value['mean'].tolist(), std=np.sqrt(value['m2'] / value['count']).tolist(),
                       count=[value['count']]) for key, value in moments.items()}
    (meta_dir / 'stats.json').write_text(json.dumps(stats, indent=2))
    (args.root / 'av_contract.json').write_text(json.dumps(dict(
        action_type='dense_joint_position_velocity', action_dim=28, state_dim=16,
        control_hz=CONTROL_HZ, split=split, observation_rule='latest frame at or before command',
        sources=[str(path) for path, _ in sources]), indent=2))


if __name__ == '__main__':
    main()
