"""Collect a bounded quota of audited successful demonstrations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from active_view.dataset import audit_episode
from active_view.task_recipes import TASK_ROLES


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=sorted(TASK_ROLES))
    p.add_argument('--task-config', default='demo_clean')
    p.add_argument('--mode', choices=['phase', 'tracking', 'fixed'], default='phase')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--max-attempts', type=int)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='train')
    p.add_argument('--output', type=Path, default=Path('data/active_view'))
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--min-free-gib', type=float, default=20)
    p.add_argument('--initial-head', nargs=2, type=float, default=[0., 0.])
    p.add_argument('--settle-retries', type=int, default=0)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.episodes < 1 or args.timeout <= 0 or args.min_free_gib < 0 or args.settle_retries < 0:
        p.error('episodes/timeout must be positive; reserve/retries must be nonnegative')
    limit = args.max_attempts if args.max_attempts is not None else 5 * args.episodes
    if limit < args.episodes:
        p.error('max-attempts must be at least episodes')
    args.output = args.output.resolve()
    config = vars(args).copy()
    config['output'] = str(args.output)
    config.pop('dry_run')
    sources = [*sorted((ROOT / 'active_view').glob('*.py')), ROOT / 'active_view/config.yml',
               ROOT / 'envs/_base_task.py', ROOT / f'envs/{args.task}.py',
               ROOT / f'env_cfg/task_config/{args.task_config}.yml', ROOT / 'scripts/collect_active_view.py']
    config['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    run = args.output / args.split / args.task / args.mode
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    run.mkdir(parents=True, exist_ok=True)
    lock = run / '.collection.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
        config_path = run / 'collection_config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing collection has different settings; use another output directory')
        write_json(config_path, config)
        attempts, accepted = [], []
        for seed in range(args.seed, args.seed + limit):
            if len(accepted) >= args.episodes:
                break
            episode = run / f'seed_{seed:06d}'
            receipt = episode / 'result.json'
            if not receipt.exists():
                if episode.exists():
                    raise RuntimeError(f'Incomplete attempt needs inspection: {episode}')
                if shutil.disk_usage(run).free < args.min_free_gib * 1024**3:
                    raise RuntimeError('Free space is below the configured reserve')
                cmd = [sys.executable, str(ROOT / 'scripts/collect_active_view.py'),
                       '--task', args.task, '--task-config', args.task_config,
                       '--mode', args.mode, '--seed', str(seed), '--split', args.split,
                       '--output', str(args.output), '--initial-head', *map(str, args.initial_head),
                       '--settle-retries', str(args.settle_retries)]
                try:
                    with (run / f'seed_{seed:06d}.log').open('w') as log:
                        result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                timeout=args.timeout, check=False)
                    failure = 'process_failure'
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    failure, code = 'timeout', -1
                if not receipt.exists():
                    episode.mkdir(exist_ok=True)
                    write_json(receipt, dict(task=args.task, seed=seed, mode=args.mode,
                                             success=False, reason=failure, returncode=code))
            row = json.loads(receipt.read_text())
            if row.get('success'):
                audit = audit_episode(episode / 'episode.hdf5')
                write_json(episode / 'audit.json', audit)
                if audit['all_checks_pass'] and audit['success']:
                    accepted.append(str((episode / 'episode.hdf5').relative_to(args.output)))
                else:
                    row = dict(row, success=False, reason='audit_failed')
            attempts.append(row)
            write_json(run / 'manifest.json', dict(root=str(args.output), episodes=accepted,
                       target=args.episodes, attempts=attempts, complete=len(accepted) == args.episodes))
            print(f'{args.task}: {len(accepted)}/{args.episodes} accepted, {len(attempts)} attempted', flush=True)
        return 0 if len(accepted) == args.episodes else 2
    finally:
        os.close(fd)
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
