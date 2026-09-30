"""Evaluate an AV policy in a separate process for each seed."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from active_view.policy import CONTROL_HZ, execute_chunk, observation, validate_contract
from active_view.task_recipes import TASK_ROLES
from collect_active_view import build_config
from collect_av import write_json


def episode(args, cfg):
    out = args.output / f'seed_{args.seed:06d}'
    out.mkdir(parents=True, exist_ok=False)
    row = dict(task=args.task, seed=args.seed, success=False, status='error',
               policy=cfg['policy'], physics_steps=0)
    task = None
    try:
        validate_contract(cfg)
        setup = SimpleNamespace(task=args.task, task_config=cfg['task_config'], seed=args.seed,
                                mode='external', initial_head=[0., 0.], save_segmentation=False,
                                sample_steps=10)
        task = getattr(importlib.import_module('envs.' + args.task), args.task)()
        task.setup_demo(**build_config(setup))
        av = task.active_view
        if not np.isclose(av.dt, 1 / CONTROL_HZ):
            raise ValueError('Unexpected simulator control frequency')
        if cfg['policy'] == 'openpi':
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy(host=cfg['host'], port=cfg['port'])
            validate_contract(client.get_server_metadata())
            client.reset()
        elif cfg['policy'] != 'hold':
            raise ValueError('Unsupported policy')
        max_steps = round(float(cfg['episode_seconds']) / av.dt)
        count = int(cfg['execute_steps'])
        if max_steps < 1 or count < 1 or count % av.sample_steps:
            raise ValueError('Use a positive duration and execute_steps divisible by sample_steps')
        row['initial_success'] = bool(task.check_success())
        row['initial_head_qpos'] = av.servo.q.tolist()
        row['policy_calls'] = 0
        if row['initial_success']:
            raise RuntimeError('Initial scene already satisfies the goal')
        while av.step < max_steps:
            obs = observation(av.capture(), args.instruction or args.task.replace('_', ' '))
            if cfg['policy'] == 'hold':
                actions = np.tile(np.r_[obs['state'], np.zeros(12)], (count, 1))
            else:
                actions = client.infer(obs)['actions']
            row['policy_calls'] += 1
            if execute_chunk(task, actions, count, max_steps):
                row['success'] = True
                break
        row.update(status='success' if row['success'] else 'time_limit', physics_steps=av.step,
                   physics_seconds=av.step * av.dt, final_head_qpos=av.servo.q.tolist())
    except Exception as exc:
        import traceback
        row.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        if task is not None and hasattr(task, 'scene'):
            task.close_env()
        write_json(out / 'result.json', row)
    return 0 if row['status'] != 'error' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=sorted(TASK_ROLES))
    p.add_argument('--policy-config', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=100000)
    p.add_argument('--output', type=Path, default=Path('eval_result/active_view'))
    p.add_argument('--instruction')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.policy_config = args.policy_config.resolve()
    cfg = yaml.safe_load(args.policy_config.read_text())
    validate_contract(cfg)
    if args.episodes < 1:
        p.error('episodes must be positive')
    if args.worker:
        return episode(args, cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', dict(policy=cfg, task=args.task, seed=args.seed,
                                                episodes=args.episodes, instruction=args.instruction))
    rows = []
    for seed in range(args.seed, args.seed + args.episodes):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--task', args.task,
                   '--policy-config', str(args.policy_config), '--seed', str(seed),
                   '--output', str(args.output)]
        if args.instruction:
            command += ['--instruction', args.instruction]
        try:
            with (args.output / f'seed_{seed:06d}.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               timeout=cfg.get('timeout_seconds', 600), check=False)
        except subprocess.TimeoutExpired:
            pass
        result_path = args.output / f'seed_{seed:06d}' / 'result.json'
        row = (json.loads(result_path.read_text()) if result_path.exists() else
               dict(seed=seed, success=False, status='error', error='worker failed or timed out'))
        rows.append(row)
        valid = [r for r in rows if r['status'] != 'error']
        success = sum(r['success'] for r in valid)
        write_json(args.output / 'summary.json', dict(requested=args.episodes, attempted=len(rows),
                   valid=len(valid), errors=len(rows)-len(valid), successes=success,
                   success_rate=success/len(valid) if valid else None,
                   success_rate_all_attempts=success/len(rows), episodes=rows))
        print(f'{len(rows)}/{args.episodes}: {success} successes, {len(rows)-len(valid)} errors', flush=True)
    return 2 if len(valid) != len(rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
