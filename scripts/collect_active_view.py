#!/usr/bin/env python3
"""Bounded, restartable one-process-per-episode AV collection; run from repo root."""
import argparse
import importlib
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_view.task_recipes import TASK_ROLES, configure_task

TASK_OBJECTS = {name: subjects + goals for name, (subjects, goals) in TASK_ROLES.items()}


def source_revision():
    if not (ROOT / '.git').exists():
        return {'git_revision': None, 'git_dirty': None}
    return {'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
            'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'],text=True,cwd=ROOT).strip())}


def build_config(args):
    import yaml
    os.chdir(ROOT)
    seed = args.seed
    random.seed(seed)
    config_name = getattr(args, "task_config", "demo_clean")
    config = yaml.safe_load((ROOT / "env_cfg/task_config" / f"{config_name}.yml").read_text())
    av = yaml.safe_load((ROOT / "active_view/config.yml").read_text())
    av.update(config.get("active_view", {}))
    av.update(mode=args.mode, initial_rad=args.initial_head, save_segmentation=args.save_segmentation)
    if args.sample_steps is not None:
        av["sample_steps"] = args.sample_steps
    config.update(task_name=args.task, task_config=config_name, now_ep_num=seed, seed=seed,
                  need_plan=True, save_data=False, collect_data=False, render_freq=0,
                  save_freq=None, active_view=av)
    robot_dir = ROOT / "assets/embodiments/aloha-agilex"
    robot_config = yaml.safe_load((robot_dir / "config.yml").read_text())
    config.update(left_robot_file=str(robot_dir), right_robot_file=str(robot_dir),
                  left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                  dual_arm_embodied=True, embodiment_name="aloha-agilex")
    return config


def episode(args):
    from active_view.recorder import Recorder
    from active_view.runtime import ObservationTimeout
    config = build_config(args)
    seed = args.seed
    task = getattr(importlib.import_module("envs." + args.task), args.task)()
    out = Path(args.output).resolve() / args.split / args.task / args.mode / f"seed_{seed:06d}"
    for other in {"train", "validation", "test"} - {args.split}:
        if list((Path(args.output).resolve() / other / args.task).glob(f"*/seed_{seed:06d}")):
            raise ValueError(f"episode family already exists in split={other}")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "episode.hdf5").exists() or (out / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    recorder = None
    result = {"task": args.task, "mode": args.mode, "seed": seed, "split": args.split,
              "initial_head": args.initial_head, "success": False, "reason": "setup_error"}
    settle_retries = getattr(args, "settle_retries", 0)
    result["settle_retries"] = settle_retries
    settling_checks = []
    if settle_retries:
        original_check = task.check_stable

        def checked_settling():
            # Repeat the exact upstream stability check while allowing more
            # physical settling time. No object is frozen or exempted.
            for _ in range(settle_retries + 1):
                stable, names = original_check()
                settling_checks.append({"stable": bool(stable), "unstable_objects": names})
                if stable:
                    break
            return stable, names
        task.check_stable = checked_settling
    started = time.time()
    try:
        task.setup_demo(**config)
        recipe = configure_task(task, args.task)
        metadata = task.active_view.metadata()
        sources = [*sorted((ROOT / "active_view").glob("*.py")), ROOT / "active_view/config.yml",
                   ROOT / "envs/_base_task.py", ROOT / f"envs/{args.task}.py", Path(__file__).resolve()]
        metadata.update(task=args.task, seed=seed, split=args.split,
                        instruction=args.task.replace("_", " "),
                        **source_revision(),
                        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                       for path in sources},
                        resolved_config=config,
                        settling_checks=settling_checks,
                        settle_retries=settle_retries,
                        parent_episode_id=f"{args.task}:{seed}")
        recorder = Recorder(out / "episode.hdf5", metadata)
        task.active_view.start(recorder)
        if recipe is not None:
            recipe.begin()
        task.play_once()
        # Record a final state even when it does not fall on a sampling boundary.
        task.active_view.capture()
        legal, excess = task.planned_joints_legal()
        passed = bool(task.plan_success and task.check_success() and legal)
        result.update(success=passed, planned_joints_legal=bool(legal), joint_excess_rad=float(excess),
                      plan_success=bool(task.plan_success),
                      reason="success" if passed else ("joint_limit" if not legal else "expert_failure"))
    except ObservationTimeout as exc:
        result.update(reason="observation_timeout", error=str(exc))
    except Exception:
        result.update(reason="runtime_error", error=traceback.format_exc())
    finally:
        if settling_checks:
            result["settling_checks"] = settling_checks
        if recorder:
            result["counts"] = recorder.counts.copy()
            recorder.finish(result["success"], result["reason"], getattr(task, "info", {}))
        result["wall_time_s"] = time.time() - started
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if hasattr(task, "scene"):
            task.close_env()
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["success"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_OBJECTS, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--mode", choices=["fixed", "tracking", "phase"], default="phase")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output", default="data/active_view")
    parser.add_argument("--initial-head", type=float, nargs=2, default=[0.0, 0.0], metavar=("PAN", "TILT"))
    parser.add_argument("--sample-steps", type=int)
    parser.add_argument("--save-segmentation", action="store_true")
    parser.add_argument("--settle-retries", type=int, default=0,
                        help="additional passes of the unchanged upstream scene stability check")
    return episode(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
