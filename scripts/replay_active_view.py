#!/usr/bin/env python3
"""Replay exact dense controls using the external-policy API and compare states.

This checks controller/data semantics; it is NOT a trained-policy evaluation.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import h5py
import numpy as np
from collect_active_view import build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--output", default="eval_result/av_replay.json")
    args = parser.parse_args()
    with h5py.File(args.episode, "r") as file:
        metadata = json.loads(file.attrs["metadata_json"])
        controls = file["commands/action"][:]
        velocities = file["commands/arm_velocity_target"][:]
        frame_steps = file["frames/step"][:]
        reference_q = file["frames/robot_state"][:]
        reference_head = file["frames/head_qpos"][:]
    setup = SimpleNamespace(task=metadata["task"], seed=metadata["seed"], mode="external",
                            initial_head=metadata["config"]["initial_rad"], save_segmentation=False,
                            sample_steps=metadata["config"]["sample_steps"])
    # Apply the collector's seed initialization, then restore the exact saved
    # scene/controller configuration instead of today's potentially edited YAML.
    build_config(setup)
    config = copy.deepcopy(metadata["resolved_config"])
    config["active_view"]["mode"] = "external"
    task = getattr(importlib.import_module("envs." + setup.task), setup.task)()
    task.setup_demo(**config)
    # external mode never consults a task phase, visibility target, or action expert.
    max_arm_error, max_head_error = 0., 0.
    frame_index = 0
    try:
        for index in range(len(controls) + 1):
            if frame_index < len(frame_steps) and index == frame_steps[frame_index]:
                state = np.asarray(task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState())
                max_arm_error = max(max_arm_error, float(np.max(np.abs(state-reference_q[frame_index]))))
                max_head_error = max(max_head_error, float(np.max(np.abs(task.active_view.servo.q-reference_head[frame_index]))))
                frame_index += 1
            if index < len(controls):
                task.active_view.policy_step(controls[index], steps=1, arm_velocity=velocities[index], observe=False)
        result = {"episode": args.episode, "frames_compared": frame_index,
                  "commands_replayed": len(controls), "task_success": bool(task.check_success()),
                  "max_robot_state_abs_error": max_arm_error, "max_head_rad_error": max_head_error,
                  "head_threshold_rad": 1e-5, "robot_state_threshold": .02}
        result["pass"] = frame_index == len(frame_steps) and max_head_error < 1e-5 and max_arm_error < .02 and result["task_success"]
    finally:
        task.close_env()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
