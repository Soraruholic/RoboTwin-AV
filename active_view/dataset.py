"""Read a real observation window and EXACT controls between its timestamps.

    Do not reinterpret 250 Hz dense controls as held 25 Hz control targets.
    A model can output the dense substeps as a chunk between image observations.
"""
import json
from pathlib import Path
import h5py
import numpy as np


def load_window(path, start, future_frames):
    with h5py.File(path, "r") as file:
        if not file.attrs.get("complete") or not file.attrs.get("success"):
            raise ValueError("BC loader requires complete successful episodes")
        if start < 0 or future_frames < 1 or start + future_frames >= len(file["frames/step"]):
            raise IndexError("window would cross the episode boundary")
        end = start + future_frames
        offsets = file["frames/command_offset"][start:end+1]
        lo, hi = int(offsets[0]), int(offsets[-1])
        cameras = json.loads(file.attrs["metadata_json"])["camera_order"]
        return {
            "rgb": {name: file[f"frames/rgb/{name}"][start:end+1] for name in cameras},
            "robot_state": file["frames/robot_state"][start],
            "head_qpos": file["frames/head_qpos"][start],
            "observation_times_s": file["frames/time_s"][start:end+1],
            "action": file["commands/action"][lo:hi],
            "action_times_s": file["commands/time_s"][lo:hi],
            "arm_velocity_target": file["commands/arm_velocity_target"][lo:hi],
            "frame_action_offsets": offsets - lo,
            "metadata": json.loads(file.attrs["metadata_json"]),
        }


def audit_episode(path):
    with h5py.File(path, "r") as file:
        meta = json.loads(file.attrs["metadata_json"])
        config = meta["config"]
        dt = meta["physics_dt_s"]
        step = file["frames/step"][:]
        command_step = file["commands/step"][:]
        q = file["frames/head_qpos"][:]
        velocity = file["frames/head_velocity"][:]
        time = file["frames/time_s"][:]
        action = file["commands/action"][:]
        limits = np.asarray(config["limits_rad"])
        checks = {
            "complete": bool(file.attrs.get("complete")),
            "positive_frame_count": len(step) > 1,
            "strict_frame_time": bool(np.all(np.diff(step) > 0)),
            "dense_command_index": bool(np.array_equal(command_step, np.arange(len(command_step)))),
            "exact_frame_time": bool(np.allclose(time, step * dt, atol=1e-10)),
            "exact_command_time": bool(np.allclose(file["commands/time_s"][:], command_step * dt, atol=1e-10)),
            "frame_command_alignment": bool(np.array_equal(step, file["frames/command_offset"][:])),
            "full_transition_coverage": int(step[0]) == 0 and int(step[-1]) == len(command_step),
            "finite_action": bool(np.isfinite(action).all()),
            "action_16d": action.shape[1:] == (16,),
            "arm_feedforward_12d": file["commands/arm_velocity_target"].shape == (len(command_step), 12),
            "head_limits": bool(np.all(q >= limits[:, 0] - 1e-6) and np.all(q <= limits[:, 1] + 1e-6)),
            "head_speed": bool(np.max(np.abs(velocity)) <= config["max_speed_rad_s"] + 1e-6),
            "head_acceleration": bool(np.all(np.abs(np.diff(velocity, axis=0)) <=
                                               config["max_accel_rad_s2"] * np.diff(time)[:, None] + 1e-6)),
            "native_three_cameras": set(meta["camera_order"]) == {"head_camera", "left_camera", "right_camera"},
        }
        def check_length(name, obj):
            if isinstance(obj, h5py.Dataset):
                if name.startswith("frames/"):
                    checks[f"length:{name}"] = len(obj) == len(step)
                elif name.startswith("commands/"):
                    checks[f"length:{name}"] = len(obj) == len(command_step)
        file.visititems(check_length)
        for camera in meta["camera_order"]:
            rgb = file[f"frames/rgb/{camera}"]
            checks[f"rgb:{camera}"] = rgb.dtype == np.uint8 and rgb.shape[1:] == (240, 320, 3)
            checks[f"image_nonempty:{camera}"] = float(rgb[0].std()) > 1
            intrinsic = file[f"frames/intrinsics/{camera}"][:]
            checks[f"fixed_intrinsics:{camera}"] = bool(np.allclose(intrinsic, intrinsic[0]))
        if meta["config"]["mode"] == "fixed":
            checks["fixed_head_stationary"] = bool(np.allclose(q, q[0]))
        return {"file": str(Path(path)), "task": meta["task"], "mode": config["mode"],
                "success": bool(file.attrs["success"]), "all_checks_pass": all(checks.values()),
                "checks": checks, "frames": len(step), "commands": len(command_step),
                "duration_s": float(time[-1]), "head_range_rad": np.ptp(q, axis=0).tolist(),
                "head_settled_fraction": float(np.mean(file["frames/head_settled"][:])),
                "head_path_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
                "bytes": Path(path).stat().st_size}
