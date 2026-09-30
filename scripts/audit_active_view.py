#!/usr/bin/env python3
"""Audit episode timing, controls and camera data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from active_view.dataset import audit_episode, load_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", default="eval_result/av_audit")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted(Path(args.directory).rglob("episode.hdf5")):
        try:
            report = audit_episode(path)
            if report["success"]:
                window = load_window(path, 0, min(4, report["frames"] - 1))
                report["window_read_pass"] = len(window["action"]) == window["frame_action_offsets"][-1]
            reports.append(report)
        except Exception as exc:
            reports.append({"file": str(path), "all_checks_pass": False, "error": repr(exc)})
    (output / "audit.json").write_text(json.dumps(reports, indent=2, default=lambda value: value.item()
                                                      if isinstance(value, np.generic) else str(value)))
    passed = bool(reports) and all(row["all_checks_pass"] and row.get("window_read_pass", True) for row in reports)
    print(json.dumps({"episodes": len(reports), "audit_pass": passed,
                      "task_successes": sum(row.get("success", False) for row in reports),
                      "frames": sum(row.get("frames", 0) for row in reports)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
