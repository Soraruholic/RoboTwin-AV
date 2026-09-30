"""Streaming exact-time observations and applied dense commands, no relabeling."""
import json
from pathlib import Path
import h5py
import numpy as np
from . import SCHEMA_VERSION


class Recorder:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs["schema_version"] = SCHEMA_VERSION
        self.file.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        self.file.attrs["complete"] = False
        self.counts = {"frames": 0, "commands": 0}

    def append(self, group, values):
        index = self.counts[group]
        for key, value in values.items():
            name = f"{group}/{key}"
            if isinstance(value, str):
                if name not in self.file:
                    self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
                dataset = self.file[name]
            else:
                value = np.asarray(value)
                if name not in self.file:
                    kwargs = {"compression": "lzf"} if value.ndim >= 2 else {}
                    self.file.create_dataset(name, shape=(0,) + value.shape,
                                             maxshape=(None,) + value.shape,
                                             chunks=(1,) + value.shape, dtype=value.dtype, **kwargs)
                dataset = self.file[name]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self.counts[group] += 1

    def finish(self, success, reason, info):
        self.file.attrs["success"] = bool(success)
        self.file.attrs["termination_reason"] = reason
        self.file.attrs["episode_info_json"] = json.dumps(info, default=str, ensure_ascii=False)
        self.file.attrs["complete"] = True
        self.file.flush()
        self.file.close()
