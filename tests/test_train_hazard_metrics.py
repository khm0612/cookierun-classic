import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_train_hazard(tmp_path, monkeypatch):
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "model_meta.json").write_text("{}", encoding="utf-8")
    (demo / "iql3_meta.json").write_text(
        json.dumps({"K": 1, "H": 8, "W": 8, "crop": [0, 0, 1, 1], "conv": [[2, 3, 1]]}),
        encoding="utf-8",
    )
    runtime = types.ModuleType("_runtime")
    runtime.DATA = tmp_path
    runtime.recording_is_complete = lambda metadata: bool(metadata.get("frames"))
    monkeypatch.setitem(sys.modules, "_runtime", runtime)
    monkeypatch.setattr(sys, "argv", ["train_hazard.py"])
    path = Path(__file__).parents[1] / "scripts" / "train_hazard.py"
    spec = importlib.util.spec_from_file_location("_test_train_hazard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_per_pit_recall_counts_overlapping_approach_windows_separately(tmp_path, monkeypatch):
    module = _load_train_hazard(tmp_path, monkeypatch)
    ts = np.arange(8, dtype=np.float64) * 0.5
    pits = np.array([4, 5])  # 0.5s apart: their 1.5s label windows overlap
    fire = np.array([False, True, False, False, False, False, False, False])

    assert module.pit_detection_counts(fire, ts, pits, 1.5) == (1, 2)


def test_hazard_split_keeps_positive_runs_on_both_sides(tmp_path, monkeypatch):
    module = _load_train_hazard(tmp_path, monkeypatch)
    positive = np.ones(1, dtype=np.float32)
    runs = [(f"r{i}", None, positive, None, None) for i in range(3)]

    names = module.validation_run_names(runs, 0.9, np.random.default_rng(0))

    assert len(names) == 2
    assert {run[0] for run in runs} - names
    with pytest.raises(SystemExit, match="at least 3"):
        module.validation_run_names(runs[:2], 0.25, np.random.default_rng(0))
