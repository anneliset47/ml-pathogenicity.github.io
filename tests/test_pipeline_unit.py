from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE_PATH = ROOT / "run_pipeline.py"
spec = importlib.util.spec_from_file_location("run_pipeline", RUN_PIPELINE_PATH)
assert spec and spec.loader
run_pipeline = importlib.util.module_from_spec(spec)
sys.modules["run_pipeline"] = run_pipeline
spec.loader.exec_module(run_pipeline)


def test_resolve_selected_steps_default_core_only() -> None:
    args = argparse.Namespace(step=None, all=False, with_nn=False)

    names = [step["name"] for step in run_pipeline.resolve_selected_steps(args)]

    assert names == ["pca", "clustering", "decision_tree", "naive_bayes", "svm"]


def test_resolve_selected_steps_adds_nn_when_requested() -> None:
    args = argparse.Namespace(step=["svm"], all=False, with_nn=True)

    names = [step["name"] for step in run_pipeline.resolve_selected_steps(args)]

    assert names == ["svm", "nn"]


def test_validate_inputs_raises_for_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"
    step = {"name": "unit-test-step", "requires": [missing_file]}

    with pytest.raises(FileNotFoundError) as exc_info:
        run_pipeline.validate_inputs(step)

    message = str(exc_info.value)
    assert "unit-test-step" in message
    assert str(missing_file) in message


def test_validate_inputs_passes_when_files_exist(tmp_path: Path) -> None:
    existing_file = tmp_path / "present.csv"
    existing_file.write_text("col\n1\n", encoding="utf-8")

    step = {"name": "unit-test-step", "requires": [existing_file]}

    run_pipeline.validate_inputs(step)
