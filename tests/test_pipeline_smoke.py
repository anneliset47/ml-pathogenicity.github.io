from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ARTIFACTS = [
    ROOT / "figures" / "pca_projection_sampled.png",
    ROOT / "figures" / "silhouette_scores_sampled.png",
    ROOT / "figures" / "decision_tree_confusion_matrix.png",
    ROOT / "figures" / "naive_bayes_confusion_matrix.png",
    ROOT / "figures" / "svm_confusion_matrices.png",
    ROOT / "data" / "processed" / "svm_results.csv",
]


def _ensure_pipeline_outputs() -> None:
    if all(path.exists() for path in REQUIRED_ARTIFACTS):
        return

    subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), "--all"],
        check=True,
        cwd=ROOT,
    )


def test_pipeline_artifacts_exist_and_nonempty() -> None:
    _ensure_pipeline_outputs()

    missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
    assert not missing, f"Missing expected artifacts: {missing}"

    empty = [str(path) for path in REQUIRED_ARTIFACTS if path.stat().st_size == 0]
    assert not empty, f"Empty artifact files: {empty}"


def test_pipeline_cli_step_mode_runs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_pipeline.py"),
            "--step",
            "svm",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[OK] svm" in result.stdout
