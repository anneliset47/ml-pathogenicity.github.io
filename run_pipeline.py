"""Run the ML Pathogenicity project pipeline in a reproducible order.

Default flow (offline, using checked-in datasets):
1. PCA
2. Clustering
3. Decision Tree
4. Naive Bayes
5. SVM

Optional:
- Neural Network (requires TensorFlow): --with-nn
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


PIPELINE_STEPS = [
    {
        "name": "pca",
        "script": BASE_DIR / "src" / "modeling" / "pca.py",
        "requires": [
            BASE_DIR / "data" / "processed" / "cleaned_ensembl.csv",
        ],
    },
    {
        "name": "clustering",
        "script": BASE_DIR / "src" / "modeling" / "clustering.py",
        "requires": [
            BASE_DIR / "data" / "processed" / "cleaned_ensembl.csv",
        ],
    },
    {
        "name": "decision_tree",
        "script": BASE_DIR / "src" / "modeling" / "decision_tree.py",
        "requires": [
            BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv",
        ],
    },
    {
        "name": "naive_bayes",
        "script": BASE_DIR / "src" / "modeling" / "naive_bayes.py",
        "requires": [
            BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv",
        ],
    },
    {
        "name": "svm",
        "script": BASE_DIR / "src" / "modeling" / "svm.py",
        "requires": [
            BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv",
        ],
    },
    {
        "name": "nn",
        "script": BASE_DIR / "src" / "modeling" / "nn.py",
        "requires": [
            BASE_DIR / "data" / "raw" / "clinvar_repeat_pathogenic_variants.csv",
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project pipeline.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full default pipeline (PCA, clustering, trees, NB, SVM).",
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=[step["name"] for step in PIPELINE_STEPS],
        help="Run one or more specific steps (can be passed multiple times).",
    )
    parser.add_argument(
        "--with-nn",
        action="store_true",
        help="Include neural network step (requires TensorFlow).",
    )
    return parser.parse_args()


def validate_inputs(step: dict) -> None:
    missing = [path for path in step["requires"] if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing required input files for step '{step['name']}':\n{missing_text}"
        )


def run_step(step: dict) -> None:
    print(f"\n[RUN] {step['name']} -> {step['script']}")
    validate_inputs(step)
    subprocess.run([sys.executable, str(step["script"])], check=True, cwd=BASE_DIR)
    print(f"[OK] {step['name']}")


def resolve_selected_steps(args: argparse.Namespace) -> list[dict]:
    if args.step:
        selected = [name for name in args.step]
    else:
        selected = ["pca", "clustering", "decision_tree", "naive_bayes", "svm"]

    if args.all and "nn" not in selected and args.with_nn:
        selected.append("nn")
    elif args.with_nn and "nn" not in selected:
        selected.append("nn")

    return [step for step in PIPELINE_STEPS if step["name"] in selected]


def main() -> None:
    args = parse_args()
    selected_steps = resolve_selected_steps(args)

    if not selected_steps:
        raise SystemExit("No steps selected. Use --all or --step.")

    print("Running ML Pathogenicity pipeline")
    print(f"Python executable: {sys.executable}")
    print(f"Project root: {BASE_DIR}")

    for step in selected_steps:
        run_step(step)

    print("\nPipeline complete.")
    print(f"Generated outputs in: {BASE_DIR / 'figures'} and {BASE_DIR / 'data' / 'processed'}")


if __name__ == "__main__":
    main()
