# Predicting the Pathogenicity of Genetic Variants Using Machine Learning

[![Reproducibility Check](https://github.com/anneliset47/ml-pathogenicity.github.io/actions/workflows/reproducibility.yml/badge.svg)](https://github.com/anneliset47/ml-pathogenicity.github.io/actions/workflows/reproducibility.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Machine learning analysis and visual reporting for classifying genetic variants as pathogenic vs non-pathogenic.

- Live project site: https://ml-pathogenicity.github.io/
- Primary audience: recruiters, collaborators, and reviewers who need a one-command reproducible run.

## Project Structure

```text
.
├── data/
│   ├── raw/            # source datasets
│   ├── processed/      # cleaned and model output tables
│   └── derived/
├── src/
│   ├── data_ingestion/
│   ├── data_processing/
│   ├── modeling/
│   └── visualization/
├── figures/            # generated plots
├── docs/               # GitHub Pages site content
├── run_pipeline.py     # reproducible pipeline entrypoint
└── Makefile            # setup/reproduce shortcuts
```

## Quickstart

### Option A: Make (fastest)

```bash
git clone https://github.com/anneliset47/ml-pathogenicity.github.io.git
cd ml-pathogenicity.github.io
make setup
make reproduce
```

### Option B: Manual

```bash
git clone https://github.com/anneliset47/ml-pathogenicity.github.io.git
cd ml-pathogenicity.github.io
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py --all
```

## What the Pipeline Runs

`python run_pipeline.py --all` runs in deterministic order:

1. PCA
2. Clustering
3. Decision Tree
4. Naive Bayes
5. SVM

Outputs are generated in:

- `figures/`
- `data/processed/`

## Optional Neural Network Step

The neural network is intentionally optional because TensorFlow installation can vary by platform.

```bash
pip install -r requirements-optional.txt
python run_pipeline.py --all --with-nn
```

## Reproducibility Guarantees

- Fixed random seeds are used where applicable (`random_state=42` / NumPy seed).
- Required input files are validated before each step.
- Input datasets are included in this repository.
- CI enforces reproducibility on every push/PR via `.github/workflows/reproducibility.yml`.

## Expected Artifacts

Core run (`--all`) should produce/update:

- `figures/pca_projection_sampled.png`
- `figures/silhouette_scores_sampled.png`
- `figures/decision_tree_confusion_matrix.png`
- `figures/naive_bayes_confusion_matrix.png`
- `figures/svm_confusion_matrices.png`
- `data/processed/svm_results.csv`

Optional NN run also produces:

- `figures/nn_confusion_matrix.png`
- `data/processed/nn_train_sample.csv`

## Repository Standards

- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`
- Changelog: `CHANGELOG.md`

## Troubleshooting

- Use `python3` if `python` is unavailable.
- Ensure the virtual environment is activated before install/run.
- If TensorFlow fails on your machine, skip `--with-nn` and run the core pipeline.
