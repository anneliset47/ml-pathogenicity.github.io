# Changelog

All notable changes to this project are documented in this file.
Follow Keep a Changelog principles and move `[Unreleased]` entries into a versioned section at release time.

## [Unreleased]

### Added

- Reproducible pipeline entrypoint via `run_pipeline.py --all` with input validation.
- Recruiter-focused setup and execution flow in `README.md`.
- GitHub Actions reproducibility workflow (`.github/workflows/reproducibility.yml`).
- Project governance and collaboration files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, templates, CODEOWNERS).
- Maintenance and metadata files (`.editorconfig`, `.gitattributes`, `.github/dependabot.yml`, `CITATION.cff`).
- Developer quality toolchain (`ruff`, `mypy`, `pytest`) via `requirements-dev.txt` and `pyproject.toml`.
- Pipeline smoke tests in `tests/test_pipeline_smoke.py`.
- Focused unit tests for pipeline utility logic in `tests/test_pipeline_unit.py`.
- Release notes template in `RELEASE_NOTES_TEMPLATE.md`.

### Changed

- Dependency manifests split into core (`requirements.txt`) and optional NN extras (`requirements-optional.txt`).
- CI upgraded to gate linting, type-checking, tests, and reproducibility.
- README expanded with architecture, limitations, and future work sections.
