# Contributing

Thank you for your interest in improving this project.

## Getting Started

1. Fork the repository.
2. Create a feature branch from `main`.
3. Set up the environment:

```bash
make setup
```

4. Re-run the reproducibility pipeline before opening a PR:

```bash
make reproduce
```

## Development Guidelines

- Keep changes focused and minimal.
- Prefer deterministic behavior (`random_state`/fixed seeds).
- Avoid hard-coded absolute file paths.
- Update docs when behavior or commands change.

## Pull Request Checklist

- [ ] Code runs locally.
- [ ] `python run_pipeline.py --all` completes successfully.
- [ ] Documentation has been updated if needed.
- [ ] Changes are scoped to the stated objective.

## Commit Style

Use clear, imperative commit messages. Examples:

- `Add reproducibility workflow`
- `Document quickstart for recruiters`
- `Fix data path validation in pipeline`

## Versioning & Releases

This repository uses Semantic Versioning (`MAJOR.MINOR.PATCH`) for release tags.

- `PATCH`: bug fixes and documentation-only improvements
- `MINOR`: backward-compatible features and workflow improvements
- `MAJOR`: breaking behavior or interface changes

### Create a release tag

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

### Release Checklist

- [ ] CI is green on `main`.
- [ ] `python run_pipeline.py --all` passes locally.
- [ ] `CHANGELOG.md` has been updated under `[Unreleased]`.
- [ ] Version tag follows SemVer and release notes summarize key changes.

## Questions

Open an issue for design discussions, bug reports, or clarification requests.
