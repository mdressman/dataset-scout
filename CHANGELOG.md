# Changelog

All notable changes to `dataset-scout` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Until `1.0.0`, **minor** version bumps may carry breaking changes;
patches are bug-fix-only. See `README.md` for the JSONL/output
contracts that are explicitly stability-tracked.

## [Unreleased]

### Added
- Repository governance scaffolding: `CODEOWNERS`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, PR and issue templates.
- Tag-triggered release workflow (`.github/workflows/release.yml`)
  that builds sdist + wheel and attaches them to a GitHub Release.
- CodeQL static analysis workflow (`.github/workflows/codeql.yml`).
- This `CHANGELOG.md`.

## [0.0.1] - 2026-01-01

### Added
- Initial pre-alpha: brief → recon → curate pipeline, HuggingFace +
  Kaggle discovery, Semantic Scholar + arXiv paper search, LLM
  strategy assessor.

[Unreleased]: https://github.com/mdressman/dataset-scout/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/mdressman/dataset-scout/releases/tag/v0.0.1
