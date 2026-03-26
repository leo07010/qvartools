# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Rename `SampleBasedKrylovDiagonalization` to `ClassicalKrylovDiagonalization` (ADR-001)
- **BREAKING**: Rename `FlowGuidedSKQD` to `FlowGuidedKrylovDiag` (ADR-001)
- **BREAKING**: Default `subspace_mode` changed from `"skqd"` to `"classical_krylov"`
- `subspace_mode="skqd"` now routes to `QuantumCircuitSKQD` (real CUDA-Q SKQD)
- Old class names kept as deprecated aliases until v0.1.0

### Added
- `TransformerAsNQS` adapter: enables `AutoregressiveTransformer` in NF training pipeline
- `NQSWithSampling` adapter: enables any `NeuralQuantumState` in HI training pipeline
- `qvartools._logging` module with `configure_logging()` and `get_logger()`
- `QVARTOOLS_LOG_LEVEL` environment variable for log level control
- CI: mypy type checking job, coverage threshold enforcement
- ADR-001 decision record at `docs/decisions/`

### Fixed
- `TransformerNFSampler._build_nqs()` used wrong parameter name `hidden_dim` instead of `hidden_dims`

## [0.0.0] - 2026-03-26

### Added
- Initial development release of qvartools
- Molecular and spin Hamiltonians with Jordan-Wigner mapping
- Neural quantum state architectures (Dense, Complex, RBM, Autoregressive Transformer)
- Normalizing flows with particle-number conservation
- Physics-guided mixed-objective training (teacher + variational energy + entropy)
- Sample-based Krylov quantum diagonalization (SKQD) with residual expansion
- CIPSI-style perturbative basis enrichment
- Unified solver interface (FCI, CCSD, SQD, SKQD, iterative variants)
- Molecule registry with 8 pre-configured benchmarks (H2 to C2H4)
- Automatic system-size scaling for network architectures and sampling budgets
- YAML-based experiment configuration system with CLI overrides
- Pipeline scripts integrated with shared config_loader (--config flag)
- Comprehensive API documentation with Sphinx (autodoc + napoleon)
- Full documentation site: getting started, user guide, API reference, tutorials, developer guide
- CODE_OF_CONDUCT.md (Contributor Covenant)
- Standalone examples (basic_h2, custom_pipeline, compare_solvers, spin_hamiltonian)
- GitHub CI workflows for testing and documentation
- Issue templates (bug report, feature request) and PR template
- Docker GPU support
- ReadTheDocs integration

[Unreleased]: https://github.com/George930502/qvartools/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/George930502/qvartools/releases/tag/v0.0.0
