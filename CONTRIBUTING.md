# Contributing to qvartools

Thank you for your interest in contributing to qvartools! This guide will help you
get started with development and the contribution process.

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork locally:

```bash
git clone https://github.com/<your-username>/qvartools.git
cd qvartools
```

3. Add the upstream remote:

```bash
git remote add upstream https://github.com/George930502/qvartools.git
```

### Development Setup

Create a virtual environment and install the package in editable mode with all
development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev,pyscf]"
```

This installs the package along with testing tools (pytest, pytest-cov), the
linter (ruff), and the type checker (mypy).

Verify your setup by running the test suite:

```bash
pytest
```

## Code Style

### Formatting and Linting

We use [Ruff](https://docs.astral.sh/ruff/) for both formatting and linting.
The configuration lives in `pyproject.toml`.

```bash
# Check for lint issues
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

All code must pass `ruff check` with zero warnings before merging.

### Type Hints

All public functions and methods must include type annotations. Use modern
Python typing syntax (Python 3.10+):

```python
def compute_energy(
    hamiltonian: np.ndarray,
    state: torch.Tensor,
    *,
    num_samples: int = 1024,
) -> float:
    ...
```

Run the type checker with:

```bash
mypy src/qvartools/
```

### Docstrings

Use **NumPy-style** docstrings for all public modules, classes, and functions:

```python
def solve(hamiltonian, method="fci"):
    """Compute the ground-state energy of a molecular Hamiltonian.

    Parameters
    ----------
    hamiltonian : np.ndarray
        The qubit Hamiltonian matrix.
    method : str, optional
        Solver method, by default ``"fci"``.

    Returns
    -------
    float
        The ground-state energy in Hartree.

    Raises
    ------
    ValueError
        If ``method`` is not a recognized solver name.
    """
```

## Testing

We use [pytest](https://docs.pytest.org/) for all testing.

### Running Tests

```bash
# Run the full test suite
pytest

# Run with coverage report
pytest --cov=qvartools --cov-report=term-missing

# Run a specific test file
pytest tests/test_hamiltonian.py

# Run tests matching a keyword
pytest -k "test_energy"
```

### Test Markers

Some tests require optional dependencies or hardware. Use markers to skip them
when those resources are unavailable:

| Marker   | Meaning                        |
|----------|--------------------------------|
| `slow`   | Long-running tests (>30 s)     |
| `gpu`    | Requires a CUDA-capable GPU    |
| `pyscf`  | Requires PySCF                 |

Skip marked tests with:

```bash
pytest -m "not slow and not gpu"
```

### Writing Tests

- Place test files in the `tests/` directory, mirroring the `src/qvartools/`
  structure.
- Name test files `test_<module>.py` and test functions `test_<behavior>`.
- Keep tests focused: one logical assertion per test.
- Use `pytest.approx` for floating-point comparisons.
- Use `pytest.mark.parametrize` to cover multiple inputs concisely.

## Pull Request Process

1. **Create a feature branch** from `main`:

   ```bash
   git checkout -b feat/my-feature main
   ```

2. **Make your changes.** Commit early and often with clear messages following
   [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat: add RBM wave-function ansatz
   fix: correct sign in Jordan-Wigner mapping
   docs: update solver API reference
   ```

3. **Ensure quality:**

   ```bash
   ruff check src/ tests/
   ruff format --check src/ tests/
   pytest --cov=qvartools
   ```

4. **Push and open a PR** against `main`:

   ```bash
   git push -u origin feat/my-feature
   ```

5. Fill out the pull request template. Link any related issues.

6. A maintainer will review your PR. Address feedback by pushing additional
   commits (do not force-push during review).

7. Once approved, a maintainer will merge your PR.

## Reporting Issues

### Bug Reports

Use the **Bug Report** issue template. Include:

- A minimal reproducing example.
- Expected vs. actual behavior.
- Your environment (OS, Python version, qvartools version, GPU if relevant).

### Feature Requests

Use the **Feature Request** issue template. Describe:

- The problem or use case.
- Your proposed solution (if any).
- Any alternatives you considered.

## License

By contributing to qvartools, you agree that your contributions will be licensed
under the MIT License.
