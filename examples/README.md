# Examples

Standalone, runnable examples demonstrating qvartools usage.

## Files

| Example | Description |
|---------|-------------|
| `basic_h2.py` | Simplest usage: run full pipeline on H2 with one function call |
| `custom_pipeline.py` | Configure pipeline manually and run stages individually |
| `compare_solvers.py` | Compare FCI, SQD, and SKQD solvers on the same molecule |
| `spin_hamiltonian.py` | Construct and diagonalize spin Hamiltonians |

## Running

```bash
# From the repository root
python examples/basic_h2.py
python examples/custom_pipeline.py
python examples/compare_solvers.py
python examples/spin_hamiltonian.py
```

All examples require the base installation (`pip install -e .`).
The molecular examples additionally require PySCF (`pip install -e ".[pyscf]"`).
