"""
lucj_sampler --- LUCJ circuit sampler using Qiskit + ffsim
==========================================================

Implements :class:`LUCJSampler`, which builds a locally-unitary
cluster-Jastrow (LUCJ) circuit from CCSD amplitudes and samples
bitstrings via ``ffsim.qiskit.FfsimSampler``.

Requires ``qiskit``, ``ffsim``, and ``pyscf`` (all optional
dependencies).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict

import numpy as np
import torch

from qvartools.samplers.sampler import Sampler, SamplerResult

__all__ = [
    "LUCJSampler",
]

logger = logging.getLogger(__name__)

try:
    import ffsim
    from ffsim.qiskit import (
        FfsimSampler,
        PrepareHartreeFockJW,
        UCJOpSpinBalancedJW,
    )
    from qiskit.circuit import QuantumCircuit

    _HAS_QISKIT_FFSIM = True
except ImportError:
    _HAS_QISKIT_FFSIM = False


class LUCJSampler(Sampler):
    """LUCJ circuit sampler using Qiskit + ffsim.

    Pipeline:

    1. PySCF RHF -> CCSD -> t1, t2 amplitudes
    2. ``ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2, t1, n_reps=…)``
    3. Build Qiskit circuit: ``PrepareHartreeFockJW`` + ``UCJOpSpinBalancedJW``
    4. Simulate with ``ffsim.qiskit.FfsimSampler``
    5. Return bitstring samples as configuration tensors

    Parameters
    ----------
    hamiltonian : Hamiltonian
        Molecular Hamiltonian providing ``n_orbitals``, ``n_alpha``,
        ``n_beta``, and ``integrals``.
    n_reps : int, optional
        Number of UCJ repetitions (default ``2``).
    device : str, optional
        Torch device for output tensors (default ``"cpu"``).

    Raises
    ------
    ImportError
        If ``qiskit`` or ``ffsim`` is not installed.

    Examples
    --------
    >>> sampler = LUCJSampler(hamiltonian, n_reps=2)
    >>> result = sampler.sample(5000)
    >>> result.configs.shape[1]  # == 2 * n_orbitals
    10
    """

    def __init__(
        self,
        hamiltonian,
        n_reps: int = 2,
        device: str = "cpu",
    ) -> None:
        if not _HAS_QISKIT_FFSIM:
            raise ImportError(
                "qiskit and ffsim are required for LUCJSampler. "
                "Install with: pip install qiskit ffsim"
            )

        self.hamiltonian = hamiltonian
        self.n_reps: int = n_reps
        self.device: str = device

        self.n_orbitals: int = hamiltonian.n_orbitals
        self.n_alpha: int = hamiltonian.n_alpha
        self.n_beta: int = hamiltonian.n_beta
        self.n_qubits: int = 2 * self.n_orbitals

        self._circuit: QuantumCircuit = self._build_circuit()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def _build_circuit(self) -> "QuantumCircuit":
        """Build the LUCJ circuit from molecular integrals.

        Returns
        -------
        QuantumCircuit
            The compiled Qiskit circuit with measurement gates.
        """
        integrals = self.hamiltonian.integrals

        # Obtain CCSD amplitudes via PySCF
        t1, t2 = self._get_ccsd_amplitudes(integrals)

        # Build UCJ operator from amplitudes
        ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2, t1_amplitudes=t1, n_reps=self.n_reps
        )

        # Assemble Qiskit circuit
        qc = QuantumCircuit(self.n_qubits)
        qc.append(
            PrepareHartreeFockJW(
                self.n_orbitals, (self.n_alpha, self.n_beta)
            ),
            range(self.n_qubits),
        )
        qc.append(
            UCJOpSpinBalancedJW(ucj_op),
            range(self.n_qubits),
        )
        qc.measure_all()

        return qc

    @staticmethod
    def _get_ccsd_amplitudes(integrals):
        """Run PySCF CCSD to obtain t1/t2 amplitudes.

        Parameters
        ----------
        integrals : object
            Molecular integrals object with geometry metadata.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(t1, t2)`` CCSD amplitude tensors.
        """
        from pyscf import cc, gto, scf

        if integrals._geometry is not None:
            mol = gto.Mole()
            mol.atom = integrals._geometry
            mol.basis = integrals._basis
            mol.charge = integrals._charge
            mol.spin = integrals._spin
            mol.build()

            mf = scf.RHF(mol)
            mf.kernel()

            mycc = cc.CCSD(mf)
            mycc.kernel()

            return mycc.t1, mycc.t2

        # Fallback for CAS systems: identity (zero) amplitudes
        n_occ = integrals._n_alpha if hasattr(integrals, "_n_alpha") else 1
        n_orb = integrals._n_orbitals if hasattr(integrals, "_n_orbitals") else 1
        n_virt = n_orb - n_occ
        t1 = np.zeros((n_occ, n_virt))
        t2 = np.zeros((n_occ, n_occ, n_virt, n_virt))
        return t1, t2

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_samples: int) -> SamplerResult:
        """Sample bitstrings from the LUCJ circuit.

        Parameters
        ----------
        n_samples : int
            Number of shots to execute.

        Returns
        -------
        SamplerResult
            Unique sampled configurations with metadata.

        Raises
        ------
        ValueError
            If ``n_samples < 1``.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")

        t_start = time.perf_counter()

        sampler = FfsimSampler(default_shots=n_samples, seed=42)
        job = sampler.run([self._circuit])
        result = job.result()
        raw_counts = result[0].data.meas.get_counts()

        # Convert bitstrings to tensor configs
        configs_list: list[list[int]] = []
        for bitstring, count in raw_counts.items():
            config = [int(b) for b in bitstring]
            configs_list.extend([config] * count)

        configs = torch.tensor(
            configs_list, dtype=torch.long, device=self.device
        )
        unique_configs = torch.unique(configs, dim=0)

        # Build bitstring counts
        bitstrings = [
            "".join(str(int(b)) for b in row) for row in configs.int()
        ]
        counts: Dict[str, int] = dict(Counter(bitstrings))

        wall_time = time.perf_counter() - t_start

        metadata: Dict[str, Any] = {
            "n_raw_samples": n_samples,
            "n_unique": len(unique_configs),
            "unique_ratio": len(unique_configs) / max(n_samples, 1),
            "n_reps": self.n_reps,
            "sampler_type": "LUCJ",
        }

        logger.info(
            "LUCJSampler: drew %d shots (%d unique) in %.3fs",
            n_samples,
            len(unique_configs),
            wall_time,
        )

        return SamplerResult(
            configs=unique_configs,
            counts=counts,
            log_probs=None,
            wall_time=wall_time,
            metadata=metadata,
        )
