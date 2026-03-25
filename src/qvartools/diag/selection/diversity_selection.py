"""
diversity_selection --- Diversity-aware basis configuration selection
====================================================================

Implements excitation-rank bucketing, Hamming-distance filtering, and
optional determinantal point process (DPP) selection to build a compact,
diverse subset of computational-basis configurations for projected
eigenvalue problems.

Classes
-------
DiversityConfig
    Dataclass holding all hyperparameters for diversity selection.
DiversitySelector
    Stateful selector that buckets, ranks, and filters configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from qvartools.diag.selection.excitation_rank import (  # noqa: F401
    bitpack_configs,
    bitpacked_hamming,
    compute_excitation_rank,
    compute_hamming_distance,
)

__all__ = [
    "DiversityConfig",
    "DiversitySelector",
    "compute_excitation_rank",
    "compute_hamming_distance",
    "bitpack_configs",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiversityConfig:
    """Hyperparameters for diversity-aware configuration selection.

    The selector partitions configurations by excitation rank relative to a
    reference state and allocates per-rank quotas according to the given
    fractions.  Within each rank bucket, configurations are selected by NQS
    importance weight and filtered to enforce a minimum pairwise Hamming
    distance.

    Parameters
    ----------
    max_configs : int
        Maximum number of configurations to select.
    rank_0_fraction : float
        Fraction of ``max_configs`` allocated to rank-0 (reference) configs.
    rank_1_fraction : float
        Fraction allocated to single-excitation configurations.
    rank_2_fraction : float
        Fraction allocated to double-excitation configurations.
    rank_3_fraction : float
        Fraction allocated to triple-excitation configurations.
    rank_4_plus_fraction : float
        Fraction allocated to quadruple-and-higher excitations.
    min_hamming_distance : int
        Minimum pairwise Hamming distance between any two selected configs.
    use_dpp_selection : bool
        If ``True``, use a determinantal point process kernel for maximal
        diversity within each rank bucket (more expensive).
    dpp_kernel_scale : float
        Length-scale parameter for the DPP Gaussian kernel.

    Examples
    --------
    >>> cfg = DiversityConfig(max_configs=500, min_hamming_distance=3)
    >>> cfg.max_configs
    500
    """

    max_configs: int = 1000
    rank_0_fraction: float = 0.05
    rank_1_fraction: float = 0.15
    rank_2_fraction: float = 0.40
    rank_3_fraction: float = 0.25
    rank_4_plus_fraction: float = 0.15
    min_hamming_distance: int = 2
    use_dpp_selection: bool = False
    dpp_kernel_scale: float = 0.1

    def __post_init__(self) -> None:
        """Validate fraction constraints after initialization."""
        total = (
            self.rank_0_fraction
            + self.rank_1_fraction
            + self.rank_2_fraction
            + self.rank_3_fraction
            + self.rank_4_plus_fraction
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Rank fractions must sum to 1.0, got {total:.6f}"
            )
        if self.max_configs < 1:
            raise ValueError(
                f"max_configs must be >= 1, got {self.max_configs}"
            )
        if self.min_hamming_distance < 0:
            raise ValueError(
                f"min_hamming_distance must be >= 0, got {self.min_hamming_distance}"
            )


# ---------------------------------------------------------------------------
# Diversity selector
# ---------------------------------------------------------------------------


class DiversitySelector:
    """Select a diverse, representative subset of configurations.

    Configurations are bucketed by excitation rank relative to a reference
    state (e.g. Hartree-Fock).  Within each bucket, the top-weighted
    configurations are greedily selected while enforcing a minimum pairwise
    Hamming distance.  An optional DPP kernel can be applied for maximal
    coverage.

    Parameters
    ----------
    config : DiversityConfig
        Selection hyperparameters.
    reference : torch.Tensor
        Reference configuration (e.g. Hartree-Fock ground state),
        shape ``(n_orbitals,)`` with binary entries.
    n_orbitals : int
        Number of orbitals (must match ``reference.shape[0]``).

    Raises
    ------
    ValueError
        If ``reference`` length does not match ``n_orbitals``.

    Examples
    --------
    >>> cfg = DiversityConfig(max_configs=100)
    >>> ref = torch.tensor([1, 1, 1, 0, 0, 0])
    >>> selector = DiversitySelector(cfg, ref, n_orbitals=6)
    >>> selected, stats = selector.select(all_configs, weights)
    """

    def __init__(
        self,
        config: DiversityConfig,
        reference: torch.Tensor,
        n_orbitals: int,
    ) -> None:
        if reference.shape[0] != n_orbitals:
            raise ValueError(
                f"Reference length {reference.shape[0]} does not match "
                f"n_orbitals={n_orbitals}"
            )
        self._config = config
        self._reference = reference.clone()
        self._n_orbitals = n_orbitals

    def select(
        self,
        configs: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Select a diverse subset of configurations.

        Parameters
        ----------
        configs : torch.Tensor
            Pool of candidate configurations, shape ``(n_pool, n_orbitals)``
            with binary entries.
        weights : torch.Tensor or None, optional
            NQS importance weights for each configuration, shape ``(n_pool,)``.
            If ``None``, uniform weights are used.

        Returns
        -------
        selected : torch.Tensor
            Selected configurations, shape ``(n_selected, n_orbitals)``.
        stats : dict
            Selection statistics with keys:

            - ``"total_pool"`` : int -- size of the input pool.
            - ``"n_selected"`` : int -- number of selected configurations.
            - ``"rank_counts"`` : dict -- count per excitation rank in pool.
            - ``"rank_selected"`` : dict -- count per rank in selection.
            - ``"rank_quotas"`` : dict -- allocated quota per rank.
        """
        n_pool = configs.shape[0]

        if weights is None:
            weights = torch.ones(n_pool, dtype=torch.float64, device=configs.device)

        # --- Compute excitation ranks for all configs (vectorized) ---
        ranks = (configs != self._reference.unsqueeze(0)).sum(dim=1)

        # --- Build rank buckets ---
        rank_fractions = {
            0: self._config.rank_0_fraction,
            1: self._config.rank_1_fraction,
            2: self._config.rank_2_fraction,
            3: self._config.rank_3_fraction,
        }
        # Rank 4+ is a catch-all bucket
        max_cfg = self._config.max_configs

        rank_quotas: Dict[int, int] = {}
        for rank_val, frac in rank_fractions.items():
            rank_quotas[rank_val] = max(1, int(frac * max_cfg))
        rank_quotas[4] = max(1, int(self._config.rank_4_plus_fraction * max_cfg))

        # Adjust quotas so they sum to max_configs
        total_allocated = sum(rank_quotas.values())
        if total_allocated != max_cfg:
            # Add/subtract difference from the largest bucket (rank 2)
            rank_quotas[2] = rank_quotas[2] + (max_cfg - total_allocated)

        # --- Bucket and select ---
        selected_indices: list[int] = []
        rank_pool_counts: Dict[int, int] = {}
        rank_selected_counts: Dict[int, int] = {}

        # Bit-pack for fast Hamming if we have a minimum distance constraint
        packed = bitpack_configs(configs) if self._config.min_hamming_distance > 0 else None

        for rank_key, quota in sorted(rank_quotas.items()):
            if rank_key < 4:
                bucket_mask = ranks == rank_key
            else:
                bucket_mask = ranks >= 4

            bucket_indices = bucket_mask.nonzero(as_tuple=False).squeeze(-1)
            rank_pool_counts[rank_key] = int(bucket_indices.shape[0])

            if bucket_indices.shape[0] == 0:
                rank_selected_counts[rank_key] = 0
                continue

            # Sort by weight descending within bucket
            bucket_weights = weights[bucket_indices]
            sorted_order = torch.argsort(bucket_weights, descending=True)
            sorted_bucket = bucket_indices[sorted_order]

            if self._config.use_dpp_selection and packed is not None:
                bucket_selected = self._dpp_select(
                    sorted_bucket, packed, quota
                )
            else:
                bucket_selected = self._greedy_select(
                    sorted_bucket, packed, quota
                )

            selected_indices.extend(bucket_selected)
            rank_selected_counts[rank_key] = len(bucket_selected)

        if len(selected_indices) == 0:
            logger.warning("No configurations selected; returning empty tensor.")
            empty = torch.empty(0, self._n_orbitals, dtype=configs.dtype, device=configs.device)
            stats = {
                "total_pool": n_pool,
                "n_selected": 0,
                "rank_counts": rank_pool_counts,
                "rank_selected": rank_selected_counts,
                "rank_quotas": dict(rank_quotas),
            }
            return empty, stats

        idx_tensor = torch.tensor(selected_indices, dtype=torch.long, device=configs.device)
        selected = configs[idx_tensor]

        stats = {
            "total_pool": n_pool,
            "n_selected": len(selected_indices),
            "rank_counts": rank_pool_counts,
            "rank_selected": rank_selected_counts,
            "rank_quotas": dict(rank_quotas),
        }

        logger.info(
            "Diversity selection: %d / %d configs selected (quotas: %s)",
            stats["n_selected"],
            stats["total_pool"],
            stats["rank_quotas"],
        )

        return selected, stats

    def _greedy_select(
        self,
        sorted_indices: torch.Tensor,
        packed: Optional[torch.Tensor],
        quota: int,
    ) -> list[int]:
        """Greedily select configs by weight, enforcing minimum Hamming distance.

        Parameters
        ----------
        sorted_indices : torch.Tensor
            Candidate indices sorted by weight (descending), shape ``(n,)``.
        packed : torch.Tensor or None
            Bit-packed configurations for fast Hamming, or ``None`` if no
            distance constraint.
        quota : int
            Maximum number of configurations to select from this bucket.

        Returns
        -------
        list of int
            Selected global indices.
        """
        selected: list[int] = []
        min_dist = self._config.min_hamming_distance

        for i in range(sorted_indices.shape[0]):
            if len(selected) >= quota:
                break

            candidate = int(sorted_indices[i].item())

            if min_dist > 0 and packed is not None and len(selected) > 0:
                # Check Hamming distance against all already-selected
                cand_idx = torch.tensor(
                    [candidate] * len(selected),
                    dtype=torch.long,
                    device=packed.device,
                )
                sel_idx = torch.tensor(
                    selected,
                    dtype=torch.long,
                    device=packed.device,
                )
                distances = bitpacked_hamming(packed, cand_idx, sel_idx)
                if distances.min().item() < min_dist:
                    continue

            selected.append(candidate)

        return selected

    def _dpp_select(
        self,
        sorted_indices: torch.Tensor,
        packed: torch.Tensor,
        quota: int,
    ) -> list[int]:
        """Select configs using a DPP-inspired greedy kernel for maximal coverage.

        Uses a Gaussian kernel over Hamming distances to define a quality-
        diversity trade-off.  The selection greedily picks the item that
        maximises the log-determinant of the kernel sub-matrix.

        Parameters
        ----------
        sorted_indices : torch.Tensor
            Candidate indices sorted by weight (descending), shape ``(n,)``.
        packed : torch.Tensor
            Bit-packed configurations for fast Hamming computation.
        quota : int
            Maximum number of configurations to select.

        Returns
        -------
        list of int
            Selected global indices.
        """
        n_candidates = min(sorted_indices.shape[0], quota * 5)
        candidates = sorted_indices[:n_candidates]
        n = candidates.shape[0]

        if n == 0:
            return []

        # Build pairwise Hamming distance matrix for candidates
        row_idx = torch.arange(n, device=packed.device).repeat_interleave(n)
        col_idx = torch.arange(n, device=packed.device).repeat(n)
        cand_a = candidates[row_idx]
        cand_b = candidates[col_idx]
        dists = bitpacked_hamming(packed, cand_a, cand_b).float().reshape(n, n)

        # Gaussian kernel: K_ij = exp(-d_ij^2 / (2 * scale^2))
        scale = self._config.dpp_kernel_scale * self._n_orbitals
        kernel = torch.exp(-dists.pow(2) / (2.0 * scale * scale))

        # Greedy DPP selection (approximate log-det maximization)
        selected: list[int] = []
        remaining = set(range(n))

        for _ in range(min(quota, n)):
            best_idx = -1
            best_score = -float("inf")

            for r in remaining:
                if len(selected) == 0:
                    score = float(kernel[r, r].item())
                else:
                    sel_tensor = torch.tensor(selected, dtype=torch.long, device=kernel.device)
                    k_ss = kernel[sel_tensor][:, sel_tensor]
                    k_new_row = kernel[r, sel_tensor].unsqueeze(0)
                    k_new_diag = kernel[r, r].unsqueeze(0).unsqueeze(0)
                    augmented = torch.cat(
                        [
                            torch.cat([k_ss, k_new_row.T], dim=1),
                            torch.cat([k_new_row, k_new_diag], dim=1),
                        ],
                        dim=0,
                    )
                    sign, logabsdet = torch.linalg.slogdet(augmented)
                    score = float(logabsdet.item()) if sign.item() > 0 else -float("inf")

                if score > best_score:
                    best_score = score
                    best_idx = r

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)

        # Map local indices back to global
        return [int(candidates[s].item()) for s in selected]
