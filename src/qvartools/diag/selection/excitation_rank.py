"""
excitation_rank --- Excitation-rank and Hamming-distance utilities
=================================================================

Provides functions for computing excitation ranks, Hamming distances,
and bit-packing of binary configuration tensors.

Functions
---------
compute_excitation_rank
    Count the number of bit flips between a configuration and a reference.
compute_hamming_distance
    Hamming distance between two binary configuration tensors.
bitpack_configs
    Pack binary configuration tensors into int64 words for fast Hamming.
bitpacked_hamming
    Vectorized Hamming distance via popcount on bit-packed tensors.
"""

from __future__ import annotations

import torch

__all__ = [
    "compute_excitation_rank",
    "compute_hamming_distance",
    "bitpack_configs",
    "bitpacked_hamming",
]


def compute_excitation_rank(config: torch.Tensor, reference: torch.Tensor) -> int:
    """Count the number of bit flips from a reference configuration.

    The excitation rank is the Hamming distance between the configuration
    and the reference, which for a fermionic system corresponds to the
    number of single-particle excitations (orbital substitutions).

    Parameters
    ----------
    config : torch.Tensor
        Binary configuration vector, shape ``(n_orbitals,)``.
    reference : torch.Tensor
        Reference configuration (e.g. Hartree-Fock), shape ``(n_orbitals,)``.

    Returns
    -------
    int
        Number of differing bits (excitation rank).

    Raises
    ------
    ValueError
        If ``config`` and ``reference`` have different shapes.

    Examples
    --------
    >>> ref = torch.tensor([1, 1, 0, 0])
    >>> cfg = torch.tensor([1, 0, 1, 0])
    >>> compute_excitation_rank(cfg, ref)
    2
    """
    if config.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: config {config.shape} vs reference {reference.shape}"
        )
    return int((config != reference).sum().item())


def compute_hamming_distance(config1: torch.Tensor, config2: torch.Tensor) -> int:
    """Compute the Hamming distance between two binary configurations.

    Parameters
    ----------
    config1 : torch.Tensor
        First binary configuration, shape ``(n_orbitals,)``.
    config2 : torch.Tensor
        Second binary configuration, shape ``(n_orbitals,)``.

    Returns
    -------
    int
        Number of positions where the two configurations differ.

    Raises
    ------
    ValueError
        If ``config1`` and ``config2`` have different shapes.

    Examples
    --------
    >>> a = torch.tensor([1, 0, 1, 0])
    >>> b = torch.tensor([1, 1, 0, 0])
    >>> compute_hamming_distance(a, b)
    2
    """
    if config1.shape != config2.shape:
        raise ValueError(
            f"Shape mismatch: config1 {config1.shape} vs config2 {config2.shape}"
        )
    return int((config1 != config2).sum().item())


def bitpack_configs(configs: torch.Tensor) -> torch.Tensor:
    """Pack binary configurations into int64 words for O(1) Hamming distance.

    Each configuration of ``n_orbitals`` binary values is packed into
    ``ceil(n_orbitals / 63)`` int64 words (63 bits per word to avoid sign
    bit issues).

    Parameters
    ----------
    configs : torch.Tensor
        Binary configuration matrix, shape ``(n_configs, n_orbitals)``
        with entries in ``{0, 1}``.

    Returns
    -------
    torch.Tensor
        Bit-packed tensor, shape ``(n_configs, n_words)``, dtype ``int64``.

    Examples
    --------
    >>> cfgs = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0]])
    >>> packed = bitpack_configs(cfgs)
    >>> packed.shape
    torch.Size([2, 1])
    """
    n_configs, n_orbitals = configs.shape
    bits_per_word = 63
    n_words = (n_orbitals + bits_per_word - 1) // bits_per_word

    packed = torch.zeros(n_configs, n_words, dtype=torch.int64, device=configs.device)

    for w in range(n_words):
        start = w * bits_per_word
        end = min(start + bits_per_word, n_orbitals)
        chunk = configs[:, start:end]

        shifts = torch.arange(end - start, dtype=torch.int64, device=configs.device)
        word_values = (chunk.to(torch.int64) << shifts.unsqueeze(0)).sum(dim=1)
        packed[:, w] = word_values

    return packed


def bitpacked_hamming(
    packed: torch.Tensor, idx_a: torch.Tensor, idx_b: torch.Tensor
) -> torch.Tensor:
    """Compute vectorized Hamming distances via popcount on bit-packed configs.

    Parameters
    ----------
    packed : torch.Tensor
        Bit-packed configurations, shape ``(n_configs, n_words)``, dtype
        ``int64``, as returned by :func:`bitpack_configs`.
    idx_a : torch.Tensor
        Index tensor for the first set of configurations, shape ``(n_pairs,)``.
    idx_b : torch.Tensor
        Index tensor for the second set of configurations, shape ``(n_pairs,)``.

    Returns
    -------
    torch.Tensor
        Hamming distances, shape ``(n_pairs,)``, dtype ``int64``.

    Examples
    --------
    >>> cfgs = torch.tensor([[1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]])
    >>> packed = bitpack_configs(cfgs)
    >>> idx_a = torch.tensor([0, 0])
    >>> idx_b = torch.tensor([1, 2])
    >>> bitpacked_hamming(packed, idx_a, idx_b)
    tensor([4, 2])
    """
    xor_vals = packed[idx_a] ^ packed[idx_b]

    # Portable popcount via Kernighan-style bit counting on each word
    distances = torch.zeros(idx_a.shape[0], dtype=torch.int64, device=packed.device)
    for w in range(packed.shape[1]):
        word = xor_vals[:, w]
        # Use bit manipulation popcount: count set bits
        count = torch.zeros_like(word)
        remaining = word.clone()
        while True:
            mask = remaining != 0
            if not mask.any():
                break
            remaining = remaining & (remaining - 1)
            count = count + mask.to(torch.int64)
        distances = distances + count

    return distances
