# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Axial Next-Nearest-Neighbor Ising (ANNNI) Hamiltonian and phase sampler.

Reference: Bermejo et al., arXiv:2408.12739, eq. (8).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ._base import pauli_term

#: Ordered list of phase labels for the ANNNI model.
PHASE_LABELS: list[str] = ["ferromagnetic", "paramagnetic", "floating", "antiphase"]


def build_hamiltonian(n: int, kappa: float, h: float, j1: float = 1.0) -> SparsePauliOp:
    r"""ANNNI Hamiltonian (Paper eq. 8).

    .. math::

        H = -J_1 \sum_{i=1}^{n-1} X_i X_{i+1}
            - J_2 \sum_{i=1}^{n-2} X_i X_{i+2}
            - B \sum_{i=1}^{n} Z_i

    with :math:`J_2 = -\kappa J_1` and :math:`B = h J_1`.

    Phase diagram (see Fig. 5 in the reference, axes :math:`\kappa` vs
    :math:`h` with :math:`J_1 = 1`):

    * **ferromagnetic** (I) — small :math:`\kappa`, small :math:`h`
    * **paramagnetic** (II) — small :math:`\kappa`, large :math:`h`
    * **floating** (III) — large :math:`\kappa`, moderate :math:`h`
    * **antiphase** (IV) — large :math:`\kappa`, small :math:`h`

    Args:
        n: Number of lattice sites (qubits).
        kappa: Dimensionless ratio :math:`\kappa = -J_2 / J_1`.
        h: Dimensionless ratio :math:`h = B / J_1`.
        j1: Overall energy scale (default 1.0).

    Returns:
        SparsePauliOp for the Hamiltonian on *n* qubits.
    """
    j2 = -kappa * j1
    b = h * j1
    terms: list[SparsePauliOp] = []
    for i in range(n - 1):
        terms.append(-j1 * pauli_term([("X", i), ("X", i + 1)], n))
    for i in range(n - 2):
        terms.append(-j2 * pauli_term([("X", i), ("X", i + 2)], n))
    for i in range(n):
        terms.append(-b * pauli_term([("Z", i)], n))
    return SparsePauliOp.sum(terms).simplify()


def sample_parameters(n_samples: int, rng: np.random.Generator) -> list[tuple[dict, str]]:
    """Sample coupling parameters uniformly from the interior of each phase.

    Sampling regions (see Fig. 5 in the reference) are placed well inside
    each phase to avoid mislabelled points near boundaries.

    Args:
        n_samples: Number of samples to draw *per class*.
        rng: NumPy random Generator instance.

    Returns:
        List of ``(params_dict, phase_label)`` tuples.  The list contains
        *n_samples* entries for each phase in :data:`PHASE_LABELS`, in order.
    """
    samples: list[tuple[dict, str]] = []
    # ferromagnetic (I): κ ∈ (0, 0.3), h ∈ (0, 0.25)
    ks = rng.uniform(0.0, 0.3, size=n_samples)
    hs = rng.uniform(0.0, 0.25, size=n_samples)
    for k, hv in zip(ks, hs):
        samples.append(({"kappa": float(k), "h": float(hv)}, "ferromagnetic"))
    # paramagnetic (II): κ ∈ (0, 0.45), h ∈ (0.9, 1.5)
    ks = rng.uniform(0.0, 0.45, size=n_samples)
    hs = rng.uniform(0.9, 1.5, size=n_samples)
    for k, hv in zip(ks, hs):
        samples.append(({"kappa": float(k), "h": float(hv)}, "paramagnetic"))
    # floating (III): κ ∈ (0.55, 0.9), h ∈ (0.25, 0.65)
    ks = rng.uniform(0.55, 0.9, size=n_samples)
    hs = rng.uniform(0.25, 0.65, size=n_samples)
    for k, hv in zip(ks, hs):
        samples.append(({"kappa": float(k), "h": float(hv)}, "floating"))
    # antiphase (IV): κ ∈ (0.55, 0.9), h ∈ (0, 0.1)
    ks = rng.uniform(0.55, 0.9, size=n_samples)
    hs = rng.uniform(0.0, 0.1, size=n_samples)
    for k, hv in zip(ks, hs):
        samples.append(({"kappa": float(k), "h": float(hv)}, "antiphase"))
    return samples
