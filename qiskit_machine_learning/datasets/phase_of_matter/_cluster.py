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

"""Cluster Hamiltonian (periodic boundary) and phase sampler.

Reference: Bermejo et al., arXiv:2408.12739, eq. (9).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ._base import pauli_term

#: Ordered list of phase labels for the Cluster model.
PHASE_LABELS: list[str] = ["haldane", "ferromagnetic", "antiferromagnetic", "trivial"]


def build_hamiltonian(n: int, j1: float, j2: float) -> SparsePauliOp:
    r"""Cluster Hamiltonian with periodic boundary conditions (Paper eq. 9).

    .. math::

        H = \sum_{i=1}^{n}
            \left( Z_i - J_1 X_i X_{i+1} - J_2 Z_{i-1} X_i Z_{i+1} \right)

    with periodic identifications :math:`X_{n+1} \equiv X_1` and
    :math:`Z_0 \equiv Z_n`.

    Phase diagram (see Fig. 6 in the reference, axes :math:`J_1` vs
    :math:`J_2`):

    * **haldane** (I) — large positive :math:`J_1`, large negative :math:`J_2`
    * **ferromagnetic** (II) — large positive :math:`J_1` and :math:`J_2`
    * **antiferromagnetic** (III) — large negative :math:`J_1` and :math:`J_2`
    * **trivial** (IV) — both :math:`|J_1|` and :math:`|J_2|` small

    Args:
        n: Number of lattice sites (qubits).
        j1: Two-body coupling constant.
        j2: Three-body cluster coupling constant.

    Returns:
        SparsePauliOp for the Hamiltonian on *n* qubits.
    """
    terms: list[SparsePauliOp] = []
    for i in range(n):
        terms.append(pauli_term([("Z", i)], n))
        i_next = (i + 1) % n
        i_prev = (i - 1) % n
        terms.append(-j1 * pauli_term([("X", i), ("X", i_next)], n))
        terms.append(-j2 * pauli_term([("Z", i_prev), ("X", i), ("Z", i_next)], n))
    return SparsePauliOp.sum(terms).simplify()


def sample_parameters(n_samples: int, rng: np.random.Generator) -> list[tuple[dict, str]]:
    """Sample coupling parameters uniformly from the interior of each phase.

    Sampling regions (see Fig. 6 in the reference) are placed well inside
    each phase to avoid mislabelled points near boundaries.

    Args:
        n_samples: Number of samples to draw *per class*.
        rng: NumPy random Generator instance.

    Returns:
        List of ``(params_dict, phase_label)`` tuples. The list contains
        *n_samples* entries for each phase in :data:`PHASE_LABELS`, in order.
    """
    samples: list[tuple[dict, str]] = []
    # haldane (I): J1 ∈ (0.8, 2.0), J2 ∈ (-2.0, -0.8)
    j1s = rng.uniform(0.8, 2.0, size=n_samples)
    j2s = rng.uniform(-2.0, -0.8, size=n_samples)
    for j1, j2 in zip(j1s, j2s):
        samples.append(({"j1": float(j1), "j2": float(j2)}, "haldane"))
    # ferromagnetic (II): J1 ∈ (0.8, 2.5), J2 ∈ (0.8, 2.5)
    j1s = rng.uniform(0.8, 2.5, size=n_samples)
    j2s = rng.uniform(0.8, 2.5, size=n_samples)
    for j1, j2 in zip(j1s, j2s):
        samples.append(({"j1": float(j1), "j2": float(j2)}, "ferromagnetic"))
    # antiferromagnetic (III): J1 ∈ (-2.5, -0.8), J2 ∈ (-2.5, -0.8)
    j1s = rng.uniform(-2.5, -0.8, size=n_samples)
    j2s = rng.uniform(-2.5, -0.8, size=n_samples)
    for j1, j2 in zip(j1s, j2s):
        samples.append(({"j1": float(j1), "j2": float(j2)}, "antiferromagnetic"))
    # trivial (IV): |J1| < 0.15, |J2| < 0.15
    j1s = rng.uniform(-0.15, 0.15, size=n_samples)
    j2s = rng.uniform(-0.15, 0.15, size=n_samples)
    for j1, j2 in zip(j1s, j2s):
        samples.append(({"j1": float(j1), "j2": float(j2)}, "trivial"))
    return samples
