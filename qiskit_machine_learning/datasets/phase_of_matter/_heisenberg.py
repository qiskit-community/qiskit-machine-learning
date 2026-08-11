# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2024, 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bond-alternating XXX Heisenberg Hamiltonian and phase sampler.

Reference: Bermejo et al., arXiv:2408.12739, eq. (6).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ._base import pauli_term

#: Ordered list of phase labels for the Heisenberg model.
PHASE_LABELS: list[str] = ["trivial", "topological"]


def build_hamiltonian(n: int, j1: float, j2: float) -> SparsePauliOp:
    r"""Bond-alternating XXX Heisenberg Hamiltonian (Paper eq. 6).

    .. math::

        H = \sum_{i=1}^{n-1} J_i
            \left( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} \right)

    where :math:`J_i = J_1` for odd *i* and :math:`J_i = J_2` for even *i*
    (1-indexed), with :math:`J_1, J_2 \geq 0`.

    Phase diagram (thermodynamic limit):

    * **trivial** — :math:`J_2 / J_1 < 1`
    * **topological** — :math:`J_2 / J_1 > 1`

    Args:
        n: Number of lattice sites (qubits).
        j1: Coupling constant on odd-indexed bonds (1-indexed: bonds 1, 3, 5, ...).
            (:math:`J_1 \geq 0`).
        j2: Coupling constant on even-indexed bonds (1-indexed: bonds 2, 4, 6, ...).
            (:math:`J_2 \geq 0`).

    Returns:
        SparsePauliOp for the Hamiltonian on *n* qubits.
    """
    terms: list[SparsePauliOp] = []
    for i in range(n - 1):
        # i is 0-indexed; i%2==0 means 0-indexed even = 1-indexed odd bond (bond 1, 3, 5, ...)
        j = j1 if not i % 2 else j2
        for pauli in ("X", "Y", "Z"):
            terms.append(j * pauli_term([(pauli, i), (pauli, i + 1)], n))
    return SparsePauliOp.sum(terms).simplify()


def sample_parameters(n_samples: int, rng: np.random.Generator) -> list[tuple[dict, str]]:
    """Sample coupling parameters uniformly from the interior of each phase.

    Parameters are drawn well away from the phase boundary (:math:`J_2/J_1 = 1`)
    to ensure clean labels.

    Args:
        n_samples: Number of samples to draw *per class*.
        rng: NumPy random Generator instance.

    Returns:
        List of ``(params_dict, phase_label)`` tuples.  The list contains
        *n_samples* entries for each phase in :data:`PHASE_LABELS`, in order.
    """
    samples: list[tuple[dict, str]] = []
    # trivial: J2/J1 ∈ (0.0, 0.8)  —  fix J1 = 1.0
    ratios = rng.uniform(0.0, 0.8, size=n_samples)
    for r in ratios:
        samples.append(({"j1": 1.0, "j2": float(r)}, "trivial"))
    # topological: J2/J1 ∈ (1.2, 3.0)
    ratios = rng.uniform(1.2, 3.0, size=n_samples)
    for r in ratios:
        samples.append(({"j1": 1.0, "j2": float(r)}, "topological"))
    return samples
