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

"""Haldane chain Hamiltonian and phase sampler.

Reference: Bermejo et al., arXiv:2408.12739, eq. (7).
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from ._base import pauli_term

#: Ordered list of phase labels for the Haldane model.
PHASE_LABELS: list[str] = ["antiferromagnetic", "paramagnetic", "spt"]


def build_hamiltonian(n: int, h1: float, h2: float, j: float = 1.0) -> SparsePauliOp:
    r"""Haldane chain Hamiltonian (Paper eq. 7).

    .. math::

        H = -J \sum_{i=1}^{n-2} Z_i X_{i+1} Z_{i+2}
            - h_1 \sum_{i=1}^{n} X_i
            - h_2 \sum_{i=1}^{n-1} X_i X_{i+1}

    with :math:`J > 0`.

    Phase diagram (see Fig. 4 in the reference, :math:`h_1/J` vs
    :math:`h_2/J`):

    * **antiferromagnetic** — small :math:`h_1`, negative :math:`h_2`
    * **paramagnetic** — large :math:`h_1`
    * **spt** (symmetry-protected topological) — small :math:`h_1`,
      positive :math:`h_2 > 0.423` (at :math:`h_1 = 0.5`)

    Args:
        n: Number of lattice sites (qubits).
        h1: Transverse-field strength (units of *J*).
        h2: Nearest-neighbour XX coupling (units of *J*).  Positive values
            favour the SPT phase; negative values favour antiferromagnetic.
        j: Overall energy scale, default 1.0.

    Returns:
        SparsePauliOp for the Hamiltonian on *n* qubits.
    """
    terms: list[SparsePauliOp] = []
    for i in range(n - 2):
        terms.append(-j * pauli_term([("Z", i), ("X", i + 1), ("Z", i + 2)], n))
    for i in range(n):
        terms.append(-h1 * pauli_term([("X", i)], n))
    for i in range(n - 1):
        terms.append(-h2 * pauli_term([("X", i), ("X", i + 1)], n))
    return SparsePauliOp.sum(terms).simplify()


def sample_parameters(n_samples: int, rng: np.random.Generator) -> list[tuple[dict, str]]:
    """Sample coupling parameters uniformly from the interior of each phase.

    Sampling regions are chosen well away from phase boundaries (see Fig. 4
    in the reference) to ensure clean labels.

    Args:
        n_samples: Number of samples to draw *per class*.
        rng: NumPy random Generator instance.

    Returns:
        List of ``(params_dict, phase_label)`` tuples. The list contains
        *n_samples* entries for each phase in :data:`PHASE_LABELS`, in order.
    """
    samples: list[tuple[dict, str]] = []
    # antiferromagnetic: small h1, negative h2
    h1s = rng.uniform(0.0, 0.15, size=n_samples)
    h2s = rng.uniform(-0.3, -0.05, size=n_samples)
    for h1, h2 in zip(h1s, h2s):
        samples.append(({"h1": float(h1), "h2": float(h2)}, "antiferromagnetic"))
    # paramagnetic: large h1, mildly positive h2
    h1s = rng.uniform(0.9, 1.5, size=n_samples)
    h2s = rng.uniform(0.0, 0.35, size=n_samples)
    for h1, h2 in zip(h1s, h2s):
        samples.append(({"h1": float(h1), "h2": float(h2)}, "paramagnetic"))
    # spt: small h1, h2 well above the ~0.423 boundary
    h1s = rng.uniform(0.0, 0.3, size=n_samples)
    h2s = rng.uniform(0.55, 1.0, size=n_samples)
    for h1, h2 in zip(h1s, h2s):
        samples.append(({"h1": float(h1), "h2": float(h2)}, "spt"))
    return samples
