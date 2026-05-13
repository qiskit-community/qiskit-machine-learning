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

"""
Phase of Matter dataset (:mod:`phase_of_matter`)

Quantum Phase of Matter classification dataset generator.

Each supported model lives in its own module:

* :mod:`._heisenberg` — Bond-alternating XXX Heisenberg chain
* :mod:`._haldane`    — Haldane chain
* :mod:`._annni`      — Axial Next-Nearest-Neighbor Ising (ANNNI) model
* :mod:`._cluster`    — Cluster Hamiltonian

The :func:`phase_of_matter_data` function is the single public entry point.

.. currentmodule:: phase_of_matter

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   phase_of_matter_data
"""

from .phase_of_matter import phase_of_matter_data

__all__ = ["phase_of_matter_data"]
